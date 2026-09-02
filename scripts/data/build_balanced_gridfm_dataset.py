from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.data import GridFMAdapter
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend


_GRIDFM_CHUNK_CONTRACT_VERSION = 1
_RAW_COMPLETION_MARKER = ".gridfm_complete.json"
_REQUIRED_GRIDFM_DATASETS = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
    "y_bus_data.parquet",
)


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def now() -> float:
    return time.perf_counter()


def print_time(label: str, start: float) -> None:
    print(f"{label}: {time.perf_counter() - start:.2f} s")


def read_parquet_columns(
    path: Path,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
) -> pd.DataFrame:
    optional_columns = optional_columns or []
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    columns = list(dict.fromkeys(required_columns + optional_columns))
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        df = pd.read_parquet(path)
        missing_required = set(required_columns) - set(df.columns)
        if missing_required:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_required)}"
            )
        available = [column for column in columns if column in df.columns]
        return df[available].copy()


def compute_branch_summary(branch_df: pd.DataFrame) -> pd.DataFrame:
    df = branch_df.copy()
    required = {
        "scenario",
        "idx",
        "br_status",
        "pf",
        "qf",
        "pt",
        "qt",
        "rate_a",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"branch_data.parquet missing columns: {sorted(missing)}")

    df["scenario"] = df["scenario"].astype(np.int64)
    df["idx"] = df["idx"].astype(np.int64)
    br_status = df["br_status"].to_numpy(dtype=np.float32)
    in_service = br_status > 0.0
    pf = df["pf"].to_numpy(dtype=np.float64)
    qf = df["qf"].to_numpy(dtype=np.float64)
    pt = df["pt"].to_numpy(dtype=np.float64)
    qt = df["qt"].to_numpy(dtype=np.float64)
    rate_a = df["rate_a"].replace(0.0, np.nan).to_numpy(dtype=np.float64)

    s_from = np.sqrt(pf * pf + qf * qf)
    s_to = np.sqrt(pt * pt + qt * qt)
    s_max = np.maximum(s_from, s_to)
    loading = s_max / rate_a * 100.0
    loading = np.where(in_service, loading, 0.0)
    loading = np.nan_to_num(loading, nan=0.0, posinf=0.0, neginf=0.0)

    df["loading_percent"] = loading.astype(np.float32)
    df["is_in_service"] = in_service
    df["is_outaged"] = ~in_service
    df["is_overloaded"] = in_service & (df["loading_percent"].to_numpy() > 100.0)
    df["is_hard_overloaded"] = in_service & (
        df["loading_percent"].to_numpy() > 120.0
    )

    active = df[df["is_in_service"]].copy()
    active_loading_summary = active.groupby("scenario", sort=False).agg(
        max_loading_percent=("loading_percent", "max"),
        mean_loading_percent=("loading_percent", "mean"),
    )
    count_summary = df.groupby("scenario", sort=False).agg(
        num_overloaded_branches=("is_overloaded", "sum"),
        num_hard_overloaded_branches=("is_hard_overloaded", "sum"),
        num_outaged_branches=("is_outaged", "sum"),
    )

    outaged = df[df["is_outaged"]][["scenario", "idx"]].copy()
    if outaged.empty:
        outaged_ids = pd.Series(
            data=[[] for _ in range(len(count_summary))],
            index=count_summary.index,
            name="outaged_branch_ids",
        )
    else:
        outaged_ids = outaged.groupby("scenario", sort=False)["idx"].apply(
            lambda values: [int(value) for value in values.to_numpy()]
        )
        outaged_ids.name = "outaged_branch_ids"

    summary = count_summary.join(active_loading_summary, how="left")
    summary = summary.join(outaged_ids, how="left")
    summary["max_loading_percent"] = summary["max_loading_percent"].fillna(0.0)
    summary["mean_loading_percent"] = summary["mean_loading_percent"].fillna(0.0)
    summary["num_overloaded_branches"] = summary[
        "num_overloaded_branches"
    ].astype(int)
    summary["num_hard_overloaded_branches"] = summary[
        "num_hard_overloaded_branches"
    ].astype(int)
    summary["num_outaged_branches"] = summary["num_outaged_branches"].astype(int)
    summary["outaged_branch_ids"] = summary["outaged_branch_ids"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    return summary.reset_index()


def compute_bus_summary(bus_df: pd.DataFrame) -> pd.DataFrame:
    df = bus_df.copy()
    required = {"scenario", "load_scenario_idx", "Vm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"bus_data.parquet missing columns: {sorted(missing)}")

    df["scenario"] = df["scenario"].astype(np.int64)
    if "min_vm_pu" not in df.columns:
        df["min_vm_pu"] = 0.94
    if "max_vm_pu" not in df.columns:
        df["max_vm_pu"] = 1.06

    vm = df["Vm"].to_numpy(dtype=np.float64)
    vmin = df["min_vm_pu"].to_numpy(dtype=np.float64)
    vmax = df["max_vm_pu"].to_numpy(dtype=np.float64)
    low_violation = np.maximum(vmin - vm, 0.0)
    high_violation = np.maximum(vm - vmax, 0.0)
    df["low_voltage_violation"] = low_violation.astype(np.float32)
    df["high_voltage_violation"] = high_violation.astype(np.float32)
    df["total_voltage_violation_part"] = (
        low_violation + high_violation
    ).astype(np.float32)
    df["is_low_voltage"] = low_violation > 0.0
    df["is_high_voltage"] = high_violation > 0.0

    summary = df.groupby("scenario", sort=False).agg(
        load_scenario_idx=("load_scenario_idx", "first"),
        min_vm_pu=("Vm", "min"),
        max_vm_pu=("Vm", "max"),
        total_low_voltage_violation=("low_voltage_violation", "sum"),
        total_high_voltage_violation=("high_voltage_violation", "sum"),
        total_voltage_violation=("total_voltage_violation_part", "sum"),
        num_low_voltage_buses=("is_low_voltage", "sum"),
        num_high_voltage_buses=("is_high_voltage", "sum"),
    )
    summary["num_low_voltage_buses"] = summary["num_low_voltage_buses"].astype(int)
    summary["num_high_voltage_buses"] = summary["num_high_voltage_buses"].astype(int)
    return summary.reset_index()


def add_seriousness_score(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["seriousness_score"] = (
        2000.0 * df["num_hard_overloaded_branches"].astype(float)
        + 200.0 * df["num_overloaded_branches"].astype(float)
        + 5.0 * df["max_loading_percent"].astype(float)
        + 20.0 * df["num_outaged_branches"].astype(float)
        + 1000.0 * df["total_voltage_violation"].astype(float)
    )
    return df


def build_fast_summary(
    raw_dir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    bus_path = raw_dir / "bus_data.parquet"
    branch_path = raw_dir / "branch_data.parquet"
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(
            total=6,
            desc="Building summary",
            unit="stage",
            dynamic_ncols=True,
        )

    def update_progress(stage_name: str) -> None:
        if progress is not None:
            progress.set_postfix_str(stage_name)
            progress.update(1)

    try:
        t0 = now()
        bus_df = read_parquet_columns(
            bus_path,
            ["scenario", "load_scenario_idx", "Vm"],
            ["min_vm_pu", "max_vm_pu"],
        )
        print_time("Read bus_data.parquet", t0)
        update_progress("bus parquet read")

        t0 = now()
        branch_df = read_parquet_columns(
            branch_path,
            [
                "scenario",
                "idx",
                "from_bus",
                "to_bus",
                "br_status",
                "pf",
                "qf",
                "pt",
                "qt",
                "rate_a",
            ],
        )
        print_time("Read branch_data.parquet", t0)
        update_progress("branch parquet read")

        t0 = now()
        bus_summary = compute_bus_summary(bus_df)
        print_time("Compute bus summary", t0)
        update_progress("bus summary")
        del bus_df

        t0 = now()
        branch_summary = compute_branch_summary(branch_df)
        print_time("Compute branch summary", t0)
        update_progress("branch summary")
        del branch_df

        t0 = now()
        summary = pd.merge(
            bus_summary,
            branch_summary,
            on="scenario",
            how="inner",
            validate="one_to_one",
        )
        print_time("Merge summaries", t0)
        update_progress("merge")
        del bus_summary
        del branch_summary

        t0 = now()
        summary = add_seriousness_score(summary)
        print_time("Add seriousness score", t0)
        update_progress("score")
        return summary
    finally:
        if progress is not None:
            progress.close()


def make_output(selected: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": selected["scenario"].astype(int),
            "load_scenario_idx": selected["load_scenario_idx"],
            "max_loading_percent": selected["max_loading_percent"],
            "mean_loading_percent": selected["mean_loading_percent"],
            "num_overloaded_branches": selected["num_overloaded_branches"].astype(int),
            "num_hard_overloaded_branches": selected[
                "num_hard_overloaded_branches"
            ].astype(int),
            "num_outaged_branches": selected["num_outaged_branches"].astype(int),
            "total_voltage_violation": selected["total_voltage_violation"],
            "seriousness_score": selected["seriousness_score"],
            "outaged_branch_ids": selected["outaged_branch_ids"].astype(str),
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _fingerprint_mapping(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "contract_version": _GRIDFM_CHUNK_CONTRACT_VERSION,
        "gridfm_datakit_version": _package_version("gridfm-datakit"),
        "dataset_name": str(args.dataset_name),
        "network_name": str(args.network_name),
        "network_source": str(args.network_source),
        "raw_network_dir_name": (
            None
            if args.raw_network_dir_name is None
            else str(args.raw_network_dir_name)
        ),
        "chunk_size": int(args.chunk_size),
        "seed_start": int(args.seed_start),
        "num_processes": int(args.num_processes),
        "sigma": float(args.sigma),
        "global_range": float(args.global_range),
        "max_scaling_factor": float(args.max_scaling_factor),
        "step_size": float(args.step_size),
        "start_scaling_factor": float(args.start_scaling_factor),
        "topology_variants": int(args.topology_variants),
        "topology_k": int(args.topology_k),
        "generation_perturbation_type": str(args.generation_perturbation_type),
        "generation_perturbation_sigma": float(
            args.generation_perturbation_sigma
        ),
        "admittance_perturbation_type": str(args.admittance_perturbation_type),
        "admittance_perturbation_sigma": float(
            args.admittance_perturbation_sigma
        ),
        "min_loading": float(args.min_loading),
        "max_loading": float(args.max_loading),
        "simple_min_loading": float(args.simple_min_loading),
        "simple_max_loading": float(args.simple_max_loading),
        "simple_max_hard": int(args.simple_max_hard),
        "simple_max_overloaded": int(args.simple_max_overloaded),
        "medium_min_loading": float(args.medium_min_loading),
        "medium_max_loading": float(args.medium_max_loading),
        "medium_max_hard": int(args.medium_max_hard),
        "medium_max_overloaded": int(args.medium_max_overloaded),
        "hard_min_loading": float(args.hard_min_loading),
        "hard_min_hard": int(args.hard_min_hard),
        "canonical_physics_fingerprint": DEFAULT_PHYSICS_CONFIG.fingerprint(),
    }


def dataset_contract_fingerprint(args: argparse.Namespace) -> str:
    return _fingerprint_mapping(dataset_contract_payload(args))


def _completion_marker_matches(
    path: Path,
    *,
    stage: str,
    contract_fingerprint: str,
    chunk_index: int,
) -> bool:
    marker = load_json(path)
    return bool(
        marker is not None
        and marker.get("stage") == stage
        and marker.get("contract_fingerprint") == contract_fingerprint
        and marker.get("chunk_index") == int(chunk_index)
    )


def _write_completion_marker(
    path: Path,
    *,
    stage: str,
    contract_fingerprint: str,
    chunk_index: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "contract_version": _GRIDFM_CHUNK_CONTRACT_VERSION,
        "contract_fingerprint": contract_fingerprint,
        "chunk_index": int(chunk_index),
    }
    if extra:
        payload.update(extra)
    save_json_atomic(path, payload)


def _dataset_path_has_data(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    try:
        return any(
            child.is_file() and child.stat().st_size > 0
            for child in path.rglob("*.parquet")
        )
    except OSError:
        return False


def gridfm_raw_artifacts_are_usable(raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return False
    if not all(
        _dataset_path_has_data(raw_dir / name)
        for name in _REQUIRED_GRIDFM_DATASETS
    ):
        return False
    try:
        return int((raw_dir / "n_scenarios.txt").read_text(encoding="utf-8").strip()) > 0
    except Exception:
        return False


def raw_completion_is_current(
    raw_dir: Path,
    *,
    contract_fingerprint: str,
    chunk_index: int,
) -> bool:
    return (
        gridfm_raw_artifacts_are_usable(raw_dir)
        and _completion_marker_matches(
            raw_dir / _RAW_COMPLETION_MARKER,
            stage="gridfm_raw",
            contract_fingerprint=contract_fingerprint,
            chunk_index=chunk_index,
        )
    )


def candidate_completion_marker_path(path: Path) -> Path:
    return path.with_suffix(".complete.json")


def candidate_manifest_is_current(
    path: Path,
    expected_contract_fingerprint: str | None = None,
    chunk_index: int | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    if not {"canonical_pf_ok", "gridfm_difficulty_class"}.issubset(columns):
        return False

    marker = load_json(candidate_completion_marker_path(path))
    if marker is None or marker.get("stage") != "canonical_candidates":
        return False
    if (
        expected_contract_fingerprint is not None
        and marker.get("contract_fingerprint") != expected_contract_fingerprint
    ):
        return False
    if chunk_index is not None and marker.get("chunk_index") != int(chunk_index):
        return False
    return True


def _latest_activity_timestamp(paths: Sequence[Path]) -> float | None:
    latest: float | None = None
    for path in paths:
        try:
            if not path.exists():
                continue
            timestamp = float(path.stat().st_mtime)
        except OSError:
            continue
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
    try:
        process.wait(timeout=10)
    except Exception:
        pass


def run_command(
    command: Sequence[str],
    log_path: Path,
    *,
    activity_paths: Sequence[Path] = (),
    inactivity_timeout_sec: float = 0.0,
) -> None:
    ensure_dir(log_path.parent)
    command_list = [str(argument) for argument in command]
    printable_command = subprocess.list2cmdline(command_list)
    print("Running command:", flush=True)
    print(printable_command, flush=True)
    print("GridFM output is streamed live below.", flush=True)

    log_path.write_text("runtime command omitted\n", encoding="utf-8")
    popen_kwargs: dict[str, Any] = {"shell": False}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command_list, **popen_kwargs)
    started = time.monotonic()
    last_activity = started
    last_timestamp = _latest_activity_timestamp(activity_paths)

    while True:
        return_code = process.poll()
        if return_code is not None:
            break

        current_timestamp = _latest_activity_timestamp(activity_paths)
        if current_timestamp is not None and (
            last_timestamp is None or current_timestamp > last_timestamp
        ):
            last_timestamp = current_timestamp
            last_activity = time.monotonic()

        if inactivity_timeout_sec > 0.0 and (
            time.monotonic() - last_activity >= float(inactivity_timeout_sec)
        ):
            message = (
                "GridFM produced no progress-file activity for "
                f"{float(inactivity_timeout_sec):.0f} seconds. "
                "Terminating the full process tree so the chunk can be retried."
            )
            print(f"\n{message}", flush=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"watchdog: {message}\n")
            _terminate_process_tree(process)
            raise RuntimeError(message)
        time.sleep(1.0)

    elapsed = time.monotonic() - started
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"exit_code: {return_code}\n")
        log_file.write(f"elapsed_seconds: {elapsed:.3f}\n")

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. "
            "See the live console output and GridFM raw/error logs."
        )


@dataclass(frozen=True)
class ClassTargets:
    simple: int
    medium: int
    hard: int

    @property
    def total(self) -> int:
        return int(self.simple + self.medium + self.hard)


def compute_class_targets(
    target_total: int,
    simple_fraction: float,
    medium_fraction: float,
    hard_fraction: float,
) -> ClassTargets:
    simple = int(round(int(target_total) * float(simple_fraction)))
    hard = int(round(int(target_total) * float(hard_fraction)))
    medium = int(target_total) - simple - hard
    if medium < 0:
        raise ValueError("Class fractions are invalid: medium target became negative.")
    return ClassTargets(simple=simple, medium=medium, hard=hard)


def make_generation_perturbation_text(
    perturbation_type: str,
    sigma: float,
) -> str:
    perturbation_type = str(perturbation_type).strip().lower()
    if perturbation_type == "none":
        return 'generation_perturbation:\n  type: "none"\n'
    if perturbation_type == "cost_permutation":
        return 'generation_perturbation:\n  type: "cost_permutation"\n'
    if perturbation_type == "cost_perturbation":
        if float(sigma) <= 0.0:
            raise ValueError(
                "generation_perturbation_sigma must be > 0 "
                "when generation_perturbation_type='cost_perturbation'."
            )
        return (
            'generation_perturbation:\n'
            '  type: "cost_perturbation"\n'
            f"  sigma: {float(sigma)}\n"
        )
    raise ValueError(
        "Unsupported generation_perturbation_type: "
        f"{perturbation_type}. Expected one of: none, cost_permutation, "
        "cost_perturbation."
    )


def make_admittance_perturbation_text(
    perturbation_type: str,
    sigma: float,
) -> str:
    perturbation_type = str(perturbation_type).strip().lower()
    if perturbation_type == "none":
        return 'admittance_perturbation:\n  type: "none"\n'
    if perturbation_type == "random_perturbation":
        if float(sigma) <= 0.0:
            raise ValueError(
                "admittance_perturbation_sigma must be > 0 "
                "when admittance_perturbation_type='random_perturbation'."
            )
        return (
            'admittance_perturbation:\n'
            '  type: "random_perturbation"\n'
            f"  sigma: {float(sigma)}\n"
        )
    raise ValueError(
        "Unsupported admittance_perturbation_type: "
        f"{perturbation_type}. Expected one of: none, random_perturbation."
    )


def make_gridfm_config_text(
    *,
    network_name: str,
    network_source: str,
    data_dir: Path,
    scenarios: int,
    seed: int,
    num_processes: int,
    sigma: float,
    global_range: float,
    max_scaling_factor: float,
    step_size: float,
    start_scaling_factor: float,
    topology_variants: int,
    topology_k: int,
    generation_perturbation_type: str,
    generation_perturbation_sigma: float,
    admittance_perturbation_type: str,
    admittance_perturbation_sigma: float,
) -> str:
    data_dir_str = str(data_dir).replace("\\", "/")
    generation_text = make_generation_perturbation_text(
        generation_perturbation_type,
        generation_perturbation_sigma,
    )
    admittance_text = make_admittance_perturbation_text(
        admittance_perturbation_type,
        admittance_perturbation_sigma,
    )
    return f'''network:
  name: "{network_name}"
  source: "{network_source}"

load:
  generator: "agg_load_profile"
  agg_profile: "default"
  scenarios: {int(scenarios)}
  sigma: {float(sigma)}
  change_reactive_power: true
  global_range: {float(global_range)}
  max_scaling_factor: {float(max_scaling_factor)}
  step_size: {float(step_size)}
  start_scaling_factor: {float(start_scaling_factor)}

topology_perturbation:
  type: "random"
  k: {int(topology_k)}
  n_topology_variants: {int(topology_variants)}
  elements: [branch]

{generation_text}
{admittance_text}
settings:
  num_processes: {int(num_processes)}
  data_dir: "{data_dir_str}"
  large_chunk_size: 1000
  overwrite: true
  mode: "pf"
  include_dc_res: false
  enable_solver_logs: true
  pf_fast: true
  dcpf_fast: true
  max_iter: 200
  seed: {int(seed)}
'''


def classify_summary(
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    df = summary.copy()
    df["difficulty_class"] = "unused"
    max_loading = df["max_loading_percent"].astype(float)
    overloaded = df["num_overloaded_branches"].astype(int)
    hard = df["num_hard_overloaded_branches"].astype(int)
    outaged = df["num_outaged_branches"].astype(int)
    base_valid = (
        (max_loading >= float(args.min_loading))
        & (max_loading <= float(args.max_loading))
        & (overloaded > 0)
        & (outaged > 0)
    )
    hard_mask = base_valid & (
        (max_loading >= float(args.hard_min_loading))
        | (hard >= int(args.hard_min_hard))
    )
    medium_mask = (
        base_valid
        & ~hard_mask
        & (max_loading >= float(args.medium_min_loading))
        & (max_loading < float(args.medium_max_loading))
        & (hard <= int(args.medium_max_hard))
        & (overloaded <= int(args.medium_max_overloaded))
    )
    simple_mask = (
        base_valid
        & ~hard_mask
        & ~medium_mask
        & (max_loading >= float(args.simple_min_loading))
        & (max_loading < float(args.simple_max_loading))
        & (hard <= int(args.simple_max_hard))
        & (overloaded <= int(args.simple_max_overloaded))
    )
    df.loc[simple_mask, "difficulty_class"] = "simple"
    df.loc[medium_mask, "difficulty_class"] = "medium"
    df.loc[hard_mask, "difficulty_class"] = "hard"
    return df


def add_source_columns(
    summary: pd.DataFrame,
    *,
    chunk_name: str,
) -> pd.DataFrame:
    """Persist only a logical chunk identity, never a filesystem location."""
    df = summary.copy()
    df["source_chunk"] = str(chunk_name)
    df["source_scenario_id"] = df["scenario"].astype(int)
    return df


_CANONICAL_SUMMARY_COLUMNS = (
    "max_loading_percent",
    "mean_loading_percent",
    "num_overloaded_branches",
    "num_hard_overloaded_branches",
    "num_outaged_branches",
    "min_vm_pu",
    "max_vm_pu",
    "total_low_voltage_violation",
    "total_high_voltage_violation",
    "total_voltage_violation",
    "num_low_voltage_buses",
    "num_high_voltage_buses",
)


def evaluate_canonical_candidates(
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    raw_dir: Path | str = Path("raw"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        evaluated = candidates.copy()
        evaluated["gridfm_difficulty_class"] = pd.Series(dtype=str)
        evaluated["canonical_pf_ok"] = pd.Series(dtype=bool)
        evaluated["canonical_failure_kind"] = pd.Series(dtype=str)
        return evaluated, evaluated.copy()

    scenario_ids = sorted(candidates["source_scenario_id"].astype(int).unique())
    adapter = GridFMAdapter(Path(raw_dir), scenario_ids=scenario_ids)
    backend = GridFMPowerFlowBackend(adapter, enable_cache=False)
    evaluated_rows: list[dict[str, Any]] = []

    print(f"\nCanonical PF validation: {len(candidates)} candidates")
    for row in candidates.itertuples(index=False):
        record = row._asdict()
        record.pop("source_raw_dir", None)
        record["gridfm_difficulty_class"] = str(record["difficulty_class"])
        scenario_id = int(record["source_scenario_id"])
        result = backend.run_power_flow(scenario_id, None)
        record["canonical_pf_ok"] = bool(result.success)
        record["canonical_failure_kind"] = (
            str(result.failure_kind.value)
            if result.failure_kind is not None
            else ""
        )
        if result.success and result.next_state is not None:
            state = result.next_state
            for column in _CANONICAL_SUMMARY_COLUMNS:
                if column in state.metrics:
                    record[column] = state.metrics[column]
            record["outaged_branch_ids"] = list(state.outaged_branch_ids)
        evaluated_rows.append(record)

    evaluated = pd.DataFrame(evaluated_rows)
    valid = evaluated[evaluated["canonical_pf_ok"]].copy()
    if not valid.empty:
        valid = classify_summary(valid, args)
        valid = add_seriousness_score(valid)
        valid = valid[
            valid["difficulty_class"].isin(["simple", "medium", "hard"])
        ].copy()
    print(
        "Canonical PF accepted "
        f"{int(evaluated['canonical_pf_ok'].sum())}/{len(evaluated)} "
        "preclassified candidates."
    )
    return evaluated, valid


def _normalize_outage_identity(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [
                part.strip()
                for part in text.strip("[]").split(",")
                if part.strip()
            ]
        if not isinstance(parsed, (list, tuple)):
            parsed = [parsed]
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        parsed = list(value)
    elif pd.isna(value):
        parsed = []
    else:
        parsed = [value]
    return tuple(sorted(int(branch_id) for branch_id in parsed))


def deduplicate_gridfm_variants(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    required = {"source_chunk", "load_scenario_idx", "outaged_branch_ids"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            "Cannot deduplicate GridFM candidates; missing columns: "
            f"{sorted(missing)}"
        )

    df = candidates.drop(columns=["source_raw_dir"], errors="ignore").copy()
    load_scenario_idx = pd.to_numeric(
        df["load_scenario_idx"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(load_scenario_idx).all():
        raise ValueError("load_scenario_idx must contain only finite values.")
    df["_dedup_load_scenario_idx"] = load_scenario_idx
    df["_dedup_outage_ids"] = df["outaged_branch_ids"].map(
        _normalize_outage_identity
    )
    before = len(df)
    df = df.drop_duplicates(
        subset=["source_chunk", "_dedup_load_scenario_idx", "_dedup_outage_ids"],
        keep="first",
    ).copy()
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate GridFM topology variants.")
    return df.drop(
        columns=["_dedup_load_scenario_idx", "_dedup_outage_ids"],
        errors="ignore",
    ).reset_index(drop=True)


def select_balanced_manifest(
    candidates: pd.DataFrame,
    targets: ClassTargets,
    seed: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    df = deduplicate_gridfm_variants(candidates)
    df = df[df["difficulty_class"].isin(["simple", "medium", "hard"])].copy()
    if df.empty:
        return df
    df = df.drop_duplicates(
        subset=["source_chunk", "source_scenario_id"], keep="first"
    ).copy()
    rng = np.random.default_rng(int(seed))
    df["_random_tiebreak"] = rng.random(len(df))
    selected_parts: list[pd.DataFrame] = []
    class_to_target = {
        "simple": int(targets.simple),
        "medium": int(targets.medium),
        "hard": int(targets.hard),
    }
    for class_name, target_count in class_to_target.items():
        part = df[df["difficulty_class"] == class_name].copy()
        if part.empty or target_count <= 0:
            continue
        if class_name == "simple":
            sort_cols = [
                "max_loading_percent",
                "num_overloaded_branches",
                "_random_tiebreak",
            ]
        elif class_name == "medium":
            sort_cols = [
                "seriousness_score",
                "max_loading_percent",
                "num_overloaded_branches",
                "_random_tiebreak",
            ]
        else:
            sort_cols = [
                "num_hard_overloaded_branches",
                "seriousness_score",
                "max_loading_percent",
                "_random_tiebreak",
            ]
        part = part.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        selected_parts.append(part.head(target_count))
    if not selected_parts:
        return pd.DataFrame()
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["_class_order"] = selected["difficulty_class"].map(
        {"simple": 0, "medium": 1, "hard": 2}
    ).astype(int)
    selected = selected.sort_values(
        ["_class_order", "source_chunk", "source_scenario_id"]
    ).reset_index(drop=True)
    selected["global_scenario_id"] = np.arange(len(selected), dtype=int)
    return selected.drop(
        columns=["_class_order", "_random_tiebreak"], errors="ignore"
    )


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "difficulty_class" not in df.columns:
        return {"simple": 0, "medium": 0, "hard": 0}
    counts = df["difficulty_class"].value_counts().to_dict()
    return {
        "simple": int(counts.get("simple", 0)),
        "medium": int(counts.get("medium", 0)),
        "hard": int(counts.get("hard", 0)),
    }


def compute_max_balanced_targets(
    candidates: pd.DataFrame,
    simple_fraction: float,
    medium_fraction: float,
    hard_fraction: float,
) -> ClassTargets:
    fractions = {
        "simple": float(simple_fraction),
        "medium": float(medium_fraction),
        "hard": float(hard_fraction),
    }
    if any(value < 0.0 for value in fractions.values()):
        raise ValueError("Class fractions must be non-negative.")
    fraction_sum = sum(fractions.values())
    if not np.isclose(fraction_sum, 1.0, atol=1e-9):
        raise ValueError(
            f"Class fractions must sum to 1.0. Received sum={fraction_sum}"
        )
    available = class_counts(candidates)
    limits = [
        available[class_name] / fraction
        for class_name, fraction in fractions.items()
        if fraction > 0.0
    ]
    if not limits:
        return ClassTargets(0, 0, 0)
    maximum_total = int(np.floor(min(limits)))
    while maximum_total > 0:
        targets = compute_class_targets(
            maximum_total,
            fractions["simple"],
            fractions["medium"],
            fractions["hard"],
        )
        if (
            targets.simple <= available["simple"]
            and targets.medium <= available["medium"]
            and targets.hard <= available["hard"]
        ):
            return targets
        maximum_total -= 1
    return ClassTargets(0, 0, 0)


def compute_final_targets(
    candidates: pd.DataFrame,
    requested: ClassTargets,
    simple_fraction: float,
    medium_fraction: float,
    hard_fraction: float,
) -> ClassTargets:
    if quotas_met(candidates, requested):
        return requested
    return compute_max_balanced_targets(
        candidates,
        simple_fraction,
        medium_fraction,
        hard_fraction,
    )


def quotas_met(selected: pd.DataFrame, targets: ClassTargets) -> bool:
    counts = class_counts(selected)
    return (
        counts["simple"] >= targets.simple
        and counts["medium"] >= targets.medium
        and counts["hard"] >= targets.hard
    )


def remap_scenario_column(
    df: pd.DataFrame,
    mapping: dict[int, int],
) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = out["scenario"].astype(int).map(mapping)
    if out["scenario"].isna().any():
        raise RuntimeError("Some scenario values were not remapped correctly.")
    out["scenario"] = out["scenario"].astype(int)
    return out


def _chunk_raw_dir(
    chunks_dir: Path,
    source_chunk: object,
    raw_network_dir_name: str,
) -> Path:
    chunk = str(source_chunk).strip()
    if not chunk or chunk in {".", ".."} or "/" in chunk or "\\" in chunk:
        raise ValueError(f"Invalid source_chunk: {source_chunk!r}")
    return expected_raw_dir(chunks_dir / chunk, raw_network_dir_name)


def merge_raw_parquet_files(
    selected: pd.DataFrame,
    output_raw_dir: Path,
    *,
    chunks_dir: Path | None = None,
    raw_network_dir_name: str | None = None,
) -> None:
    ensure_dir(output_raw_dir)
    if selected.empty:
        raise RuntimeError("Cannot merge raw files: selected manifest is empty.")

    if chunks_dir is None or raw_network_dir_name is None:
        legacy = selected.get("source_raw_dir")
        if legacy is None:
            raise ValueError(
                "chunks_dir and raw_network_dir_name are required for portable manifests."
            )
        grouped = [
            (Path(str(raw_dir)), group)
            for raw_dir, group in selected.groupby("source_raw_dir", sort=False)
        ]
    else:
        grouped = [
            (
                _chunk_raw_dir(chunks_dir, source_chunk, raw_network_dir_name),
                group,
            )
            for source_chunk, group in selected.groupby("source_chunk", sort=False)
        ]

    first_raw_dir = grouped[0][0]
    parquet_names = sorted(path.name for path in first_raw_dir.glob("*.parquet"))
    if not parquet_names:
        raise RuntimeError(f"No parquet files found in first raw dir: {first_raw_dir}")

    print("\nMerging raw parquet files...")
    for parquet_name in parquet_names:
        parts: list[pd.DataFrame] = []
        copied_static_file = False
        print(f"  {parquet_name}")
        for raw_dir, group in grouped:
            src_path = raw_dir / parquet_name
            if not src_path.exists():
                print(f"    missing in {raw_dir}, skipping")
                continue
            df = pd.read_parquet(src_path)
            if "scenario" not in df.columns:
                if not copied_static_file:
                    df.to_parquet(output_raw_dir / parquet_name, index=False)
                    copied_static_file = True
                continue
            source_ids = set(int(value) for value in group["source_scenario_id"].values)
            mapping = {
                int(row.source_scenario_id): int(row.global_scenario_id)
                for row in group.itertuples(index=False)
            }
            filtered = df[df["scenario"].astype(int).isin(source_ids)].copy()
            if filtered.empty:
                continue
            parts.append(remap_scenario_column(filtered, mapping))
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged = merged.sort_values("scenario").reset_index(drop=True)
            merged.to_parquet(output_raw_dir / parquet_name, index=False)

    (output_raw_dir / "n_scenarios.txt").write_text(
        str(int(len(selected))), encoding="utf-8"
    )


def build_transitions_from_manifest(selected: pd.DataFrame) -> pd.DataFrame:
    df = selected.drop(columns=["source_raw_dir"], errors="ignore").copy()
    df["scenario"] = df["global_scenario_id"].astype(int)
    transitions = make_output(df)
    transitions["difficulty_class"] = df["difficulty_class"].values
    transitions["source_chunk"] = df["source_chunk"].values
    transitions["source_scenario_id"] = df["source_scenario_id"].astype(int).values
    front = [
        "scenario_id",
        "difficulty_class",
        "source_chunk",
        "source_scenario_id",
    ]
    rest = [column for column in transitions.columns if column not in front]
    return transitions[front + rest].copy()


def stratified_split(
    transitions: pd.DataFrame,
    train_fraction: float,
    self_play_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    self_play_parts: list[pd.DataFrame] = []
    for _, group in transitions.groupby("difficulty_class", sort=False):
        group = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        train_end = int(round(len(group) * float(train_fraction)))
        self_play_start = int(
            round(len(group) * (1.0 - float(self_play_fraction)))
        )
        train_parts.append(group.iloc[:train_end].copy())
        val_parts.append(group.iloc[train_end:self_play_start].copy())
        self_play_parts.append(group.iloc[self_play_start:].copy())
    train = pd.concat(train_parts, ignore_index=True).sort_values("scenario_id")
    val = pd.concat(val_parts, ignore_index=True).sort_values("scenario_id")
    self_play = pd.concat(self_play_parts, ignore_index=True).sort_values("scenario_id")
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        self_play.reset_index(drop=True),
    )


def chunk_name(index: int) -> str:
    return f"chunk{int(index):02d}"


def expected_raw_dir(chunk_dir: Path, raw_network_dir_name: str) -> Path:
    return chunk_dir / str(raw_network_dir_name) / "raw"


def _gridfm_activity_paths(raw_dir: Path) -> tuple[Path, ...]:
    return (
        raw_dir / "tqdm.log",
        raw_dir / "error.log",
        raw_dir / "args.log",
        raw_dir / "scenarios_agg_load_profile.parquet",
        raw_dir / "n_scenarios.txt",
    )


def _read_generated_scenario_count(raw_dir: Path) -> int:
    try:
        return int((raw_dir / "n_scenarios.txt").read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def _drop_legacy_manifest_paths(manifest_dir: Path) -> None:
    if not manifest_dir.is_dir():
        return
    for path in manifest_dir.glob("*.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "source_raw_dir" not in frame.columns:
            continue
        frame = frame.drop(columns=["source_raw_dir"])
        temp = path.with_name(path.name + ".tmp")
        frame.to_csv(temp, index=False)
        temp.replace(path)

    summary_path = manifest_dir / "summary.json"
    payload = load_json(summary_path)
    if payload is not None and "paths" in payload:
        payload.pop("paths", None)
        save_json_atomic(summary_path, payload)


def _remove_path_bearing_build_files(paths: dict[str, Path]) -> None:
    for directory_key in ("configs_dir", "logs_dir"):
        directory = paths[directory_key]
        if directory.exists():
            shutil.rmtree(directory)
        ensure_dir(directory)
    chunks_dir = paths["chunks_dir"]
    if chunks_dir.is_dir():
        for path in chunks_dir.rglob("*.log"):
            path.unlink(missing_ok=True)


def process_chunk(
    *,
    chunk_index: int,
    args: argparse.Namespace,
    paths: dict[str, Path],
    contract_fingerprint: str,
) -> pd.DataFrame:
    name = chunk_name(chunk_index)
    chunk_dir = paths["chunks_dir"] / name
    raw_network_dir_name = (
        str(args.raw_network_dir_name)
        if args.raw_network_dir_name is not None
        else str(args.network_name)
    )
    raw_dir = expected_raw_dir(chunk_dir, raw_network_dir_name)
    config_path = paths["configs_dir"] / f"{name}.yaml"
    log_path = paths["logs_dir"] / f"{name}.runner.log"
    config_text = make_gridfm_config_text(
        network_name=str(args.network_name),
        network_source=str(args.network_source),
        data_dir=chunk_dir,
        scenarios=int(args.chunk_size),
        seed=int(args.seed_start) + int(chunk_index),
        num_processes=int(args.num_processes),
        sigma=float(args.sigma),
        global_range=float(args.global_range),
        max_scaling_factor=float(args.max_scaling_factor),
        step_size=float(args.step_size),
        start_scaling_factor=float(args.start_scaling_factor),
        topology_variants=int(args.topology_variants),
        topology_k=int(args.topology_k),
        generation_perturbation_type=str(args.generation_perturbation_type),
        generation_perturbation_sigma=float(args.generation_perturbation_sigma),
        admittance_perturbation_type=str(args.admittance_perturbation_type),
        admittance_perturbation_sigma=float(args.admittance_perturbation_sigma),
    )
    write_text(config_path, config_text)

    raw_current = bool(args.resume) and raw_completion_is_current(
        raw_dir,
        contract_fingerprint=contract_fingerprint,
        chunk_index=chunk_index,
    )
    if raw_current:
        print(f"\n{name}: verified completed GridFM raw chunk - reusing it")
    else:
        if chunk_dir.exists():
            reason = (
                "stale or incomplete resume chunk"
                if args.resume
                else "fresh generation requested"
            )
            print(f"\n{name}: removing {reason}: {chunk_dir}")
            shutil.rmtree(chunk_dir)
        command = [
            sys.executable,
            "-m",
            "gridfm_datakit.cli",
            "generate",
            str(config_path),
        ]
        attempts = 1 + max(int(args.gridfm_retries), 0)
        for attempt in range(1, attempts + 1):
            ensure_dir(chunk_dir)
            write_text(config_path, config_text)
            print("\n" + "=" * 100)
            print(f"Generating {name} | attempt {attempt}/{attempts}")
            print("=" * 100)
            try:
                run_command(
                    command,
                    log_path=log_path,
                    activity_paths=_gridfm_activity_paths(raw_dir),
                    inactivity_timeout_sec=float(
                        args.gridfm_inactivity_timeout_sec
                    ),
                )
                if not gridfm_raw_artifacts_are_usable(raw_dir):
                    raise RuntimeError(
                        "GridFM exited successfully but required raw artifacts "
                        f"are incomplete: {raw_dir}"
                    )
                _write_completion_marker(
                    raw_dir / _RAW_COMPLETION_MARKER,
                    stage="gridfm_raw",
                    contract_fingerprint=contract_fingerprint,
                    chunk_index=chunk_index,
                    extra={
                        "requested_scenarios": int(args.chunk_size),
                        "generated_scenarios": _read_generated_scenario_count(raw_dir),
                    },
                )
                break
            except Exception as exc:
                if attempt >= attempts:
                    raise
                print(f"\n{name}: GridFM attempt {attempt} failed: {exc}")
                if chunk_dir.exists():
                    shutil.rmtree(chunk_dir)
                time.sleep(min(2.0 * attempt, 5.0))

    if not raw_completion_is_current(
        raw_dir,
        contract_fingerprint=contract_fingerprint,
        chunk_index=chunk_index,
    ):
        raise RuntimeError(
            f"GridFM raw chunk has no valid completion marker: {raw_dir}"
        )

    print("\n" + "=" * 100)
    print(f"Summarizing {name}")
    print("=" * 100)
    t0 = now()
    summary = build_fast_summary(raw_dir=raw_dir, show_progress=True)
    print_time(f"Build summary for {name}", t0)
    summary = add_source_columns(classify_summary(summary, args), chunk_name=name)
    gridfm_candidates = summary[
        summary["difficulty_class"].isin(["simple", "medium", "hard"])
    ].copy()
    evaluated, candidates = evaluate_canonical_candidates(
        gridfm_candidates,
        args,
        raw_dir=raw_dir,
    )
    candidates = deduplicate_gridfm_variants(candidates)

    ensure_dir(paths["manifest_dir"])
    summary_path = paths["manifest_dir"] / f"{name}_summary.csv"
    canonical_path = paths["manifest_dir"] / f"{name}_canonical.csv"
    candidates_path = paths["manifest_dir"] / f"{name}_candidates.csv"
    summary.to_csv(summary_path, index=False)
    evaluated.to_csv(canonical_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    _write_completion_marker(
        candidate_completion_marker_path(candidates_path),
        stage="canonical_candidates",
        contract_fingerprint=contract_fingerprint,
        chunk_index=chunk_index,
        extra={
            "gridfm_candidates": int(len(gridfm_candidates)),
            "canonical_candidates": int(len(candidates)),
        },
    )

    config_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    for path in raw_dir.rglob("*.log"):
        path.unlink(missing_ok=True)

    gridfm_counts = class_counts(gridfm_candidates)
    counts = class_counts(candidates)
    print("\nChunk candidates:")
    print(
        f"  GridFM:    simple={gridfm_counts['simple']}, "
        f"medium={gridfm_counts['medium']}, hard={gridfm_counts['hard']}"
    )
    print(
        f"  canonical: simple={counts['simple']}, "
        f"medium={counts['medium']}, hard={counts['hard']}"
    )
    print(f"  total canonical: {len(candidates)}")
    return candidates


def load_existing_candidates(
    manifest_dir: Path,
    *,
    contract_fingerprint: str,
) -> pd.DataFrame:
    _drop_legacy_manifest_paths(manifest_dir)
    files: list[Path] = []
    for path in sorted(manifest_dir.glob("chunk*_candidates.csv")):
        try:
            chunk_index = int(path.stem.split("_", 1)[0].replace("chunk", ""))
        except (TypeError, ValueError):
            continue
        if candidate_manifest_is_current(
            path,
            expected_contract_fingerprint=contract_fingerprint,
            chunk_index=chunk_index,
        ):
            files.append(path)
    if not files:
        return pd.DataFrame()
    return deduplicate_gridfm_variants(
        pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    )


def make_paths(output_root: Path) -> dict[str, Path]:
    return {
        "root": output_root,
        "chunks_dir": output_root / "chunks",
        "raw_dir": output_root / "raw",
        "configs_dir": output_root / "configs",
        "logs_dir": output_root / "logs",
        "manifest_dir": output_root / "manifest",
        "transitions_dir": output_root / "transitions",
    }


def write_outputs(
    *,
    selected: pd.DataFrame,
    all_candidates: pd.DataFrame,
    targets: ClassTargets,
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    ensure_dir(paths["manifest_dir"])
    ensure_dir(paths["transitions_dir"])
    all_candidates = all_candidates.drop(columns=["source_raw_dir"], errors="ignore")
    selected = selected.drop(columns=["source_raw_dir"], errors="ignore")

    all_candidates_path = paths["manifest_dir"] / "all_candidates.csv"
    selected_manifest_path = paths["manifest_dir"] / "selected_manifest.csv"
    class_summary_path = paths["manifest_dir"] / "class_summary.csv"
    all_candidates.to_csv(all_candidates_path, index=False)
    selected.to_csv(selected_manifest_path, index=False)

    transitions = build_transitions_from_manifest(selected)
    transitions_path = paths["transitions_dir"] / "transitions_balanced.csv"
    train_path = paths["transitions_dir"] / "transitions_train.csv"
    val_path = paths["transitions_dir"] / "transitions_val.csv"
    self_play_path = paths["transitions_dir"] / "transitions_self_play.csv"
    transitions.to_csv(transitions_path, index=False)
    train, val, self_play = stratified_split(
        transitions,
        float(args.train_fraction),
        float(args.self_play_fraction),
        int(args.split_seed),
    )
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    self_play.to_csv(self_play_path, index=False)

    summary_rows = []
    for class_name in ["simple", "medium", "hard"]:
        available = int((all_candidates["difficulty_class"] == class_name).sum())
        selected_count = int((selected["difficulty_class"] == class_name).sum())
        target = {
            "simple": targets.simple,
            "medium": targets.medium,
            "hard": targets.hard,
        }[class_name]
        summary_rows.append(
            {
                "difficulty_class": class_name,
                "target": int(target),
                "available": available,
                "selected": selected_count,
                "missing": int(max(target - selected_count, 0)),
            }
        )
    class_summary = pd.DataFrame(summary_rows)
    class_summary.to_csv(class_summary_path, index=False)
    print("\n" + "=" * 100)
    print("Balanced dataset summary")
    print("=" * 100)
    print(class_summary.to_string(index=False))
    print("\nTransitions:")
    print(f"  all:       {transitions_path} ({len(transitions)})")
    print(f"  train:     {train_path} ({len(train)})")
    print(f"  val:       {val_path} ({len(val)})")
    print(f"  self-play: {self_play_path} ({len(self_play)})")

    save_json(
        paths["manifest_dir"] / "summary.json",
        {
            "dataset_name": str(args.dataset_name),
            "target_total": int(args.target_total),
            "selected_total": int(len(selected)),
            "targets": {
                "simple": int(targets.simple),
                "medium": int(targets.medium),
                "hard": int(targets.hard),
            },
            "selected": class_counts(selected),
            "available": class_counts(all_candidates),
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and build a balanced GridFM dataset."
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--network-name", type=str, required=True)
    parser.add_argument("--network-source", type=str, default="pglib")
    parser.add_argument("--raw-network-dir-name", type=str, default=None)
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--simple-fraction", type=float, default=0.25)
    parser.add_argument("--medium-fraction", type=float, default=0.50)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-chunks", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--gridfm-retries", type=int, default=2)
    parser.add_argument("--gridfm-inactivity-timeout-sec", type=float, default=900.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--global-range", type=float, default=0.5)
    parser.add_argument("--max-scaling-factor", type=float, default=2.0)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--start-scaling-factor", type=float, default=1.0)
    parser.add_argument("--topology-variants", type=int, default=10)
    parser.add_argument("--topology-k", type=int, default=1)
    parser.add_argument(
        "--generation-perturbation-type",
        type=str,
        default="none",
        choices=["none", "cost_permutation", "cost_perturbation"],
    )
    parser.add_argument("--generation-perturbation-sigma", type=float, default=0.0)
    parser.add_argument(
        "--admittance-perturbation-type",
        type=str,
        default="none",
        choices=["none", "random_perturbation"],
    )
    parser.add_argument("--admittance-perturbation-sigma", type=float, default=0.0)
    parser.add_argument("--min-loading", type=float, default=105.0)
    parser.add_argument("--max-loading", type=float, default=260.0)
    parser.add_argument("--simple-min-loading", type=float, default=105.0)
    parser.add_argument("--simple-max-loading", type=float, default=120.0)
    parser.add_argument("--simple-max-hard", type=int, default=0)
    parser.add_argument("--simple-max-overloaded", type=int, default=2)
    parser.add_argument("--medium-min-loading", type=float, default=120.0)
    parser.add_argument("--medium-max-loading", type=float, default=150.0)
    parser.add_argument("--medium-max-hard", type=int, default=1)
    parser.add_argument("--medium-max-overloaded", type=int, default=5)
    parser.add_argument("--hard-min-loading", type=float, default=150.0)
    parser.add_argument("--hard-min-hard", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--self-play-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> None:
    configure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args()
    if args.gridfm_retries < 0:
        parser.error("--gridfm-retries must be >= 0")
    if args.gridfm_inactivity_timeout_sec < 0.0:
        parser.error("--gridfm-inactivity-timeout-sec must be >= 0")
    if not 0.0 <= args.train_fraction <= 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    if not 0.0 <= args.self_play_fraction <= 1.0:
        parser.error("--self-play-fraction must be between 0 and 1")
    if args.train_fraction + args.self_play_fraction > 1.0:
        parser.error("--train-fraction + --self-play-fraction must be <= 1")

    total_start = now()
    output_root = Path(args.output_root)
    paths = make_paths(output_root)
    for path in paths.values():
        ensure_dir(path)
    _drop_legacy_manifest_paths(paths["manifest_dir"])
    _remove_path_bearing_build_files(paths)

    targets = compute_class_targets(
        int(args.target_total),
        float(args.simple_fraction),
        float(args.medium_fraction),
        float(args.hard_fraction),
    )
    contract_fingerprint = dataset_contract_fingerprint(args)
    print("=" * 100)
    print("Balanced GridFM dataset builder")
    print("=" * 100)
    print(f"Dataset name: {args.dataset_name}")
    print(f"Network:      {args.network_name}")
    print(f"Source:       {args.network_source}")
    print(f"Output root:  {output_root}")
    print(f"Target total: {args.target_total}")
    print(f"Contract:     {contract_fingerprint}")

    all_candidates = (
        load_existing_candidates(
            paths["manifest_dir"],
            contract_fingerprint=contract_fingerprint,
        )
        if args.resume
        else pd.DataFrame()
    )
    selected = select_balanced_manifest(
        all_candidates,
        targets,
        int(args.split_seed),
    )

    if quotas_met(selected, targets):
        print("\nExisting canonical candidates already satisfy quotas.")
    else:
        for chunk_index in range(int(args.max_chunks)):
            name = chunk_name(chunk_index)
            candidate_path = paths["manifest_dir"] / f"{name}_candidates.csv"
            if args.resume and candidate_manifest_is_current(
                candidate_path,
                expected_contract_fingerprint=contract_fingerprint,
                chunk_index=chunk_index,
            ):
                print(
                    f"\n{name}: verified canonical candidates already exist, "
                    "skipping chunk processing"
                )
            else:
                chunk_candidates = process_chunk(
                    chunk_index=chunk_index,
                    args=args,
                    paths=paths,
                    contract_fingerprint=contract_fingerprint,
                )
                all_candidates = (
                    chunk_candidates.copy()
                    if all_candidates.empty
                    else pd.concat(
                        [all_candidates, chunk_candidates], ignore_index=True
                    )
                )
            if args.resume:
                all_candidates = load_existing_candidates(
                    paths["manifest_dir"],
                    contract_fingerprint=contract_fingerprint,
                )
            selected = select_balanced_manifest(
                all_candidates,
                targets,
                int(args.split_seed),
            )
            if quotas_met(selected, targets):
                print("\nCanonical class quotas are satisfied.")
                break

    if all_candidates.empty:
        raise RuntimeError(
            "GridFM generation produced no canonical classified candidates."
        )
    if not quotas_met(selected, targets) and not args.allow_partial:
        raise RuntimeError(
            "Could not satisfy canonical balanced dataset quotas. "
            "Increase --max-chunks, increase --chunk-size, or relax thresholds."
        )

    final_targets = compute_final_targets(
        all_candidates,
        targets,
        float(args.simple_fraction),
        float(args.medium_fraction),
        float(args.hard_fraction),
    )
    if final_targets.total <= 0:
        raise RuntimeError(
            "Could not build a proportional canonical dataset from "
            f"available candidates: {class_counts(all_candidates)}"
        )
    selected = select_balanced_manifest(
        all_candidates,
        final_targets,
        int(args.split_seed),
    )
    write_outputs(
        selected=selected,
        all_candidates=all_candidates,
        targets=final_targets,
        args=args,
        paths=paths,
    )

    print("\n" + "=" * 100)
    print("Merging selected raw scenarios")
    print("=" * 100)
    if paths["raw_dir"].exists():
        shutil.rmtree(paths["raw_dir"])
    raw_network_dir_name = (
        str(args.raw_network_dir_name)
        if args.raw_network_dir_name is not None
        else str(args.network_name)
    )
    merge_raw_parquet_files(
        selected,
        paths["raw_dir"],
        chunks_dir=paths["chunks_dir"],
        raw_network_dir_name=raw_network_dir_name,
    )
    _remove_path_bearing_build_files(paths)

    print("\nFinal output:")
    print(f"  raw:          {paths['raw_dir']}")
    print(f"  transitions:  {paths['transitions_dir'] / 'transitions_balanced.csv'}")
    print(f"  train:        {paths['transitions_dir'] / 'transitions_train.csv'}")
    print(f"  val:          {paths['transitions_dir'] / 'transitions_val.csv'}")
    print(f"  self-play:    {paths['transitions_dir'] / 'transitions_self_play.csv'}")
    print(f"  manifest:     {paths['manifest_dir'] / 'selected_manifest.csv'}")
    print_time("\nTotal runtime", total_start)
    print("\nDone.")


if __name__ == "__main__":
    main()

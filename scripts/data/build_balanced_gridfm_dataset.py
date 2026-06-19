from __future__ import annotations
import sys
import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.data.build_serious_transitions import build_fast_summary, make_output


# ======================================================================================
# Helpers
# ======================================================================================


def configure_utf8_stdio() -> None:
    """
    Force UTF-8 for redirected stdout/stderr on Windows.

    GridFM, Julia and IPOPT may print characters that cannot be encoded
    with the default Windows cp1251 console encoding.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if callable(reconfigure):
            reconfigure(
                encoding="utf-8",
                errors="replace",
            )

def now() -> float:
    return time.perf_counter()


def print_time(label: str, start: float) -> None:
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.2f} s")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def run_command(command: str, log_path: Path) -> None:
    ensure_dir(log_path.parent)

    print(f"Running command:")
    print(command)
    print(f"Log: {log_path}")

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            try:
                print(line, end="", flush=True)
            except UnicodeEncodeError:
                # Last-resort protection for older Windows terminals.
                safe_line = line.encode(
                    sys.stdout.encoding or "utf-8",
                    errors="replace",
                ).decode(
                    sys.stdout.encoding or "utf-8",
                    errors="replace",
                )
                print(safe_line, end="", flush=True)

            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        print("\nGridFM command failed. Last log lines:")
        try:
            lines = read_text(log_path).splitlines()
            for line in lines[-80:]:
                print(line)
        except Exception:
            pass

        raise RuntimeError(
            f"Command failed with exit code {return_code}: {command}"
        )


# ======================================================================================
# Config
# ======================================================================================


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
    target_total = int(target_total)

    simple = int(round(target_total * float(simple_fraction)))
    hard = int(round(target_total * float(hard_fraction)))
    medium = target_total - simple - hard

    if medium < 0:
        raise ValueError("Class fractions are invalid: medium target became negative.")

    return ClassTargets(
        simple=int(simple),
        medium=int(medium),
        hard=int(hard),
    )

def make_generation_perturbation_text(
    perturbation_type: str,
    sigma: float,
) -> str:
    """
    Build GridFM generation_perturbation YAML block.

    Supported by GridFM:
    - none
    - cost_permutation
    - cost_perturbation
    """

    perturbation_type = str(perturbation_type).strip().lower()

    if perturbation_type == "none":
        return """generation_perturbation:
  type: "none"
"""

    if perturbation_type == "cost_permutation":
        return """generation_perturbation:
  type: "cost_permutation"
"""

    if perturbation_type == "cost_perturbation":
        if float(sigma) <= 0.0:
            raise ValueError(
                "generation_perturbation_sigma must be > 0 "
                "when generation_perturbation_type='cost_perturbation'."
            )

        return f"""generation_perturbation:
  type: "cost_perturbation"
  sigma: {float(sigma)}
"""

    raise ValueError(
        "Unsupported generation_perturbation_type: "
        f"{perturbation_type}. "
        "Expected one of: none, cost_permutation, cost_perturbation."
    )


def make_admittance_perturbation_text(
    perturbation_type: str,
    sigma: float,
) -> str:
    """
    Build GridFM admittance_perturbation YAML block.

    Supported by GridFM:
    - none
    - random_perturbation
    """

    perturbation_type = str(perturbation_type).strip().lower()

    if perturbation_type == "none":
        return """admittance_perturbation:
  type: "none"
"""

    if perturbation_type == "random_perturbation":
        if float(sigma) <= 0.0:
            raise ValueError(
                "admittance_perturbation_sigma must be > 0 "
                "when admittance_perturbation_type='random_perturbation'."
            )

        return f"""admittance_perturbation:
  type: "random_perturbation"
  sigma: {float(sigma)}
"""

    raise ValueError(
        "Unsupported admittance_perturbation_type: "
        f"{perturbation_type}. "
        "Expected one of: none, random_perturbation."
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
    """
    Generate GridFM YAML text.

    This is intentionally kept inside this pipeline to avoid creating another
    config-template file. Change settings in the PowerShell wrapper.
    """

    data_dir_str = str(data_dir).replace("\\", "/")

    generation_perturbation_text = make_generation_perturbation_text(
        perturbation_type=generation_perturbation_type,
        sigma=generation_perturbation_sigma,
    )

    admittance_perturbation_text = make_admittance_perturbation_text(
        perturbation_type=admittance_perturbation_type,
        sigma=admittance_perturbation_sigma,
    )

    return f"""network:
  name: "{str(network_name)}"
  source: "{str(network_source)}"

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

{generation_perturbation_text}
{admittance_perturbation_text}

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
"""


# ======================================================================================
# Scenario classification and selection
# ======================================================================================


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
    raw_dir: Path,
) -> pd.DataFrame:
    df = summary.copy()

    df["source_chunk"] = str(chunk_name)
    df["source_raw_dir"] = str(raw_dir)
    df["source_scenario_id"] = df["scenario"].astype(int)

    return df


def select_balanced_manifest(
    candidates: pd.DataFrame,
    targets: ClassTargets,
    seed: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    df = candidates.copy()
    df = df[df["difficulty_class"].isin(["simple", "medium", "hard"])].copy()

    if df.empty:
        return df

    # Avoid duplicates from resume/repeated chunk processing.
    df = df.drop_duplicates(
        subset=["source_chunk", "source_scenario_id"],
        keep="first",
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

        # For every class we still prefer operationally relevant scenarios,
        # but random tie-break prevents deterministic duplicates.
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

        part = part.sort_values(
            sort_cols,
            ascending=[False for _ in sort_cols],
        )

        selected_parts.append(part.head(target_count))

    if not selected_parts:
        return pd.DataFrame()

    selected = pd.concat(selected_parts, ignore_index=True)

    class_order = {
        "simple": 0,
        "medium": 1,
        "hard": 2,
    }

    selected["_class_order"] = selected["difficulty_class"].map(class_order).astype(int)

    selected = selected.sort_values(
        ["_class_order", "source_chunk", "source_scenario_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    selected["global_scenario_id"] = np.arange(len(selected), dtype=int)

    selected = selected.drop(columns=["_class_order", "_random_tiebreak"], errors="ignore")

    return selected


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "difficulty_class" not in df.columns:
        return {
            "simple": 0,
            "medium": 0,
            "hard": 0,
        }

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
    """
    Find the largest possible balanced dataset that preserves the requested
    class proportions and does not exceed the available scenarios.

    Example:
        available = simple 152, medium 80, hard 59
        fractions = 0.25, 0.50, 0.25

        maximum balanced selection:
        simple 40, medium 80, hard 40
        total 160
    """

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
            "Class fractions must sum to 1.0. "
            f"Received: {fraction_sum}"
        )

    available = class_counts(candidates)

    limits = [
        available[class_name] / fraction
        for class_name, fraction in fractions.items()
        if fraction > 0.0
    ]

    if not limits:
        return ClassTargets(simple=0, medium=0, hard=0)

    maximum_total = int(np.floor(min(limits)))

    # Rounding in compute_class_targets can occasionally make one class exceed
    # its availability by one item, so verify and decrease if necessary.
    while maximum_total > 0:
        targets = compute_class_targets(
            target_total=maximum_total,
            simple_fraction=fractions["simple"],
            medium_fraction=fractions["medium"],
            hard_fraction=fractions["hard"],
        )

        if (
            targets.simple <= available["simple"]
            and targets.medium <= available["medium"]
            and targets.hard <= available["hard"]
        ):
            return targets

        maximum_total -= 1

    return ClassTargets(simple=0, medium=0, hard=0)

def quotas_met(selected: pd.DataFrame, targets: ClassTargets) -> bool:
    counts = class_counts(selected)

    return (
        counts["simple"] >= targets.simple
        and counts["medium"] >= targets.medium
        and counts["hard"] >= targets.hard
    )


# ======================================================================================
# Raw merge/remap
# ======================================================================================


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


def merge_raw_parquet_files(
    selected: pd.DataFrame,
    output_raw_dir: Path,
) -> None:
    """
    Merge selected scenarios from several chunk raw dirs into one balanced raw dir.

    All scenario ids are remapped to global_scenario_id.
    """

    ensure_dir(output_raw_dir)

    if selected.empty:
        raise RuntimeError("Cannot merge raw files: selected manifest is empty.")

    grouped = selected.groupby("source_raw_dir", sort=False)

    first_raw_dir = Path(str(selected["source_raw_dir"].iloc[0]))
    parquet_names = sorted(path.name for path in first_raw_dir.glob("*.parquet"))

    if not parquet_names:
        raise RuntimeError(f"No parquet files found in first raw dir: {first_raw_dir}")

    print("\nMerging raw parquet files...")

    for parquet_name in parquet_names:
        parts: list[pd.DataFrame] = []
        copied_static_file = False

        print(f"  {parquet_name}")

        for raw_dir_str, group in grouped:
            raw_dir = Path(str(raw_dir_str))
            src_path = raw_dir / parquet_name

            if not src_path.exists():
                print(f"    missing in {raw_dir}, skipping")
                continue

            df = pd.read_parquet(src_path)

            if "scenario" not in df.columns:
                # Static parquet without scenario column. Copy only once.
                if not copied_static_file:
                    dst_path = output_raw_dir / parquet_name
                    df.to_parquet(dst_path, index=False)
                    copied_static_file = True
                continue

            source_ids = set(int(x) for x in group["source_scenario_id"].values)

            mapping = {
                int(row.source_scenario_id): int(row.global_scenario_id)
                for row in group.itertuples(index=False)
            }

            filtered = df[df["scenario"].astype(int).isin(source_ids)].copy()

            if filtered.empty:
                continue

            filtered = remap_scenario_column(filtered, mapping)
            parts.append(filtered)

        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged = merged.sort_values("scenario").reset_index(drop=True)
            merged.to_parquet(output_raw_dir / parquet_name, index=False)

    # Copy non-parquet files from the first raw dir, except n_scenarios.txt.
    for src in first_raw_dir.iterdir():
        if not src.is_file():
            continue

        if src.suffix.lower() == ".parquet":
            continue

        if src.name == "n_scenarios.txt":
            continue

        try:
            shutil.copy2(src, output_raw_dir / src.name)
        except Exception:
            pass

    (output_raw_dir / "n_scenarios.txt").write_text(
        str(int(len(selected))),
        encoding="utf-8",
    )


def build_transitions_from_manifest(selected: pd.DataFrame) -> pd.DataFrame:
    df = selected.copy()

    # make_output expects scenario column.
    df["scenario"] = df["global_scenario_id"].astype(int)

    transitions = make_output(df)
    transitions["difficulty_class"] = df["difficulty_class"].values
    transitions["source_chunk"] = df["source_chunk"].values
    transitions["source_scenario_id"] = df["source_scenario_id"].astype(int).values

    # Keep class near the front.
    columns = list(transitions.columns)
    front = [
        "scenario_id",
        "difficulty_class",
        "source_chunk",
        "source_scenario_id",
    ]

    rest = [col for col in columns if col not in front]

    return transitions[front + rest].copy()


def stratified_split(
    transitions: pd.DataFrame,
    train_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []

    for class_name, group in transitions.groupby("difficulty_class", sort=False):
        group = group.copy()

        order = rng.permutation(len(group))
        group = group.iloc[order].reset_index(drop=True)

        n_train = int(round(len(group) * float(train_fraction)))

        train_parts.append(group.iloc[:n_train].copy())
        val_parts.append(group.iloc[n_train:].copy())

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)

    train = train.sort_values("scenario_id").reset_index(drop=True)
    val = val.sort_values("scenario_id").reset_index(drop=True)

    return train, val


# ======================================================================================
# Pipeline
# ======================================================================================


def chunk_name(index: int) -> str:
    return f"chunk{int(index):02d}"


def expected_raw_dir(
    chunk_dir: Path,
    raw_network_dir_name: str,
) -> Path:
    return chunk_dir / str(raw_network_dir_name) / "raw"


def process_chunk(
    *,
    chunk_index: int,
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> pd.DataFrame:
    name = chunk_name(chunk_index)
    chunk_dir = paths["chunks_dir"] / name
    raw_network_dir_name = (
        str(args.raw_network_dir_name)
        if args.raw_network_dir_name is not None
        else str(args.network_name)
    )

    raw_dir = expected_raw_dir(
        chunk_dir=chunk_dir,
        raw_network_dir_name=raw_network_dir_name,
    )

    config_path = paths["configs_dir"] / f"{name}.yaml"
    log_path = paths["logs_dir"] / f"{name}.log"

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
        generation_perturbation_type=str(
            args.generation_perturbation_type
        ),
        generation_perturbation_sigma=float(
            args.generation_perturbation_sigma
        ),
        admittance_perturbation_type=str(
            args.admittance_perturbation_type
        ),
        admittance_perturbation_sigma=float(
            args.admittance_perturbation_sigma
        ),
    )

    write_text(config_path, config_text)

    if raw_dir.exists() and args.resume:
        print(f"\n{name}: raw exists, resume mode - skipping GridFM generation")
    else:
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

        ensure_dir(chunk_dir)
        write_text(config_path, config_text)

        command = str(args.gridfm_command_template).replace(
            "{config}",
            str(config_path),
        )

        print("\n" + "=" * 100)
        print(f"Generating {name}")
        print("=" * 100)
        print(f"Chunk dir: {chunk_dir}")
        print(f"Raw dir:   {raw_dir}")
        print(f"Config:    {config_path}")

        run_command(command, log_path=log_path)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Expected GridFM raw dir was not created: {raw_dir}\n"
            f"Check GridFM command and log: {log_path}"
        )

    print("\n" + "=" * 100)
    print(f"Summarizing {name}")
    print("=" * 100)

    t0 = now()
    summary = build_fast_summary(raw_dir=raw_dir, show_progress=True)
    print_time(f"Build summary for {name}", t0)

    summary = classify_summary(summary, args)
    summary = add_source_columns(
        summary,
        chunk_name=name,
        raw_dir=raw_dir,
    )

    candidates = summary[summary["difficulty_class"].isin(["simple", "medium", "hard"])].copy()

    ensure_dir(paths["manifest_dir"])

    summary_path = paths["manifest_dir"] / f"{name}_summary.csv"
    candidates_path = paths["manifest_dir"] / f"{name}_candidates.csv"

    summary.to_csv(summary_path, index=False)
    candidates.to_csv(candidates_path, index=False)

    counts = class_counts(candidates)

    print("\nChunk candidates:")
    print(f"  simple: {counts['simple']}")
    print(f"  medium: {counts['medium']}")
    print(f"  hard:   {counts['hard']}")
    print(f"  total:  {len(candidates)}")

    return candidates


def load_existing_candidates(manifest_dir: Path) -> pd.DataFrame:
    files = sorted(manifest_dir.glob("chunk*_candidates.csv"))

    if not files:
        return pd.DataFrame()

    parts = [pd.read_csv(path) for path in files]

    return pd.concat(parts, ignore_index=True)


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

    all_candidates_path = paths["manifest_dir"] / "all_candidates.csv"
    selected_manifest_path = paths["manifest_dir"] / "selected_manifest.csv"
    class_summary_path = paths["manifest_dir"] / "class_summary.csv"

    all_candidates.to_csv(all_candidates_path, index=False)
    selected.to_csv(selected_manifest_path, index=False)

    transitions = build_transitions_from_manifest(selected)

    transitions_path = paths["transitions_dir"] / "transitions_balanced.csv"
    train_path = paths["transitions_dir"] / "transitions_train.csv"
    val_path = paths["transitions_dir"] / "transitions_val.csv"

    transitions.to_csv(transitions_path, index=False)

    train, val = stratified_split(
        transitions=transitions,
        train_fraction=float(args.train_fraction),
        seed=int(args.split_seed),
    )

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

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
                "available": int(available),
                "selected": int(selected_count),
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
    print(f"  all:   {transitions_path} ({len(transitions)})")
    print(f"  train: {train_path} ({len(train)})")
    print(f"  val:   {val_path} ({len(val)})")

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
            "paths": {
                "raw": str(paths["raw_dir"]),
                "transitions": str(transitions_path),
                "train": str(train_path),
                "val": str(val_path),
                "selected_manifest": str(selected_manifest_path),
                "all_candidates": str(all_candidates_path),
            },
        },
    )


def main() -> None:

    configure_utf8_stdio()
    
    parser = argparse.ArgumentParser(
        description="Generate and build a balanced GridFM dataset."
    )

    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)

    parser.add_argument(
        "--network-name",
        type=str,
        required=True,
        help="GridFM network name, for example case118_ieee, case39, case300_ieee.",
    )

    parser.add_argument(
        "--network-source",
        type=str,
        default="pglib",
        help="GridFM network source, for example pglib.",
    )

    parser.add_argument(
        "--raw-network-dir-name",
        type=str,
        default=None,
        help=(
            "Directory name created by GridFM under each chunk directory. "
            "If omitted, network-name is used."
        ),
    )

    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--simple-fraction", type=float, default=0.25)
    parser.add_argument("--medium-fraction", type=float, default=0.50)
    parser.add_argument("--hard-fraction", type=float, default=0.25)

    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-chunks", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--gridfm-command-template", type=str, required=True)

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
        choices=[
            "none",
            "cost_permutation",
            "cost_perturbation",
        ],
        help=(
            "GridFM generation perturbation type. "
            "Supported values: none, cost_permutation, cost_perturbation."
        ),
    )

    parser.add_argument(
        "--generation-perturbation-sigma",
        type=float,
        default=0.0,
        help=(
            "Sigma used when generation perturbation type "
            "is cost_perturbation."
        ),
    )

    parser.add_argument(
        "--admittance-perturbation-type",
        type=str,
        default="none",
        choices=[
            "none",
            "random_perturbation",
        ],
        help=(
            "GridFM admittance perturbation type. "
            "Supported values: none, random_perturbation."
        ),
    )

    parser.add_argument(
        "--admittance-perturbation-sigma",
        type=float,
        default=0.0,
        help=(
            "Sigma used when admittance perturbation type "
            "is random_perturbation."
        ),
    )
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
    parser.add_argument("--hard-min-hard", type=int, default=1)

    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already generated chunks and existing candidate CSVs.",
    )

    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write partial dataset if max chunks are exhausted before quotas are met.",
    )

    args = parser.parse_args()

    total_start = now()

    output_root = Path(args.output_root)
    paths = make_paths(output_root)

    for path in paths.values():
        ensure_dir(path)

    targets = compute_class_targets(
        target_total=int(args.target_total),
        simple_fraction=float(args.simple_fraction),
        medium_fraction=float(args.medium_fraction),
        hard_fraction=float(args.hard_fraction),
    )

    print("=" * 100)
    print("Balanced GridFM dataset builder")
    print("=" * 100)
    print(f"Dataset name: {args.dataset_name}")
    print(f"Network:      {args.network_name}")
    print(f"Source:       {args.network_source}")
    print(
        "Raw net dir:  "
        f"{args.raw_network_dir_name if args.raw_network_dir_name is not None else args.network_name}"
    )
    print(f"Output root:  {output_root}")
    print(f"Target total: {args.target_total}")
    print(f"Targets:      simple={targets.simple}, medium={targets.medium}, hard={targets.hard}")
    print(f"Chunk size:   {args.chunk_size}")
    print(f"Max chunks:   {args.max_chunks}")
    print(f"Resume:       {args.resume}")

    all_candidates = load_existing_candidates(paths["manifest_dir"]) if args.resume else pd.DataFrame()

    # After the minimum requested quotas are reached, use all available candidates
    # to build the largest dataset that preserves the requested class proportions.
    final_targets = compute_max_balanced_targets(
        candidates=all_candidates,
        simple_fraction=float(args.simple_fraction),
        medium_fraction=float(args.medium_fraction),
        hard_fraction=float(args.hard_fraction),
    )

    if final_targets.total <= 0:
        counts_available = class_counts(all_candidates)

        raise RuntimeError(
            "Could not build a proportional balanced dataset from available "
            f"candidates: {counts_available}"
        )

    print("\nMaximum proportional selection:")
    print(
        f"  simple={final_targets.simple}, "
        f"medium={final_targets.medium}, "
        f"hard={final_targets.hard}, "
        f"total={final_targets.total}"
    )

    selected = select_balanced_manifest(
        candidates=all_candidates,
        targets=final_targets,
        seed=int(args.split_seed),
    )

    if quotas_met(selected, targets):
        print("\nExisting candidates already satisfy quotas.")
    else:
        for chunk_index in range(int(args.max_chunks)):
            name = chunk_name(chunk_index)

            candidate_path = paths["manifest_dir"] / f"{name}_candidates.csv"

            if args.resume and candidate_path.exists():
                print(f"\n{name}: candidates already exist, skipping chunk processing")
            else:
                chunk_candidates = process_chunk(
                    chunk_index=chunk_index,
                    args=args,
                    paths=paths,
                )

                if all_candidates.empty:
                    all_candidates = chunk_candidates
                else:
                    all_candidates = pd.concat(
                        [all_candidates, chunk_candidates],
                        ignore_index=True,
                    )

            if args.resume:
                all_candidates = load_existing_candidates(paths["manifest_dir"])

            selected = select_balanced_manifest(
                candidates=all_candidates,
                targets=final_targets,
                seed=int(args.split_seed),
            )

            counts_available = class_counts(all_candidates)
            counts_selected = class_counts(selected)

            print("\nCurrent accumulated candidates:")
            print(
                f"  available simple={counts_available['simple']}, "
                f"medium={counts_available['medium']}, "
                f"hard={counts_available['hard']}"
            )
            print(
                f"  selected  simple={counts_selected['simple']}/{targets.simple}, "
                f"medium={counts_selected['medium']}/{targets.medium}, "
                f"hard={counts_selected['hard']}/{targets.hard}"
            )

            if quotas_met(selected, targets):
                print("\nAll class quotas are satisfied.")
                break

    if not quotas_met(selected, targets) and not args.allow_partial:
        counts_selected = class_counts(selected)

        raise RuntimeError(
            "Could not satisfy balanced dataset quotas.\n"
            f"Selected simple={counts_selected['simple']}/{targets.simple}, "
            f"medium={counts_selected['medium']}/{targets.medium}, "
            f"hard={counts_selected['hard']}/{targets.hard}.\n"
            "Increase --max-chunks, increase --chunk-size, or relax class thresholds."
        )

    # Trim exactly to target counts if allow_partial was not used.
    selected = select_balanced_manifest(
        candidates=all_candidates,
        targets=targets,
        seed=int(args.split_seed),
    )

    write_outputs(
        selected=selected,
        all_candidates=all_candidates,
        targets=targets,
        args=args,
        paths=paths,
    )

    print("\n" + "=" * 100)
    print("Merging selected raw scenarios")
    print("=" * 100)

    if paths["raw_dir"].exists():
        shutil.rmtree(paths["raw_dir"])

    merge_raw_parquet_files(
        selected=selected,
        output_raw_dir=paths["raw_dir"],
    )

    print("\nFinal output:")
    print(f"  raw:          {paths['raw_dir']}")
    print(f"  transitions:  {paths['transitions_dir'] / 'transitions_balanced.csv'}")
    print(f"  train:        {paths['transitions_dir'] / 'transitions_train.csv'}")
    print(f"  val:          {paths['transitions_dir'] / 'transitions_val.csv'}")
    print(f"  manifest:     {paths['manifest_dir'] / 'selected_manifest.csv'}")

    print_time("\nTotal runtime", total_start)
    print("\nDone.")


if __name__ == "__main__":
    main()
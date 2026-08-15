from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pypower.idx_brch import BR_STATUS

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.contracts import PHYSICS_CONFIG_CONTRACT_VERSION
from grid_topology_ai.power_flow_problem import build_scenario_power_flow_template
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.state_builder import GridFMState, GridFMStateBuilder
from grid_topology_ai.state_store import GridFMStateStore


RUNTIME_SCENARIO_STORE_SCHEMA_VERSION = 1
_RUNTIME_DIRECTORY_NAME = ".powergrid_runtime"
_RUNTIME_STORE_NAME = "scenario_store_v1"
_MANIFEST_NAME = "manifest.json"
_SOURCE_FILES = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
)

_BUS_COLUMNS = (
    "scenario",
    "load_scenario_idx",
    "bus",
    "Pd",
    "Qd",
    "Pg",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "vn_kv",
    "GS",
    "BS",
    "min_vm_pu",
    "max_vm_pu",
)
_BRANCH_COLUMNS = (
    "scenario",
    "load_scenario_idx",
    "idx",
    "from_bus",
    "to_bus",
    "pf",
    "qf",
    "pt",
    "qt",
    "r",
    "x",
    "b",
    "tap",
    "shift",
    "rate_a",
    "br_status",
    "ang_min",
    "ang_max",
)
_GEN_COLUMNS = (
    "scenario",
    "idx",
    "bus",
    "p_mw",
    "q_mvar",
    "min_p_mw",
    "max_p_mw",
    "min_q_mvar",
    "max_q_mvar",
    "in_service",
)
_TABLES = {
    "bus": ("bus_data.parquet", "bus", _BUS_COLUMNS),
    "branch": ("branch_data.parquet", "idx", _BRANCH_COLUMNS),
    "gen": ("gen_data.parquet", "idx", _GEN_COLUMNS),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(raw_dir: Path) -> tuple[str, dict[str, dict[str, object]]]:
    digest = hashlib.sha256()
    details: dict[str, dict[str, object]] = {}

    for name in _SOURCE_FILES:
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required GridFM file not found: {path}")
        file_digest = _sha256_file(path)
        stat = path.stat()
        details[name] = {
            "sha256": file_digest,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        digest.update(name.encode("utf-8"))
        digest.update(file_digest.encode("ascii"))

    return digest.hexdigest(), details


def _structured_records(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    fields: list[tuple[str, np.dtype[Any]]] = []
    arrays: dict[str, np.ndarray] = {}

    for column in columns:
        values = frame[column].to_numpy()
        if values.dtype.kind not in "biuf":
            raise TypeError(
                f"Runtime column {column!r} must be numeric or boolean, "
                f"got dtype {values.dtype}."
            )
        values = np.ascontiguousarray(values)
        arrays[column] = values
        fields.append((column, values.dtype))

    result = np.empty(len(frame), dtype=np.dtype(fields))
    for column, values in arrays.items():
        result[column] = values
    return result


def _read_source_table(
    raw_dir: Path,
    *,
    file_name: str,
    order_column: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    path = raw_dir / file_name
    frame = pd.read_parquet(path, columns=list(columns))
    if frame.empty:
        raise ValueError(f"No rows were loaded from {path}.")

    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    frame = frame.sort_values(["scenario", order_column]).reset_index(drop=True)
    if frame["scenario"].isna().any():
        raise ValueError(f"Scenario column contains null values: {path}")
    return frame


def _write_runtime_table(
    raw_dir: Path,
    output_dir: Path,
    *,
    table_name: str,
    file_name: str,
    order_column: str,
    columns: Sequence[str],
) -> tuple[dict[str, object], list[int]]:
    frame = _read_source_table(
        raw_dir,
        file_name=file_name,
        order_column=order_column,
        columns=columns,
    )
    records = _structured_records(frame, columns)
    output_name = f"{table_name}.npy"
    output_path = output_dir / output_name
    np.save(output_path, records, allow_pickle=False)

    scenario_ids = sorted(int(value) for value in frame["scenario"].unique())
    metadata: dict[str, object] = {
        "file": output_name,
        "columns": list(columns),
        "rows": int(records.shape[0]),
        "bytes": int(output_path.stat().st_size),
        "sha256": _sha256_file(output_path),
    }
    return metadata, scenario_ids


def _manifest_path(store_dir: Path) -> Path:
    return store_dir / _MANIFEST_NAME


def _load_manifest(store_dir: Path) -> dict[str, object]:
    path = _manifest_path(store_dir)
    if not path.exists():
        raise FileNotFoundError(f"Runtime store manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Runtime store manifest must be a JSON object.")
    return payload


def validate_runtime_scenario_store(
    store_dir: str | Path,
    *,
    expected_source_fingerprint: str | None = None,
    verify_hashes: bool = False,
) -> dict[str, object]:
    store_dir = Path(store_dir)
    manifest = _load_manifest(store_dir)

    if int(manifest.get("schema_version", -1)) != RUNTIME_SCENARIO_STORE_SCHEMA_VERSION:
        raise ValueError("Unsupported runtime scenario-store schema version.")
    if (
        expected_source_fingerprint is not None
        and manifest.get("source_fingerprint") != expected_source_fingerprint
    ):
        raise ValueError("Runtime scenario store was built from different source data.")

    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("Runtime scenario-store manifest has no table metadata.")

    for table_name in _TABLES:
        metadata = tables.get(table_name)
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing runtime table metadata: {table_name}")
        file_name = metadata.get("file")
        if not isinstance(file_name, str):
            raise ValueError(f"Invalid runtime table file: {table_name}")
        path = store_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Runtime table not found: {path}")
        if int(metadata.get("bytes", -1)) != int(path.stat().st_size):
            raise ValueError(f"Runtime table size mismatch: {path}")
        if verify_hashes and metadata.get("sha256") != _sha256_file(path):
            raise ValueError(f"Runtime table checksum mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if array.dtype.names is None:
                raise ValueError(f"Runtime table is not structured: {path}")
            expected_columns = tuple(metadata.get("columns", ()))
            if tuple(array.dtype.names) != expected_columns:
                raise ValueError(f"Runtime table columns mismatch: {path}")
            if int(array.shape[0]) != int(metadata.get("rows", -1)):
                raise ValueError(f"Runtime table row count mismatch: {path}")
        finally:
            del array

    return manifest


def ensure_runtime_scenario_store(
    raw_dir: str | Path,
    *,
    store_root: str | Path | None = None,
) -> Path:
    """Build or validate the immutable memory-mapped GridFM runtime store.

    Parquet remains the source of truth. The runtime store is a derived artifact
    containing only numeric columns needed by the teacher and power-flow path.
    A manifest is written last, so interrupted builds are never accepted.
    """

    raw_dir = Path(raw_dir).resolve()
    source_fingerprint, source_files = _source_fingerprint(raw_dir)
    root = (
        Path(store_root).resolve()
        if store_root is not None
        else raw_dir / _RUNTIME_DIRECTORY_NAME
    )
    store_dir = root / _RUNTIME_STORE_NAME

    try:
        validate_runtime_scenario_store(
            store_dir,
            expected_source_fingerprint=source_fingerprint,
            verify_hashes=True,
        )
        return store_dir
    except (FileNotFoundError, ValueError, OSError):
        pass

    root.mkdir(parents=True, exist_ok=True)
    temp_dir = root / f"{_RUNTIME_STORE_NAME}.tmp-{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        table_metadata: dict[str, dict[str, object]] = {}
        scenario_sets: list[list[int]] = []
        for table_name, (file_name, order_column, columns) in _TABLES.items():
            metadata, scenario_ids = _write_runtime_table(
                raw_dir,
                temp_dir,
                table_name=table_name,
                file_name=file_name,
                order_column=order_column,
                columns=columns,
            )
            table_metadata[table_name] = metadata
            scenario_sets.append(scenario_ids)

        if not scenario_sets or any(ids != scenario_sets[0] for ids in scenario_sets[1:]):
            raise ValueError(
                "bus/branch/gen runtime tables do not contain the same scenarios."
            )

        manifest = {
            "schema_version": RUNTIME_SCENARIO_STORE_SCHEMA_VERSION,
            "source_root": str(raw_dir),
            "source_fingerprint": source_fingerprint,
            "source_files": source_files,
            "scenario_ids": scenario_sets[0],
            "tables": table_metadata,
        }
        _manifest_path(temp_dir).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        validate_runtime_scenario_store(
            temp_dir,
            expected_source_fingerprint=source_fingerprint,
            verify_hashes=True,
        )

        if store_dir.exists():
            shutil.rmtree(store_dir)
        temp_dir.replace(store_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return store_dir


class MemoryMappedScenarioStore:
    """Read-only view over compiled GridFM tables backed by ``numpy.memmap``."""

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir).resolve()
        self.manifest = validate_runtime_scenario_store(self.store_dir)
        self._scenario_ids = tuple(
            int(value) for value in self.manifest.get("scenario_ids", [])
        )
        tables = self.manifest["tables"]
        assert isinstance(tables, dict)
        self._arrays: dict[str, np.ndarray] = {}
        self._columns: dict[str, tuple[str, ...]] = {}
        for table_name in _TABLES:
            metadata = tables[table_name]
            assert isinstance(metadata, dict)
            path = self.store_dir / str(metadata["file"])
            self._arrays[table_name] = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
            self._columns[table_name] = tuple(
                str(value) for value in metadata["columns"]
            )

    def scenario_ids(self) -> tuple[int, ...]:
        return self._scenario_ids

    def frame(self, table_name: str, scenario_id: int) -> pd.DataFrame:
        try:
            records = self._arrays[table_name]
            columns = self._columns[table_name]
        except KeyError as exc:
            raise KeyError(f"Unknown runtime table: {table_name}") from exc

        scenario_values = records["scenario"]
        scenario_id = int(scenario_id)
        left = int(np.searchsorted(scenario_values, scenario_id, side="left"))
        right = int(np.searchsorted(scenario_values, scenario_id, side="right"))
        if left == right:
            raise ValueError(
                f"Scenario {scenario_id} not found in runtime table {table_name}."
            )

        rows = records[left:right]
        return pd.DataFrame(
            {
                column: np.array(rows[column], copy=True)
                for column in columns
            }
        )

    def scenario_frames(self, scenario_id: int) -> dict[str, pd.DataFrame]:
        return {
            "bus": self.frame("bus", scenario_id),
            "branch": self.frame("branch", scenario_id),
            "gen": self.frame("gen", scenario_id),
        }


class MemoryMappedGridFMAdapter:
    """Teacher adapter that materializes only the currently requested scenario."""

    def __init__(
        self,
        store_dir: str | Path,
        *,
        scenario_ids: Sequence[int] | None = None,
        physics_config: PhysicsConfig | None = None,
    ) -> None:
        self.store = MemoryMappedScenarioStore(store_dir)
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG
        self.raw_dir = Path(str(self.store.manifest["source_root"]))

        available = set(self.store.scenario_ids())
        if scenario_ids is None:
            self._scenario_ids = tuple(sorted(available))
        else:
            requested = tuple(sorted({int(value) for value in scenario_ids}))
            if not requested:
                raise ValueError("scenario_ids was provided, but it is empty.")
            missing = sorted(set(requested) - available)
            if missing:
                raise ValueError(
                    "Runtime store is missing requested scenarios: "
                    f"{missing[:20]}"
                )
            self._scenario_ids = requested

    def scenario_ids(self) -> list[int]:
        return list(self._scenario_ids)

    def scenario_frames(self, scenario_id: int) -> dict[str, pd.DataFrame]:
        scenario_id = int(scenario_id)
        if scenario_id not in self._scenario_ids:
            raise ValueError(f"Scenario {scenario_id} is outside this worker shard.")
        return self.store.scenario_frames(scenario_id)

    def build_state(self, scenario_id: int) -> GridFMState:
        frames = self.scenario_frames(scenario_id)
        builder = GridFMStateBuilder(physics_config=self.physics_config)
        return builder.build_from_frames(
            scenario_id=int(scenario_id),
            bus_df=frames["bus"],
            branch_df=frames["branch"],
            gen_df=frames["gen"],
            power_flow_converged=False,
        )


class MemoryMappedGridFMPowerFlowBackend(GridFMPowerFlowBackend):
    """Exact-cache backend whose immutable source frames come from mmap storage."""

    def _scenario_problem_resources(self, scenario_id: int):
        scenario_id = int(scenario_id)
        template = self._active_problem_template
        frames = self._active_problem_frames
        if (
            template is not None
            and frames is not None
            and int(template.scenario_id) == scenario_id
        ):
            return template, frames

        provider = getattr(self.adapter, "scenario_frames", None)
        if not callable(provider):
            return super()._scenario_problem_resources(scenario_id)

        frames = provider(scenario_id)
        bus_df = frames["bus"].sort_values("bus").reset_index(drop=True)
        branch_df = frames["branch"].sort_values("idx").reset_index(drop=True)
        gen_df = frames["gen"].sort_values("idx").reset_index(drop=True)
        template = build_scenario_power_flow_template(
            scenario_id=scenario_id,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
            base_mva=self.base_mva,
        )
        normalized_frames = {
            "bus": bus_df,
            "branch": branch_df,
            "gen": gen_df,
        }
        self._active_problem_template = template
        self._active_problem_frames = normalized_frames
        return template, normalized_frames

    def _build_ppc(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None,
    ):
        template, frames = self._scenario_problem_resources(int(scenario_id))
        bus = template.bus.copy()
        branch = template.branch.copy()
        gen = template.gen.copy()
        result_frames = frames

        if switched_off_branch_id is not None:
            branch_id = int(switched_off_branch_id)
            positions = np.flatnonzero(template.branch_ids == branch_id)
            if positions.size != 1:
                raise ValueError(
                    f"Expected exactly one branch id {branch_id} in scenario "
                    f"{scenario_id}, found {positions.size}."
                )
            branch[int(positions[0]), BR_STATUS] = 0.0
            branch_frame = frames["branch"].copy()
            branch_frame.loc[
                branch_frame["idx"].astype(int) == branch_id,
                "br_status",
            ] = 0.0
            result_frames = {
                "bus": frames["bus"],
                "branch": branch_frame,
                "gen": frames["gen"],
            }

        return (
            {
                "version": "2",
                "baseMVA": float(template.base_mva),
                "bus": bus,
                "branch": branch,
                "gen": gen,
            },
            result_frames,
        )


def build_memory_mapped_teacher_context(
    *,
    runtime_store_dir: str | Path,
    states_dir: str | Path,
    task_config: Mapping[str, Any],
    scenario_ids: Sequence[int],
    memory_registry=None,
) -> dict[str, Any]:
    """Construct the same teacher components using a read-only mmap adapter."""

    if task_config.get("physics_config_contract_version") != PHYSICS_CONFIG_CONTRACT_VERSION:
        raise ValueError("Unsupported physics config contract in worker payload.")
    physics_config = PhysicsConfig.from_mapping(task_config["physics_config"])
    if physics_config.fingerprint() != task_config.get("physics_config_fingerprint"):
        raise ValueError("PhysicsConfig fingerprint mismatch in worker payload.")

    adapter = MemoryMappedGridFMAdapter(
        runtime_store_dir,
        scenario_ids=scenario_ids,
        physics_config=physics_config,
    )
    cache_enabled = not bool(task_config.get("disable_cache", False))
    backend = MemoryMappedGridFMPowerFlowBackend(
        adapter=adapter,  # type: ignore[arg-type]
        physics_config=physics_config,
        enable_cache=cache_enabled,
    )
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=cache_enabled,
    )
    return {
        "adapter": adapter,
        "backend": backend,
        "action_space": action_space,
        "reward_fn": GridFMReward(physics_config=physics_config),
        "physics_config": physics_config,
        "state_store": GridFMStateStore(Path(states_dir)),
        "task_config": dict(task_config),
        "processed_in_worker": 0,
        "memory_registry": memory_registry,
    }

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pypower.idx_brch import BR_STATUS, PF, PT, QF, QT

from grid_topology_ai.actions import GridFMActionSpace
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.power_flow import InvalidPhysicalState
from grid_topology_ai.power_flow.problem import (
    ScenarioPowerFlowTemplate,
    _build_branch_matrix,
    _build_bus_matrix,
    _build_gen_matrix,
    build_scenario_power_flow_template,
)
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend
from grid_topology_ai.physics.constraints import (
    PhysicalNetworkArrays,
    calculate_physical_metrics,
)
from grid_topology_ai.physics.utility import GridFMReward
from grid_topology_ai.state import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
    GridFMStateBuilder,
    GridFMStateStore,
)


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

        if not scenario_sets or any(
            ids != scenario_sets[0] for ids in scenario_sets[1:]
        ):
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
            {column: np.array(rows[column], copy=True) for column in columns}
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
                    f"Runtime store is missing requested scenarios: {missing[:20]}"
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

    physics_config = PhysicsConfig.from_mapping(task_config["physics_config"])

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


_BUS_FEATURE_INDEX = {name: index for index, name in enumerate(BUS_FEATURE_COLUMNS)}
_BRANCH_FEATURE_INDEX = {
    name: index for index, name in enumerate(BRANCH_FEATURE_COLUMNS)
}


@dataclass(frozen=True, slots=True)
class _RuntimeColumnView:
    values: np.ndarray

    def to_numpy(self, dtype=None, copy: bool = False) -> np.ndarray:
        result = np.asarray(self.values, dtype=dtype)
        return result.copy() if copy else result

    def __iter__(self):
        return iter(self.values)


@dataclass(frozen=True, slots=True)
class _RuntimeFrameView:
    records: np.ndarray

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.records.dtype.names or ())

    def __len__(self) -> int:
        return int(self.records.shape[0])

    def __getitem__(self, column: str) -> _RuntimeColumnView:
        names = self.records.dtype.names or ()
        if column not in names:
            raise KeyError(column)
        return _RuntimeColumnView(self.records[column])


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeData:
    scenario_id: int
    bus: np.ndarray
    branch: np.ndarray
    gen: np.ndarray

    def frame_views(self) -> dict[str, _RuntimeFrameView]:
        return {
            "bus": _RuntimeFrameView(self.bus),
            "branch": _RuntimeFrameView(self.branch),
            "gen": _RuntimeFrameView(self.gen),
        }


def _integral_values(
    values: np.ndarray,
    *,
    label: str,
    scenario_id: int,
) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: {label} must contain finite integral values."
        )

    limits = np.iinfo(np.int64)
    if np.any(numeric < limits.min) or np.any(numeric > limits.max):
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: {label} cannot be represented as int64."
        )
    return numeric.astype(np.int64)


def _unique_ids(
    values: np.ndarray,
    *,
    label: str,
    scenario_id: int,
) -> np.ndarray:
    ids = _integral_values(values, label=label, scenario_id=scenario_id)
    if np.unique(ids).size != ids.size:
        raise InvalidPhysicalState(f"scenario {scenario_id}: duplicate {label}.")
    return ids


def _binary_values(
    values: np.ndarray,
    *,
    label: str,
    scenario_id: int,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all() or not np.isin(result, (0.0, 1.0)).all():
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: {label} must contain only 0 or 1."
        )
    return result


def _float32_feature(values: np.ndarray, *, label: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise InvalidPhysicalState(f"{label} contains NaN or infinity.")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        feature = numeric.astype(np.float32)
    if not np.isfinite(feature).all():
        raise InvalidPhysicalState(f"{label} cannot be represented in float32.")
    return feature


def _compile_runtime_template(
    data: ScenarioRuntimeData,
    base_mva: float,
) -> ScenarioPowerFlowTemplate:
    frames = data.frame_views()
    bus = frames["bus"]
    branch = frames["branch"]
    gen = frames["gen"]
    base_mva = float(base_mva)
    if not np.isfinite(base_mva) or base_mva <= 0.0:
        raise ValueError(f"base_mva must be finite and positive, got {base_mva}.")

    return ScenarioPowerFlowTemplate(
        scenario_id=int(data.scenario_id),
        base_mva=base_mva,
        bus_ids=np.asarray(data.bus["bus"], dtype=np.int64).copy(),
        branch_ids=np.asarray(data.branch["idx"], dtype=np.int64).copy(),
        generator_ids=np.asarray(data.gen["idx"], dtype=np.int64).copy(),
        bus=_build_bus_matrix(bus, base_mva),  # type: ignore[arg-type]
        branch=_build_branch_matrix(branch),  # type: ignore[arg-type]
        gen=_build_gen_matrix(gen, bus, base_mva),  # type: ignore[arg-type]
    )


def _fill_generator_features(
    *,
    bus_features: np.ndarray,
    bus_ids: np.ndarray,
    gen_rows: np.ndarray,
    scenario_id: int,
) -> None:
    gen_ids = _unique_ids(
        gen_rows["idx"],
        label="generator IDs",
        scenario_id=scenario_id,
    )
    if np.any(np.diff(gen_ids) < 0):
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: generator runtime rows are not sorted."
        )

    gen_bus_ids = _integral_values(
        gen_rows["bus"],
        label="generator bus",
        scenario_id=scenario_id,
    )
    bus_position = {int(bus_id): position for position, bus_id in enumerate(bus_ids)}
    try:
        gen_bus_pos = np.asarray(
            [bus_position[int(bus_id)] for bus_id in gen_bus_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: generator references unknown bus {exc.args[0]}."
        ) from exc

    status = _binary_values(
        gen_rows["in_service"],
        label="generator status",
        scenario_id=scenario_id,
    )
    pg = np.asarray(gen_rows["p_mw"], dtype=np.float64)
    qg = np.asarray(gen_rows["q_mvar"], dtype=np.float64)
    p_min = np.asarray(gen_rows["min_p_mw"], dtype=np.float64)
    p_max = np.asarray(gen_rows["max_p_mw"], dtype=np.float64)
    q_min = np.asarray(gen_rows["min_q_mvar"], dtype=np.float64)
    q_max = np.asarray(gen_rows["max_q_mvar"], dtype=np.float64)

    if not np.isfinite(pg).all() or not np.isfinite(qg).all():
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: generator output contains NaN or infinity."
        )
    if not all(np.isfinite(values).all() for values in (p_min, p_max, q_min, q_max)):
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: generator limits contain NaN or infinity."
        )
    if np.any(p_min > p_max) or np.any(q_min > q_max):
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: generator limits are inverted."
        )

    active = status > 0.0
    n_bus = int(bus_features.shape[0])
    active_pos = gen_bus_pos[active]
    sums = {
        "Pg": np.zeros(n_bus, dtype=np.float64),
        "Qg": np.zeros(n_bus, dtype=np.float64),
        "gen_online_count": np.zeros(n_bus, dtype=np.float64),
        "gen_p_min_mw": np.zeros(n_bus, dtype=np.float64),
        "gen_p_max_mw": np.zeros(n_bus, dtype=np.float64),
        "gen_q_min_mvar": np.zeros(n_bus, dtype=np.float64),
        "gen_q_max_mvar": np.zeros(n_bus, dtype=np.float64),
        "gen_p_limit_violation_count": np.zeros(n_bus, dtype=np.float64),
        "gen_q_limit_violation_count": np.zeros(n_bus, dtype=np.float64),
    }
    minima = {
        "gen_min_p_down_margin_mw": np.full(n_bus, np.inf),
        "gen_min_p_up_margin_mw": np.full(n_bus, np.inf),
        "gen_min_q_down_margin_mvar": np.full(n_bus, np.inf),
        "gen_min_q_up_margin_mvar": np.full(n_bus, np.inf),
    }

    if active.any():
        pg_a = pg[active]
        qg_a = qg[active]
        p_min_a = p_min[active]
        p_max_a = p_max[active]
        q_min_a = q_min[active]
        q_max_a = q_max[active]

        p_down = pg_a - p_min_a
        p_up = p_max_a - pg_a
        q_down = qg_a - q_min_a
        q_up = q_max_a - qg_a

        for name, values in (
            ("Pg", pg_a),
            ("Qg", qg_a),
            ("gen_p_min_mw", p_min_a),
            ("gen_p_max_mw", p_max_a),
            ("gen_q_min_mvar", q_min_a),
            ("gen_q_max_mvar", q_max_a),
        ):
            np.add.at(sums[name], active_pos, values)

        np.add.at(sums["gen_online_count"], active_pos, 1.0)
        np.add.at(
            sums["gen_p_limit_violation_count"],
            active_pos,
            ((p_down < 0.0) | (p_up < 0.0)).astype(np.float64),
        )
        np.add.at(
            sums["gen_q_limit_violation_count"],
            active_pos,
            ((q_down < 0.0) | (q_up < 0.0)).astype(np.float64),
        )
        for name, values in (
            ("gen_min_p_down_margin_mw", p_down),
            ("gen_min_p_up_margin_mw", p_up),
            ("gen_min_q_down_margin_mvar", q_down),
            ("gen_min_q_up_margin_mvar", q_up),
        ):
            np.minimum.at(minima[name], active_pos, values)

    available = sums["gen_online_count"] > 0.0
    for values in minima.values():
        values[~available] = 0.0

    sums.update(minima)
    sums["gen_available"] = available.astype(np.float64)
    sums["gen_p_down_margin_mw"] = sums["Pg"] - sums["gen_p_min_mw"]
    sums["gen_p_up_margin_mw"] = sums["gen_p_max_mw"] - sums["Pg"]
    sums["gen_q_down_margin_mvar"] = sums["Qg"] - sums["gen_q_min_mvar"]
    sums["gen_q_up_margin_mvar"] = sums["gen_q_max_mvar"] - sums["Qg"]

    for name, values in sums.items():
        bus_features[:, _BUS_FEATURE_INDEX[name]] = _float32_feature(
            values,
            label=f"Bus feature {name}",
        )


def _build_runtime_state(
    *,
    data: ScenarioRuntimeData,
    template: ScenarioPowerFlowTemplate,
    physics_config: PhysicsConfig,
) -> GridFMState:
    scenario_id = int(data.scenario_id)
    bus_ids = _unique_ids(
        data.bus["bus"],
        label="bus IDs",
        scenario_id=scenario_id,
    )
    branch_ids = _unique_ids(
        data.branch["idx"],
        label="branch IDs",
        scenario_id=scenario_id,
    )

    if np.any(np.diff(bus_ids) < 0) or np.any(np.diff(branch_ids) < 0):
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: runtime rows are not sorted."
        )

    branch_from = _integral_values(
        data.branch["from_bus"],
        label="branch from_bus",
        scenario_id=scenario_id,
    )
    branch_to = _integral_values(
        data.branch["to_bus"],
        label="branch to_bus",
        scenario_id=scenario_id,
    )
    branch_status64 = _binary_values(
        data.branch["br_status"],
        label="branch status",
        scenario_id=scenario_id,
    )

    bus_position = {int(bus_id): position for position, bus_id in enumerate(bus_ids)}
    try:
        edge_index = np.vstack(
            (
                np.asarray(
                    [bus_position[int(value)] for value in branch_from],
                    dtype=np.int64,
                ),
                np.asarray(
                    [bus_position[int(value)] for value in branch_to],
                    dtype=np.int64,
                ),
            )
        )
    except KeyError as exc:
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: branch references unknown bus {exc.args[0]}."
        ) from exc

    vmin = np.asarray(data.bus["min_vm_pu"], dtype=np.float64)
    vmax = np.asarray(data.bus["max_vm_pu"], dtype=np.float64)
    if not np.isfinite(vmin).all() or not np.isfinite(vmax).all():
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: bus voltage limits must be finite."
        )
    if np.any(vmin > vmax):
        raise InvalidPhysicalState(
            f"scenario {scenario_id}: bus voltage limits are inverted."
        )

    bus_features = np.zeros(
        (len(data.bus), len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    for name in (
        "Pd",
        "Qd",
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
    ):
        bus_features[:, _BUS_FEATURE_INDEX[name]] = _float32_feature(
            data.bus[name],
            label=f"Bus feature {name}",
        )
    _fill_generator_features(
        bus_features=bus_features,
        bus_ids=bus_ids,
        gen_rows=data.gen,
        scenario_id=scenario_id,
    )

    branch_features = np.zeros(
        (len(data.branch), len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    for name in ("pf", "qf", "pt", "qt", "r", "x", "b", "tap", "shift"):
        branch_features[:, _BRANCH_FEATURE_INDEX[name]] = _float32_feature(
            data.branch[name],
            label=f"Branch feature {name}",
        )

    rate_a64 = np.asarray(data.branch["rate_a"], dtype=np.float64)
    if not np.isfinite(rate_a64).all() or np.any(rate_a64 < 0.0):
        raise InvalidPhysicalState("Branch RATE_A must be finite and non-negative.")
    if physics_config.zero_rate_a_policy.value == "error" and np.any(
        (branch_status64 > 0.0) & (rate_a64 == 0.0)
    ):
        raise InvalidPhysicalState(
            "Active branch RATE_A=0 is forbidden by PhysicsConfig."
        )

    rate_feature = _float32_feature(
        rate_a64,
        label="Branch feature rate_a",
    )
    if np.any((rate_a64 > 0.0) & (rate_feature == 0.0)):
        raise InvalidPhysicalState(
            "Positive RATE_A underflows to zero in feature precision."
        )
    branch_features[:, _BRANCH_FEATURE_INDEX["rate_a"]] = rate_feature
    branch_status = branch_status64.astype(np.float32)
    branch_features[:, _BRANCH_FEATURE_INDEX["br_status"]] = branch_status

    pf = np.asarray(data.branch["pf"], dtype=np.float64)
    qf = np.asarray(data.branch["qf"], dtype=np.float64)
    pt = np.asarray(data.branch["pt"], dtype=np.float64)
    qt = np.asarray(data.branch["qt"], dtype=np.float64)
    s_from = np.hypot(pf, qf)
    s_to = np.hypot(pt, qt)
    s_max = np.maximum(s_from, s_to)
    active = branch_status64 > 0.0
    rated = active & (rate_a64 > 0.0)
    loading = np.zeros_like(s_max)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        loading[rated] = s_max[rated] / rate_a64[rated] * 100.0

    for name, values in (
        ("s_from_mva", s_from),
        ("s_to_mva", s_to),
        ("s_max_mva", s_max),
        ("loading_percent", loading),
        ("unlimited_rating", (rate_a64 == 0.0).astype(np.float64)),
    ):
        branch_features[:, _BRANCH_FEATURE_INDEX[name]] = _float32_feature(
            values,
            label=f"Branch feature {name}",
        )

    physical_branch = np.zeros((len(data.branch), QT + 1), dtype=np.float64)
    physical_branch[:, : template.branch.shape[1]] = template.branch
    physical_branch[:, PF] = pf
    physical_branch[:, QF] = qf
    physical_branch[:, PT] = pt
    physical_branch[:, QT] = qt
    physical_metrics = calculate_physical_metrics(
        PhysicalNetworkArrays(
            bus=template.bus,
            branch=physical_branch,
            gen=template.gen,
        ),
        power_flow_converged=False,
        physics_config=physics_config,
    )

    vm = np.asarray(data.bus["Vm"], dtype=np.float64)
    metrics = {
        "num_buses": int(len(data.bus)),
        "num_branches": int(len(data.branch)),
        "num_generators": int(len(data.gen)),
        "mean_loading_percent": (
            float(np.mean(loading[rated])) if np.any(rated) else 0.0
        ),
        "min_vm_pu": float(np.min(vm)),
        "max_vm_pu": float(np.max(vm)),
        "num_outaged_branches": int(np.sum(branch_status64 <= 0.0)),
        **physical_metrics,
    }
    outaged = branch_status64 <= 0.0

    return GridFMState(
        scenario_id=scenario_id,
        load_scenario_idx=float(data.bus["load_scenario_idx"][0]),
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        branch_ids=branch_ids,
        branch_status=branch_status,
        metrics=metrics,
        outaged_branch_ids=[int(value) for value in branch_ids[outaged]],
        bus_ids=bus_ids,
    )


class NumPyMemoryMappedGridFMAdapter:
    """Teacher adapter that never materializes DataFrames on its hot path."""

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
                    f"Runtime store is missing requested scenarios: {missing[:20]}"
                )
            self._scenario_ids = requested

        self._ranges: dict[str, dict[int, tuple[int, int]]] = {
            "bus": {},
            "branch": {},
            "gen": {},
        }
        self._active_data: ScenarioRuntimeData | None = None
        self._active_template: ScenarioPowerFlowTemplate | None = None

    def scenario_ids(self) -> list[int]:
        return list(self._scenario_ids)

    def _require_scenario(self, scenario_id: int) -> int:
        scenario_id = int(scenario_id)
        if scenario_id not in self._scenario_ids:
            raise ValueError(f"Scenario {scenario_id} is outside this worker shard.")
        return scenario_id

    def _rows(self, table_name: str, scenario_id: int) -> np.ndarray:
        records = self.store._arrays[table_name]
        ranges = self._ranges[table_name]
        bounds = ranges.get(scenario_id)
        if bounds is None:
            scenario_values = records["scenario"]
            left = int(np.searchsorted(scenario_values, scenario_id, side="left"))
            right = int(np.searchsorted(scenario_values, scenario_id, side="right"))
            if left == right:
                raise ValueError(
                    f"Scenario {scenario_id} not found in runtime table {table_name}."
                )
            bounds = (left, right)
            ranges[scenario_id] = bounds

        left, right = bounds
        return records[left:right]

    def scenario_data(self, scenario_id: int) -> ScenarioRuntimeData:
        scenario_id = self._require_scenario(scenario_id)
        active = self._active_data
        if active is not None and int(active.scenario_id) == scenario_id:
            return active

        data = ScenarioRuntimeData(
            scenario_id=scenario_id,
            bus=self._rows("bus", scenario_id),
            branch=self._rows("branch", scenario_id),
            gen=self._rows("gen", scenario_id),
        )
        self._active_data = data
        self._active_template = None
        return data

    def scenario_frames(self, scenario_id: int):
        scenario_id = self._require_scenario(scenario_id)
        return self.store.scenario_frames(scenario_id)

    def scenario_power_flow_template(
        self,
        scenario_id: int,
    ) -> ScenarioPowerFlowTemplate:
        scenario_id = self._require_scenario(scenario_id)
        template = self._active_template
        if template is not None and int(template.scenario_id) == scenario_id:
            return template

        template = _compile_runtime_template(
            self.scenario_data(scenario_id),
            self.physics_config.base_mva,
        )
        self._active_template = template
        return template

    def scenario_power_flow_resources(
        self,
        scenario_id: int,
    ) -> tuple[ScenarioPowerFlowTemplate, dict[str, _RuntimeFrameView]]:
        data = self.scenario_data(scenario_id)
        return self.scenario_power_flow_template(scenario_id), data.frame_views()

    def build_state(self, scenario_id: int) -> GridFMState:
        data = self.scenario_data(scenario_id)
        template = self.scenario_power_flow_template(scenario_id)
        return _build_runtime_state(
            data=data,
            template=template,
            physics_config=self.physics_config,
        )


class NumPyMemoryMappedGridFMPowerFlowBackend(MemoryMappedGridFMPowerFlowBackend):
    """Memory-mapped backend whose repeated scenario resources stay in NumPy."""

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

        provider = getattr(self.adapter, "scenario_power_flow_resources", None)
        if not callable(provider):
            return super()._scenario_problem_resources(scenario_id)

        template, runtime_frames = provider(scenario_id)
        self._active_problem_template = template
        self._active_problem_frames = runtime_frames  # type: ignore[assignment]
        return template, runtime_frames

    def _build_ppc(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None,
    ):
        template_provider = getattr(
            self.adapter,
            "scenario_power_flow_template",
            None,
        )
        frame_provider = getattr(self.adapter, "scenario_frames", None)
        if not callable(template_provider) or not callable(frame_provider):
            return super()._build_ppc(scenario_id, switched_off_branch_id)

        template = template_provider(int(scenario_id))
        frames = frame_provider(int(scenario_id))
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


def build_numpy_teacher_context(
    *,
    runtime_store_dir: str | Path,
    states_dir: str | Path,
    task_config: Mapping[str, Any],
    scenario_ids: Sequence[int],
    memory_registry=None,
) -> dict[str, Any]:
    """Build teacher components without DataFrame scenario materialization."""

    physics_config = PhysicsConfig.from_mapping(task_config["physics_config"])

    adapter = NumPyMemoryMappedGridFMAdapter(
        runtime_store_dir,
        scenario_ids=scenario_ids,
        physics_config=physics_config,
    )
    cache_enabled = not bool(task_config.get("disable_cache", False))
    backend = NumPyMemoryMappedGridFMPowerFlowBackend(
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

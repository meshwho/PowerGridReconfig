from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.self_play.physical_lineage import (
    PHYSICAL_LINEAGE_CONTRACT_VERSION,
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
    PhysicalLineage,
    require_physical_lineage,
)

LINEAGE_COLUMNS: tuple[str, ...] = (
    "lineage_contract_version",
    "base_case_id",
    "load_profile_id",
    "contingency_family_id",
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
)

_BUS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "scenario",
    "bus",
    "Pd",
    "Qd",
)
_BUS_BASE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "vn_kv",
    "GS",
    "BS",
    "min_vm_pu",
    "max_vm_pu",
    "PQ",
    "PV",
    "REF",
)
_BRANCH_REQUIRED_COLUMNS: tuple[str, ...] = (
    "scenario",
    "idx",
    "from_bus",
    "to_bus",
    "r",
    "x",
    "b",
    "br_status",
)
_BRANCH_BASE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "tap",
    "shift",
    "rate_a",
    "ang_min",
    "ang_max",
)
_GEN_REQUIRED_COLUMNS: tuple[str, ...] = (
    "scenario",
    "idx",
    "bus",
)
_GEN_BASE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "max_q_mvar",
    "min_q_mvar",
    "max_p_mw",
    "min_p_mw",
)


def _read_parquet_columns(
    path: Path,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Physical lineage parquet not found: {path}")

    requested = list(dict.fromkeys([*required, *optional]))
    try:
        frame = pd.read_parquet(path, columns=requested)
    except Exception:
        frame = pd.read_parquet(path)

    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path} is missing physical lineage columns: {missing}."
        )

    available = [name for name in requested if name in frame.columns]
    return frame[available].copy()


def _coerce_scenario_id(value: object, *, source: str) -> int:
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid scenario_id in {source}: {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid scenario_id in {source}: {value!r}."
        ) from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid scenario_id in {source}: {value!r}.")
    return int(number)


def _canonical_scalar(value: object, *, source: str) -> object:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite physical value in {source}: {value!r}.")
        if number == 0.0:
            number = 0.0
        return number.hex()
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty physical value in {source}.")
    return text


def _frame_digest(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    source: str,
) -> str:
    columns = list(dict.fromkeys([*key_columns, *value_columns]))
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}.")
    if frame.empty:
        raise ValueError(f"{source} contains no rows.")

    ordered = frame[columns].sort_values(
        list(key_columns),
        kind="mergesort",
    )
    payload = [
        [
            _canonical_scalar(value, source=source)
            for value in row
        ]
        for row in ordered.itertuples(index=False, name=None)
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scenario_subset(
    frame: pd.DataFrame,
    scenario_id: int,
    *,
    source: Path,
) -> pd.DataFrame:
    scenario_values = frame["scenario"].map(
        lambda value: _coerce_scenario_id(value, source=str(source))
    )
    subset = frame.loc[scenario_values == int(scenario_id)].copy()
    if subset.empty:
        raise ValueError(
            f"Scenario {scenario_id} is missing from physical data: {source}."
        )
    return subset


def _base_case_id(
    *,
    bus_rows: pd.DataFrame,
    branch_rows: pd.DataFrame,
    gen_rows: pd.DataFrame | None,
    scenario_id: int,
) -> str:
    bus_values = [
        name
        for name in _BUS_BASE_OPTIONAL_COLUMNS
        if name in bus_rows.columns
    ]
    branch_values = [
        "from_bus",
        "to_bus",
        "r",
        "x",
        "b",
        *[
            name
            for name in _BRANCH_BASE_OPTIONAL_COLUMNS
            if name in branch_rows.columns
        ],
    ]
    payload: dict[str, str] = {
        "bus": _frame_digest(
            bus_rows,
            key_columns=("bus",),
            value_columns=bus_values,
            source=f"bus base case scenario {scenario_id}",
        ),
        "branch": _frame_digest(
            branch_rows,
            key_columns=("idx",),
            value_columns=branch_values,
            source=f"branch base case scenario {scenario_id}",
        ),
    }
    if gen_rows is not None:
        gen_values = [
            "bus",
            *[
                name
                for name in _GEN_BASE_OPTIONAL_COLUMNS
                if name in gen_rows.columns
            ],
        ]
        payload["gen"] = _frame_digest(
            gen_rows,
            key_columns=("idx",),
            value_columns=gen_values,
            source=f"generator base case scenario {scenario_id}",
        )

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_profile_id(
    bus_rows: pd.DataFrame,
    *,
    scenario_id: int,
) -> str:
    return "sha256:" + _frame_digest(
        bus_rows,
        key_columns=("bus",),
        value_columns=("Pd", "Qd"),
        source=f"load profile scenario {scenario_id}",
    )


def _contingency_family(
    branch_rows: pd.DataFrame,
    *,
    scenario_id: int,
) -> tuple[str, ...]:
    status = pd.to_numeric(branch_rows["br_status"], errors="coerce")
    if status.isna().any():
        raise ValueError(
            f"Scenario {scenario_id} has invalid br_status values."
        )
    outaged = sorted(
        int(value)
        for value in branch_rows.loc[status <= 0.0, "idx"].tolist()
    )
    if not outaged:
        return ("none",)
    return tuple(f"branch:{branch_id}" for branch_id in outaged)


def build_scenario_lineages(
    *,
    raw_dir: str | Path,
    scenario_ids: Iterable[int],
) -> dict[int, PhysicalLineage]:
    raw_path = Path(raw_dir)
    requested = sorted({int(value) for value in scenario_ids})
    if not requested:
        raise ValueError("scenario_ids must not be empty.")

    bus_path = raw_path / "bus_data.parquet"
    branch_path = raw_path / "branch_data.parquet"
    gen_path = raw_path / "gen_data.parquet"
    buses = _read_parquet_columns(
        bus_path,
        required=_BUS_REQUIRED_COLUMNS,
        optional=_BUS_BASE_OPTIONAL_COLUMNS,
    )
    branches = _read_parquet_columns(
        branch_path,
        required=_BRANCH_REQUIRED_COLUMNS,
        optional=_BRANCH_BASE_OPTIONAL_COLUMNS,
    )
    generators = None
    if gen_path.is_file():
        generators = _read_parquet_columns(
            gen_path,
            required=_GEN_REQUIRED_COLUMNS,
            optional=_GEN_BASE_OPTIONAL_COLUMNS,
        )

    lineages: dict[int, PhysicalLineage] = {}
    for scenario_id in requested:
        bus_rows = _scenario_subset(
            buses,
            scenario_id,
            source=bus_path,
        )
        branch_rows = _scenario_subset(
            branches,
            scenario_id,
            source=branch_path,
        )
        gen_rows = (
            None
            if generators is None
            else _scenario_subset(
                generators,
                scenario_id,
                source=gen_path,
            )
        )
        lineages[scenario_id] = PhysicalLineage.build(
            base_case_id=_base_case_id(
                bus_rows=bus_rows,
                branch_rows=branch_rows,
                gen_rows=gen_rows,
                scenario_id=scenario_id,
            ),
            load_profile_id=_load_profile_id(
                bus_rows,
                scenario_id=scenario_id,
            ),
            contingency_family_id=_contingency_family(
                branch_rows,
                scenario_id=scenario_id,
            ),
            source=f"raw scenario {scenario_id}",
        )
    return lineages


def _existing_lineage_columns(frame: pd.DataFrame) -> set[str]:
    return set(LINEAGE_COLUMNS) & set(frame.columns)


def _attach_lineages(
    frame: pd.DataFrame,
    *,
    scenario_column: str,
    lineages: Mapping[int, PhysicalLineage],
    source: str,
) -> pd.DataFrame:
    if scenario_column not in frame.columns:
        raise ValueError(
            f"{source} is missing {scenario_column}."
        )
    existing = _existing_lineage_columns(frame)
    if existing and existing != set(LINEAGE_COLUMNS):
        missing = sorted(set(LINEAGE_COLUMNS) - existing)
        raise ValueError(
            f"{source} has partial physical lineage columns: missing {missing}."
        )

    result = frame.copy()
    scenario_ids = result[scenario_column].map(
        lambda value: _coerce_scenario_id(value, source=source)
    )
    missing_scenarios = sorted(set(scenario_ids) - set(lineages))
    if missing_scenarios:
        raise ValueError(
            f"{source} has scenarios without physical lineage: "
            f"{missing_scenarios[:20]}."
        )

    if existing:
        for index, row in result.iterrows():
            scenario_id = int(scenario_ids.loc[index])
            observed = require_physical_lineage(
                row.to_dict(),
                source=f"{source} row {index}",
            )
            expected = lineages[scenario_id]
            if observed.fingerprint != expected.fingerprint:
                raise ValueError(
                    f"{source} row {index} physical lineage does not match "
                    f"raw scenario {scenario_id}."
                )

    for column in LINEAGE_COLUMNS:
        result[column] = [
            lineages[int(scenario_id)].as_dict()[column]
            for scenario_id in scenario_ids
        ]
    return result


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def annotate_transitions_csv(
    *,
    transitions_csv: str | Path,
    raw_dir: str | Path,
) -> dict[int, PhysicalLineage]:
    path = Path(transitions_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Transitions CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Transitions CSV is empty: {path}")
    if "scenario_id" not in frame.columns:
        raise ValueError(f"Transitions CSV is missing scenario_id: {path}")

    scenario_ids = [
        _coerce_scenario_id(value, source=str(path))
        for value in frame["scenario_id"]
    ]
    lineages = build_scenario_lineages(
        raw_dir=raw_dir,
        scenario_ids=scenario_ids,
    )
    annotated = _attach_lineages(
        frame,
        scenario_column="scenario_id",
        lineages=lineages,
        source=str(path),
    )
    _write_csv_atomic(annotated, path)
    return lineages


def _lineages_from_transitions(
    transitions: pd.DataFrame,
    *,
    source: Path,
) -> dict[int, PhysicalLineage]:
    if transitions.empty:
        raise ValueError(f"Transitions CSV is empty: {source}")
    if "scenario_id" not in transitions.columns:
        raise ValueError(f"Transitions CSV is missing scenario_id: {source}")
    missing = sorted(set(LINEAGE_COLUMNS) - set(transitions.columns))
    if missing:
        raise ValueError(
            f"Transitions CSV is missing physical lineage columns: {missing}. "
            f"File: {source}"
        )

    lineages: dict[int, PhysicalLineage] = {}
    for index, row in transitions.iterrows():
        scenario_id = _coerce_scenario_id(
            row["scenario_id"],
            source=f"{source} row {index}",
        )
        lineage = require_physical_lineage(
            row.to_dict(),
            source=f"{source} row {index}",
        )
        previous = lineages.get(scenario_id)
        if previous is not None and previous.fingerprint != lineage.fingerprint:
            raise ValueError(
                f"Scenario {scenario_id} maps to multiple physical lineages "
                f"in {source}."
            )
        lineages[scenario_id] = lineage
    return lineages


def annotate_examples_csv(
    *,
    examples_csv: str | Path,
    transitions_csv: str | Path,
) -> None:
    examples_path = Path(examples_csv)
    transitions_path = Path(transitions_csv)
    if not examples_path.is_file():
        raise FileNotFoundError(f"Examples CSV not found: {examples_path}")
    if not transitions_path.is_file():
        raise FileNotFoundError(
            f"Transitions CSV not found: {transitions_path}"
        )

    examples = pd.read_csv(examples_path)
    transitions = pd.read_csv(transitions_path)
    if examples.empty:
        raise ValueError(f"Examples CSV is empty: {examples_path}")
    lineages = _lineages_from_transitions(
        transitions,
        source=transitions_path,
    )
    annotated = _attach_lineages(
        examples,
        scenario_column="scenario_id",
        lineages=lineages,
        source=str(examples_path),
    )
    _write_csv_atomic(annotated, examples_path)


def validate_lineage_columns(
    frame: pd.DataFrame,
    *,
    source: str | Path,
) -> None:
    missing = sorted(set(LINEAGE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source} is missing physical lineage columns: {missing}."
        )
    for index, row in frame.iterrows():
        lineage = require_physical_lineage(
            row.to_dict(),
            source=f"{source} row {index}",
        )
        version = int(row["lineage_contract_version"])
        if version != PHYSICAL_LINEAGE_CONTRACT_VERSION:
            raise ValueError(
                f"{source} row {index} has unsupported physical lineage "
                f"contract version {version}."
            )
        if lineage.fingerprint != str(
            row[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
        ).strip().lower():
            raise ValueError(
                f"{source} row {index} has invalid physical lineage."
            )

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import (
    physics_provenance,
    require_exact_contract_version,
    require_physics_provenance,
)
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
    sha256_json,
)

PHYSICAL_LINEAGE_CONTRACT_VERSION = 1
PHYSICAL_LINEAGE_FINGERPRINT_FIELD = "physical_lineage_fingerprint"
_REQUIRED_FIELDS = (
    "base_case_id",
    "load_profile_id",
    "contingency_family_id",
)


def _source_label(source: str | None) -> str:
    text = str(source or "").strip()
    return text or "physical lineage"


def _normalize_identifier(
    value: object,
    *,
    name: str,
    source: str,
) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{source} has invalid {name}: {value!r}.")

    if isinstance(value, Integral):
        return str(int(value))

    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{source} has invalid {name}: {value!r}.")
        return str(int(number))

    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "<na>"}:
        raise ValueError(f"{source} has invalid {name}: {value!r}.")

    try:
        number = Decimal(text)
    except InvalidOperation:
        return text.casefold()

    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{source} has invalid {name}: {value!r}.")
    return str(int(number))


def _contingency_items(value: object, *, source: str) -> list[object]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{source} has an empty contingency_family_id.")
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source} has invalid contingency_family_id JSON."
                ) from exc
            if not isinstance(decoded, list):
                raise ValueError(
                    f"{source} contingency_family_id JSON must be a list."
                )
            return decoded
        return text.split(",") if "," in text else [text]

    if isinstance(value, Mapping):
        raise ValueError(
            f"{source} contingency_family_id must not be a mapping."
        )

    if isinstance(value, Iterable) and not isinstance(
        value,
        (bytes, bytearray),
    ):
        return list(value)

    return [value]


def normalize_contingency_family(
    value: object,
    *,
    source: str | None = None,
) -> str:
    label = _source_label(source)
    items = _contingency_items(value, source=label)
    normalized = {
        _normalize_identifier(
            item,
            name="contingency_family_id item",
            source=label,
        )
        for item in items
    }
    if not normalized:
        raise ValueError(f"{label} has an empty contingency_family_id.")
    return ",".join(sorted(normalized))


def _canonical_payload(
    *,
    base_case_id: object,
    load_profile_id: object,
    contingency_family_id: object,
    source: str | None = None,
) -> dict[str, object]:
    label = _source_label(source)
    return {
        "lineage_contract_version": PHYSICAL_LINEAGE_CONTRACT_VERSION,
        "base_case_id": _normalize_identifier(
            base_case_id,
            name="base_case_id",
            source=label,
        ),
        "load_profile_id": _normalize_identifier(
            load_profile_id,
            name="load_profile_id",
            source=label,
        ),
        "contingency_family_id": normalize_contingency_family(
            contingency_family_id,
            source=label,
        ),
    }


def physical_lineage_fingerprint(
    *,
    base_case_id: object,
    load_profile_id: object,
    contingency_family_id: object,
    source: str | None = None,
) -> str:
    payload = _canonical_payload(
        base_case_id=base_case_id,
        load_profile_id=load_profile_id,
        contingency_family_id=contingency_family_id,
        source=source,
    )
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PhysicalLineage:
    base_case_id: str
    load_profile_id: str
    contingency_family_id: str
    fingerprint: str

    @classmethod
    def build(
        cls,
        *,
        base_case_id: object,
        load_profile_id: object,
        contingency_family_id: object,
        source: str | None = None,
    ) -> "PhysicalLineage":
        payload = _canonical_payload(
            base_case_id=base_case_id,
            load_profile_id=load_profile_id,
            contingency_family_id=contingency_family_id,
            source=source,
        )
        fingerprint = physical_lineage_fingerprint(
            base_case_id=payload["base_case_id"],
            load_profile_id=payload["load_profile_id"],
            contingency_family_id=payload["contingency_family_id"],
            source=source,
        )
        return cls(
            base_case_id=str(payload["base_case_id"]),
            load_profile_id=str(payload["load_profile_id"]),
            contingency_family_id=str(payload["contingency_family_id"]),
            fingerprint=fingerprint,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "lineage_contract_version": PHYSICAL_LINEAGE_CONTRACT_VERSION,
            "base_case_id": self.base_case_id,
            "load_profile_id": self.load_profile_id,
            "contingency_family_id": self.contingency_family_id,
            PHYSICAL_LINEAGE_FINGERPRINT_FIELD: self.fingerprint,
        }


def physical_lineage_from_row(
    row: Mapping[str, Any],
    *,
    source: str | None = None,
) -> PhysicalLineage:
    label = _source_label(source)
    if not isinstance(row, Mapping):
        raise ValueError(f"{label} row must be a mapping.")

    missing = [field for field in _REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"{label} is missing physical lineage fields: "
            + ", ".join(missing)
            + "."
        )

    return PhysicalLineage.build(
        base_case_id=row["base_case_id"],
        load_profile_id=row["load_profile_id"],
        contingency_family_id=row["contingency_family_id"],
        source=label,
    )


def _require_fingerprint(value: object, *, source: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(
            f"{source} has invalid {PHYSICAL_LINEAGE_FINGERPRINT_FIELD}."
        )
    return text


def require_physical_lineage(
    row: Mapping[str, Any],
    *,
    source: str | None = None,
) -> PhysicalLineage:
    label = _source_label(source)
    lineage = physical_lineage_from_row(row, source=label)
    if PHYSICAL_LINEAGE_FINGERPRINT_FIELD not in row:
        raise ValueError(
            f"{label} is missing {PHYSICAL_LINEAGE_FINGERPRINT_FIELD}."
        )
    declared = _require_fingerprint(
        row[PHYSICAL_LINEAGE_FINGERPRINT_FIELD],
        source=label,
    )
    if declared != lineage.fingerprint:
        raise ValueError(
            f"{label} physical lineage fingerprint mismatch: "
            f"expected {lineage.fingerprint}, observed {declared}."
        )
    return lineage


def require_one_lineage_per_scenario(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str | None = None,
) -> dict[str, str]:
    label = _source_label(source)
    assignments: dict[str, str] = {}

    for index, row in enumerate(rows):
        row_source = f"{label} row {index}"
        if "scenario_id" not in row:
            raise ValueError(f"{row_source} is missing scenario_id.")
        scenario_id = _normalize_identifier(
            row["scenario_id"],
            name="scenario_id",
            source=row_source,
        )
        lineage = physical_lineage_from_row(row, source=row_source)
        declared = row.get(PHYSICAL_LINEAGE_FINGERPRINT_FIELD)
        if declared is not None:
            require_physical_lineage(row, source=row_source)

        previous = assignments.get(scenario_id)
        if previous is not None and previous != lineage.fingerprint:
            raise ValueError(
                f"{label} scenario_id={scenario_id!r} maps "
                "to multiple physical lineages."
            )
        assignments[scenario_id] = lineage.fingerprint

    if not assignments:
        raise ValueError(f"{label} contains no rows.")
    return assignments


def load_physical_lineages(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str | None = None,
) -> dict[str, PhysicalLineage]:
    label = _source_label(source)
    lineages: dict[str, PhysicalLineage] = {}

    for index, row in enumerate(rows):
        row_source = f"{label} row {index}"
        lineage = physical_lineage_from_row(row, source=row_source)
        declared = row.get(PHYSICAL_LINEAGE_FINGERPRINT_FIELD)
        if declared is not None:
            require_physical_lineage(row, source=row_source)
        lineages[lineage.fingerprint] = lineage

    if not lineages:
        raise ValueError(f"{label} contains no rows.")
    return lineages


LINEAGE_COLUMNS: tuple[str, ...] = (
    "lineage_contract_version",
    "base_case_id",
    "load_profile_id",
    "contingency_family_id",
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
)
_BUS_REQUIRED_COLUMNS: tuple[str, ...] = ("scenario", "bus", "Pd", "Qd")
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
_GEN_REQUIRED_COLUMNS: tuple[str, ...] = ("scenario", "idx", "bus")
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
            raise ValueError(
                f"Non-finite physical value in {source}: {value!r}."
            )
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

    ordered = frame[columns].sort_values(list(key_columns), kind="mergesort")
    payload = [
        [_canonical_scalar(value, source=source) for value in row]
        for row in ordered.itertuples(index=False, name=None)
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scenario_groups(
    frame: pd.DataFrame,
    scenario_ids: Iterable[int],
    *,
    source: Path,
) -> dict[int, pd.DataFrame]:
    requested = {int(value) for value in scenario_ids}
    scenario_values = frame["scenario"].map(
        lambda value: _coerce_scenario_id(value, source=str(source))
    )
    normalized = frame.copy()
    normalized["__scenario_id"] = scenario_values
    selected = normalized.loc[normalized["__scenario_id"].isin(requested)]
    groups = {
        int(scenario_id): group.drop(columns="__scenario_id").copy()
        for scenario_id, group in selected.groupby("__scenario_id", sort=False)
    }
    missing = sorted(requested - set(groups))
    if missing:
        raise ValueError(
            "Scenarios are missing from physical data "
            f"{source}: {missing[:20]}."
        )
    return groups


def _base_case_id(
    *,
    bus_rows: pd.DataFrame,
    branch_rows: pd.DataFrame,
    gen_rows: pd.DataFrame | None,
    scenario_id: int,
) -> str:
    bus_values = [
        name for name in _BUS_BASE_OPTIONAL_COLUMNS if name in bus_rows.columns
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

    bus_groups = _scenario_groups(buses, requested, source=bus_path)
    branch_groups = _scenario_groups(branches, requested, source=branch_path)
    generator_groups = (
        None
        if generators is None
        else _scenario_groups(generators, requested, source=gen_path)
    )

    lineages: dict[int, PhysicalLineage] = {}
    for scenario_id in requested:
        bus_rows = bus_groups[scenario_id]
        branch_rows = branch_groups[scenario_id]
        gen_rows = (
            None
            if generator_groups is None
            else generator_groups[scenario_id]
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
        raise ValueError(f"{source} is missing {scenario_column}.")
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


VALIDATION_SNAPSHOT_SCHEMA_VERSION = 1
_VALIDATION_SPLIT = "validation"


@dataclass(frozen=True, slots=True)
class ValidationSnapshot:
    frame: pd.DataFrame
    metadata: dict[str, Any]


def _fingerprints(frame: pd.DataFrame) -> set[str]:
    if PHYSICAL_LINEAGE_FINGERPRINT_FIELD not in frame.columns:
        raise ValueError(
            "Validation snapshot is missing physical lineage fingerprints."
        )
    return {
        str(value).strip().lower()
        for value in frame[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
    }


def _validation_assignments(manifest: Mapping[str, Any]) -> set[str]:
    assignments = manifest.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("Physical split manifest has no assignments.")
    result = {
        str(fingerprint).strip().lower()
        for fingerprint, entry in assignments.items()
        if isinstance(entry, Mapping)
        and str(entry.get("split", "")).strip() == _VALIDATION_SPLIT
    }
    if not result:
        raise ValueError("Physical split manifest has no validation lineages.")
    return result


def load_validation_snapshot(
    *,
    csv_path: str | Path,
    metadata_path: str | Path,
    manifest: Mapping[str, Any],
    physics_config: PhysicsConfig,
) -> ValidationSnapshot | None:
    csv_path = Path(csv_path)
    metadata_path = Path(metadata_path)
    exists = (csv_path.is_file(), metadata_path.is_file())
    if exists == (False, False):
        return None
    if exists != (True, True):
        raise FileNotFoundError(
            "Physical validation snapshot is incomplete: "
            f"csv={csv_path.is_file()}, metadata={metadata_path.is_file()}."
        )

    metadata = load_json(metadata_path)
    require_exact_contract_version(
        metadata.get("schema_version"),
        expected=VALIDATION_SNAPSHOT_SCHEMA_VERSION,
        name="physical validation snapshot schema",
        source=str(metadata_path),
        regeneration_command="remove the run directory and restart self-play",
    )
    require_physics_provenance(
        metadata,
        source=str(metadata_path),
        expected_physics_config=physics_config,
    )
    if metadata.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError(
            f"Physical validation snapshot hash mismatch: {csv_path}."
        )

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Physical validation snapshot is empty: {csv_path}.")
    validate_lineage_columns(frame, source=csv_path)
    observed = _fingerprints(frame)
    assigned = _validation_assignments(manifest)
    unexpected = sorted(observed - assigned)
    if unexpected:
        raise ValueError(
            "Physical validation snapshot contains non-validation lineages: "
            f"{unexpected[:5]}."
        )

    recorded = metadata.get("physical_lineage_fingerprints")
    if not isinstance(recorded, list):
        raise ValueError(
            f"Validation snapshot metadata has no lineage list: {metadata_path}."
        )
    normalized_recorded = sorted(str(value).strip().lower() for value in recorded)
    if normalized_recorded != sorted(observed):
        raise ValueError(
            f"Validation snapshot lineage list mismatch: {metadata_path}."
        )
    if metadata.get("physical_lineages_sha256") != sha256_json(
        normalized_recorded
    ):
        raise ValueError(
            f"Validation snapshot lineage hash mismatch: {metadata_path}."
        )
    if int(metadata.get("row_count", -1)) != len(frame):
        raise ValueError(
            f"Validation snapshot row count mismatch: {metadata_path}."
        )
    return ValidationSnapshot(frame=frame, metadata=metadata)


def update_validation_snapshot(
    *,
    current_validation: pd.DataFrame,
    manifest: Mapping[str, Any],
    physics_config: PhysicsConfig,
    iteration: int,
    csv_path: str | Path,
    metadata_path: str | Path,
) -> ValidationSnapshot:
    iteration = int(iteration)
    if iteration <= 0:
        raise ValueError("iteration must be > 0")

    csv_path = Path(csv_path)
    metadata_path = Path(metadata_path)
    assigned = _validation_assignments(manifest)
    loaded = load_validation_snapshot(
        csv_path=csv_path,
        metadata_path=metadata_path,
        manifest=manifest,
        physics_config=physics_config,
    )

    if loaded is None:
        assignments = manifest.get("assignments")
        if not isinstance(assignments, Mapping):
            raise ValueError("Physical split manifest has no assignments.")
        prior_validation = any(
            isinstance(entry, Mapping)
            and str(entry.get("split", "")).strip() == _VALIDATION_SPLIT
            and int(entry.get("assigned_iteration", iteration)) < iteration
            for entry in assignments.values()
        )
        if prior_validation:
            raise FileNotFoundError(
                "Persistent physical validation snapshot is missing for an "
                "existing split. Restore the run artifact or restart the run."
            )
        if current_validation.empty:
            raise ValueError(
                "Cannot create physical validation snapshot without "
                "validation examples."
            )
        validate_lineage_columns(
            current_validation,
            source="current validation replay",
        )
        current_fingerprints = _fingerprints(current_validation)
        missing = sorted(assigned - current_fingerprints)
        if missing:
            raise ValueError(
                "New validation lineages have no replay examples: "
                f"{missing[:5]}."
            )
        snapshot = current_validation.copy()
        created_iteration = iteration
    else:
        snapshot = loaded.frame.copy()
        existing = _fingerprints(snapshot)
        missing = assigned - existing
        if missing:
            if current_validation.empty:
                raise ValueError(
                    "New validation lineages have no replay examples: "
                    f"{sorted(missing)[:5]}."
                )
            validate_lineage_columns(
                current_validation,
                source="current validation replay",
            )
            current_fingerprints = _fingerprints(current_validation)
            unavailable = sorted(missing - current_fingerprints)
            if unavailable:
                raise ValueError(
                    "New validation lineages have no replay examples: "
                    f"{unavailable[:5]}."
                )
            additions = current_validation.loc[
                current_validation[
                    PHYSICAL_LINEAGE_FINGERPRINT_FIELD
                ].astype(str).str.strip().str.lower().isin(missing)
            ].copy()
            snapshot = pd.concat([snapshot, additions], ignore_index=True)
        created_iteration = int(
            loaded.metadata.get("created_iteration", iteration)
        )

    snapshot_fingerprints = _fingerprints(snapshot)
    if snapshot_fingerprints != assigned:
        missing = sorted(assigned - snapshot_fingerprints)
        unexpected = sorted(snapshot_fingerprints - assigned)
        raise ValueError(
            "Physical validation snapshot assignment mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}."
        )

    _write_csv_atomic(snapshot, csv_path)
    metadata: dict[str, Any] = {
        "schema_version": VALIDATION_SNAPSHOT_SCHEMA_VERSION,
        **physics_provenance(physics_config),
        "created_iteration": created_iteration,
        "last_updated_iteration": iteration,
        "row_count": int(len(snapshot)),
        "physical_lineage_count": len(snapshot_fingerprints),
        "physical_lineage_fingerprints": sorted(snapshot_fingerprints),
        "physical_lineages_sha256": sha256_json(sorted(snapshot_fingerprints)),
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "assignment_strategy": manifest.get("assignment_strategy"),
        "source_hashes": dict(manifest.get("source_hashes", {})),
    }
    save_json(metadata, metadata_path)
    return ValidationSnapshot(frame=snapshot, metadata=metadata)

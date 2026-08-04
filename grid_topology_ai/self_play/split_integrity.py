from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.self_play.artifacts import sha256_file
from grid_topology_ai.self_play.physical_lineage import (
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
)

_RAW_LINEAGE_FILES = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
)


def physical_split_source_hashes(
    *,
    transitions_csv: str | Path,
    raw_dir: str | Path,
) -> dict[str, str]:
    transitions_path = Path(transitions_csv)
    raw_path = Path(raw_dir)
    hashes = {
        "pool_transitions": sha256_file(transitions_path),
    }
    raw_files = [raw_path / name for name in _RAW_LINEAGE_FILES]
    existing = [path for path in raw_files if path.is_file()]
    if existing:
        required = raw_files[:2]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Physical lineage raw files are incomplete: "
                + ", ".join(str(path) for path in missing)
            )
        for path in existing:
            hashes[f"pool_raw:{path.name}"] = sha256_file(path)
    return hashes


def require_exact_source_hashes(
    manifest: Mapping[str, Any],
    *,
    expected: Mapping[str, str],
    source: str | Path,
) -> None:
    recorded = manifest.get("source_hashes")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"Physical split source_hashes are missing: {source}.")
    normalized = {
        str(name): str(value).strip().lower()
        for name, value in recorded.items()
    }
    expected_normalized = {
        str(name): str(value).strip().lower()
        for name, value in expected.items()
    }
    if normalized != expected_normalized:
        missing = sorted(set(expected_normalized) - set(normalized))
        unexpected = sorted(set(normalized) - set(expected_normalized))
        changed = sorted(
            name
            for name in set(normalized) & set(expected_normalized)
            if normalized[name] != expected_normalized[name]
        )
        raise ValueError(
            "Physical split source provenance mismatch for "
            f"{source}: missing={missing}, unexpected={unexpected}, "
            f"changed={changed}. Remove the run directory and restart."
        )


def _scenario_id(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("scenario_id must be an integer.")
    number = float(value)
    if not number.is_integer():
        raise ValueError("scenario_id must be an integer.")
    return int(number)


def manifest_scenario_lineages(
    manifest: Mapping[str, Any],
    *,
    source: str | Path,
) -> dict[int, str]:
    assignments = manifest.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError(f"Physical split manifest has no assignments: {source}.")
    result: dict[int, str] = {}
    for fingerprint, raw in assignments.items():
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Physical split assignment must be an object: {source}."
            )
        scenario_ids = raw.get("scenario_ids")
        if not isinstance(scenario_ids, list):
            raise ValueError(
                f"Physical split scenario_ids must be a list: {source}."
            )
        normalized_fingerprint = str(fingerprint).strip().lower()
        for value in scenario_ids:
            scenario_id = _scenario_id(value)
            previous = result.get(scenario_id)
            if previous is not None and previous != normalized_fingerprint:
                raise ValueError(
                    f"Scenario {scenario_id} maps to multiple physical "
                    f"lineages in {source}: {previous}, "
                    f"{normalized_fingerprint}."
                )
            result[scenario_id] = normalized_fingerprint
    return result


def require_current_scenario_consistency(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    source: str | Path,
) -> None:
    if "scenario_id" not in frame.columns:
        raise ValueError(f"{source} is missing scenario_id.")
    if PHYSICAL_LINEAGE_FINGERPRINT_FIELD not in frame.columns:
        raise ValueError(
            f"{source} is missing {PHYSICAL_LINEAGE_FINGERPRINT_FIELD}."
        )
    recorded = manifest_scenario_lineages(manifest, source=source)
    current: dict[int, str] = {}
    for scenario_value, fingerprint_value in zip(
        frame["scenario_id"],
        frame[PHYSICAL_LINEAGE_FINGERPRINT_FIELD],
        strict=True,
    ):
        scenario_id = _scenario_id(scenario_value)
        fingerprint = str(fingerprint_value).strip().lower()
        previous = current.get(scenario_id)
        if previous is not None and previous != fingerprint:
            raise ValueError(
                f"Scenario {scenario_id} maps to multiple physical lineages "
                f"in {source}."
            )
        current[scenario_id] = fingerprint
        recorded_fingerprint = recorded.get(scenario_id)
        if (
            recorded_fingerprint is not None
            and recorded_fingerprint != fingerprint
        ):
            raise ValueError(
                f"Scenario {scenario_id} changed physical lineage: "
                f"recorded={recorded_fingerprint}, current={fingerprint}."
            )

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from grid_topology_ai.self_play.lineage_artifacts import (
    validate_lineage_columns,
)
from grid_topology_ai.self_play.physical_lineage import (
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
)
from grid_topology_ai.self_play.physical_split import VALIDATION_SPLIT

VALIDATION_SNAPSHOT_SCHEMA_VERSION = 1


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
        and str(entry.get("split", "")).strip() == VALIDATION_SPLIT
    }
    if not result:
        raise ValueError("Physical split manifest has no validation lineages.")
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
    normalized_recorded = sorted(
        str(value).strip().lower()
        for value in recorded
    )
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
            and str(entry.get("split", "")).strip() == VALIDATION_SPLIT
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
            snapshot = pd.concat(
                [snapshot, additions],
                ignore_index=True,
            )
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
        "physical_lineages_sha256": sha256_json(
            sorted(snapshot_fingerprints)
        ),
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "assignment_strategy": manifest.get("assignment_strategy"),
        "source_hashes": dict(manifest.get("source_hashes", {})),
    }
    save_json(metadata, metadata_path)
    return ValidationSnapshot(frame=snapshot, metadata=metadata)

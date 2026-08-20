from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.self_play.artifacts import (
    save_json,
    sha256_file,
    sha256_json,
)
from grid_topology_ai.self_play.lineage_artifacts import (
    validate_lineage_columns,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.physical_lineage import (
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
)
from grid_topology_ai.self_play.physical_split import (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    assign_physical_split,
    load_physical_split_manifest,
    manifest_scenario_lineages,
    physical_split_source_hashes,
    require_current_scenario_consistency,
    require_exact_source_hashes,
)
from grid_topology_ai.self_play.replay import (
    EpisodeSamplingMixin,
    _save_manifest,
)
from grid_topology_ai.self_play.validation_snapshot import (
    update_validation_snapshot,
)


class _ReplayBuffer(Protocol):
    buffer: list[dict[str, Any]]
    config: Any
    physics_config: PhysicsConfig

    def _split_fresh_old(
        self,
        *,
        current_iteration: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


def _fingerprints_for_split(
    manifest: Mapping[str, Any],
    split: str,
) -> set[str]:
    assignments = manifest.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("Physical split manifest has no assignments.")

    fingerprints = {
        str(fingerprint).strip().lower()
        for fingerprint, entry in assignments.items()
        if isinstance(entry, Mapping)
        and str(entry.get("split", "")).strip() == split
    }
    if not fingerprints:
        raise ValueError(
            f"Physical split manifest contains no {split} lineages."
        )
    return fingerprints


def _frame_fingerprints(frame: pd.DataFrame) -> set[str]:
    if PHYSICAL_LINEAGE_FINGERPRINT_FIELD not in frame.columns:
        raise ValueError(
            "Replay examples are missing physical lineage fingerprints."
        )
    return {
        str(value).strip().lower()
        for value in frame[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
    }


def _scenario_count(frame: pd.DataFrame) -> int:
    if "scenario_id" not in frame.columns:
        raise ValueError("Replay examples are missing scenario_id.")
    return int(frame["scenario_id"].nunique(dropna=False))


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _split_active_replay(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_lineage_columns(frame, source="replay buffer")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("Physical split manifest has no assignments.")
    split_by_fingerprint = {
        str(fingerprint).strip().lower(): str(entry.get("split", "")).strip()
        for fingerprint, entry in assignments.items()
        if isinstance(entry, Mapping)
    }
    fingerprints = (
        frame[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    missing = sorted(set(fingerprints) - set(split_by_fingerprint))
    if missing:
        raise ValueError(
            "Replay buffer contains unassigned physical lineages: "
            f"{missing[:5]}."
        )
    labels = fingerprints.map(split_by_fingerprint)
    invalid = sorted(set(labels) - {TRAIN_SPLIT, VALIDATION_SPLIT})
    if invalid:
        raise ValueError(
            f"Physical split manifest contains invalid labels: {invalid}."
        )
    train = frame.loc[labels == TRAIN_SPLIT].copy()
    validation = frame.loc[labels == VALIDATION_SPLIT].copy()
    if train.empty:
        raise ValueError(
            "Active replay buffer contains no physical training lineages."
        )
    return train, validation


def _refresh_prediction_errors(
    replay_buffer: _ReplayBuffer,
    *,
    current_iteration: int,
) -> dict[str, Any] | None:
    refresh = getattr(replay_buffer, "_refresh_prediction_errors", None)
    if not callable(refresh):
        return None
    result = refresh(current_iteration=current_iteration)
    if not isinstance(result, Mapping):
        raise TypeError("Replay prediction-error refresh must return a mapping.")
    return dict(result)


def _export_train_batch(
    replay_buffer: _ReplayBuffer,
    *,
    train_rows: list[dict[str, Any]],
    active_train_fingerprints: set[str],
    output_path: Path,
    current_iteration: int,
    n_examples: int | None,
    fresh_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if not train_rows:
        raise ValueError("Physical split contains no replay rows for training.")

    error_refresh = _refresh_prediction_errors(
        replay_buffer,
        current_iteration=current_iteration,
    )
    original_buffer = replay_buffer.buffer
    replay_buffer.buffer = train_rows
    try:
        metadata = EpisodeSamplingMixin.export_mixed_batch(
            replay_buffer,
            output_path=output_path,
            current_iteration=current_iteration,
            n_examples=n_examples,
            fresh_fraction=fresh_fraction,
            seed=seed,
        )
    finally:
        replay_buffer.buffer = original_buffer

    metadata.update(
        {
            "eligible_examples": len(train_rows),
            "eligible_physical_lineage_count": len(
                active_train_fingerprints
            ),
            "eligible_split": TRAIN_SPLIT,
        }
    )
    if error_refresh is not None:
        prediction_errors = getattr(replay_buffer, "prediction_errors", {})
        metadata.update(
            {
                "prediction_error_entries": len(prediction_errors),
                "prediction_error_refresh": error_refresh,
            }
        )
    _save_manifest(metadata, output_path.with_suffix(".metadata.json"))
    return metadata


def prepare_physical_iteration_split(
    *,
    replay_buffer: _ReplayBuffer,
    paths: SelfPlayPaths,
    physics_config: PhysicsConfig,
    iteration: int,
    split_seed: int,
    sampling_seed: int,
    validation_fraction: float,
    min_validation_lineages: int,
    n_examples: int | None,
    fresh_fraction: float,
    train_batch_path: str | Path,
    train_examples_path: str | Path,
    validation_examples_path: str | Path,
    metadata_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not replay_buffer.buffer:
        raise ValueError("Cannot split an empty replay buffer.")

    train_batch_path = Path(train_batch_path)
    train_examples_path = Path(train_examples_path)
    validation_examples_path = Path(validation_examples_path)
    metadata_path = Path(metadata_path)

    replay_frame = pd.DataFrame(replay_buffer.buffer)
    source_hashes = physical_split_source_hashes(
        transitions_csv=paths.pool_transitions_csv,
        raw_dir=paths.pool_raw_dir,
    )

    previous = load_physical_split_manifest(
        paths.physical_split_manifest,
        physics_config=physics_config,
        seed=int(split_seed),
        validation_fraction=float(validation_fraction),
        min_validation_lineages=int(min_validation_lineages),
        source_hashes=source_hashes,
    )
    previous_assignments = (
        set(previous["assignments"])
        if previous is not None
        else set()
    )
    if previous is not None:
        require_exact_source_hashes(
            previous,
            expected=source_hashes,
            source=paths.physical_split_manifest,
        )
        manifest_scenario_lineages(
            previous,
            source=paths.physical_split_manifest,
        )
        require_current_scenario_consistency(
            replay_frame,
            previous,
            source="replay buffer",
        )

    manifest = assign_physical_split(
        replay_frame,
        manifest_path=paths.physical_split_manifest,
        physics_config=physics_config,
        seed=int(split_seed),
        validation_fraction=float(validation_fraction),
        min_validation_lineages=int(min_validation_lineages),
        iteration=int(iteration),
        source="replay buffer",
        source_hashes=source_hashes,
    )
    require_exact_source_hashes(
        manifest,
        expected=source_hashes,
        source=paths.physical_split_manifest,
    )
    manifest_scenario_lineages(
        manifest,
        source=paths.physical_split_manifest,
    )

    train_fingerprints = _fingerprints_for_split(manifest, TRAIN_SPLIT)
    validation_fingerprints = _fingerprints_for_split(
        manifest,
        VALIDATION_SPLIT,
    )
    train_replay, active_validation = _split_active_replay(
        replay_frame,
        manifest=manifest,
    )
    active_train_fingerprints = _frame_fingerprints(train_replay)

    validation_snapshot = update_validation_snapshot(
        current_validation=active_validation,
        manifest=manifest,
        physics_config=physics_config,
        iteration=int(iteration),
        csv_path=paths.physical_validation_snapshot,
        metadata_path=paths.physical_validation_snapshot_metadata,
    )
    validation_replay = validation_snapshot.frame

    train_rows = train_replay.to_dict(orient="records")
    train_batch_metadata = _export_train_batch(
        replay_buffer,
        train_rows=train_rows,
        active_train_fingerprints=active_train_fingerprints,
        output_path=train_batch_path,
        current_iteration=int(iteration),
        n_examples=n_examples,
        fresh_fraction=float(fresh_fraction),
        seed=int(sampling_seed),
    )

    sampled_train = pd.read_csv(train_batch_path)
    sampled_fingerprints = _frame_fingerprints(sampled_train)
    validation_row_fingerprints = _frame_fingerprints(validation_replay)
    if not sampled_fingerprints <= train_fingerprints:
        raise RuntimeError("Train replay sampler selected a validation lineage.")
    if validation_row_fingerprints != validation_fingerprints:
        raise RuntimeError(
            "Persistent validation snapshot does not cover all assigned "
            "validation lineages."
        )
    overlap = sampled_fingerprints & validation_row_fingerprints
    if overlap:
        raise RuntimeError(
            "Physical lineage leakage detected between train and validation: "
            f"{sorted(overlap)[:5]}."
        )

    train_examples_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(train_batch_path, train_examples_path)
    _write_csv_atomic(validation_replay, validation_examples_path)

    assignments = manifest["assignments"]
    new_assignments = sorted(set(assignments) - previous_assignments)
    split_metadata: dict[str, Any] = {
        "schema_version": 3,
        "split_unit": "physical_lineage",
        "assignment_strategy": manifest["assignment_strategy"],
        "persistent_manifest_path": str(paths.physical_split_manifest),
        "persistent_manifest_sha256": sha256_file(
            paths.physical_split_manifest
        ),
        "persistent_validation_snapshot_path": str(
            paths.physical_validation_snapshot
        ),
        "persistent_validation_snapshot_sha256": sha256_file(
            paths.physical_validation_snapshot
        ),
        "persistent_validation_snapshot_metadata_path": str(
            paths.physical_validation_snapshot_metadata
        ),
        "persistent_validation_snapshot_metadata_sha256": sha256_file(
            paths.physical_validation_snapshot_metadata
        ),
        "source_replay_manifest": str(paths.replay_manifest),
        "source_replay_manifest_sha256": (
            sha256_file(paths.replay_manifest)
            if paths.replay_manifest.is_file()
            else None
        ),
        "source_hashes": dict(sorted(source_hashes.items())),
        "iteration": int(iteration),
        "split_seed": int(split_seed),
        "sampling_seed": int(sampling_seed),
        "validation_fraction_target": float(validation_fraction),
        "min_validation_lineages": int(min_validation_lineages),
        "new_assignments_this_iteration": len(new_assignments),
        "new_physical_lineage_fingerprints": new_assignments,
        "total_examples": int(len(replay_frame)),
        "train_examples": int(len(sampled_train)),
        "validation_examples": int(len(validation_replay)),
        "active_validation_examples": int(len(active_validation)),
        "total_scenarios": _scenario_count(replay_frame),
        "train_scenarios": _scenario_count(sampled_train),
        "validation_scenarios": _scenario_count(validation_replay),
        "total_lineages": int(manifest["lineage_count"]),
        "train_lineages": int(manifest["train_lineage_count"]),
        "validation_lineages": int(
            manifest["validation_lineage_count"]
        ),
        "active_train_lineages": len(active_train_fingerprints),
        "active_validation_lineages": len(
            _frame_fingerprints(active_validation)
        ),
        "sampled_train_lineages": len(sampled_fingerprints),
        "train_csv": str(train_examples_path),
        "validation_csv": str(validation_examples_path),
        "train_csv_sha256": sha256_file(train_examples_path),
        "validation_csv_sha256": sha256_file(validation_examples_path),
        "train_physical_fingerprints_sha256": sha256_json(
            sorted(train_fingerprints)
        ),
        "validation_physical_fingerprints_sha256": sha256_json(
            sorted(validation_fingerprints)
        ),
        "validation_scenario_ids": sorted(
            int(value)
            for value in validation_replay["scenario_id"].unique()
        ),
        "validation_snapshot_created_iteration": int(
            validation_snapshot.metadata["created_iteration"]
        ),
        "validation_snapshot_last_updated_iteration": int(
            validation_snapshot.metadata["last_updated_iteration"]
        ),
    }
    save_json(split_metadata, metadata_path)
    return train_batch_metadata, split_metadata

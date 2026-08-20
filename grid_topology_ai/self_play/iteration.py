from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.evaluation.checkpoint import load_scenario_ids
from grid_topology_ai.evaluation.paired_results import (
    compare_evaluation_results,
)
from grid_topology_ai.evaluation.policy_comparison import (
    PolicyMode,
    require_policy_mode_metrics,
    require_primary_policy_mode,
)
from grid_topology_ai.self_play.acceptance import (
    accept_candidate,
    passes_confidence_gates,
    require_metrics_pf_alg,
    require_metrics_physics_config,
)
from grid_topology_ai.self_play.artifacts import (
    save_json,
    sha256_file,
    sha256_json,
)
from grid_topology_ai.self_play.checkpoint_state import promote_candidate
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
from grid_topology_ai.self_play.pool_sampling import sample_from_pool
from grid_topology_ai.self_play.pool_state import update_and_save_pool_metadata
from grid_topology_ai.self_play.replay import (
    EpisodeSamplingMixin,
    RollingReplayBuffer,
    _save_manifest,
)
from grid_topology_ai.self_play.stages import (
    run_evaluate,
    run_generate,
    run_train,
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


_MATCHING_EVALUATION_FIELDS = (
        "transitions_sha256",
        "raw_data_sha256",
        "scenario_ids_sha256",
        "task_config_sha256",
        "physics_config_fingerprint",
        "evaluation_metrics_contract_version",
        "git_revision",
        "git_dirty",
    )


@dataclass(frozen=True, slots=True)
class _SelfPlaySeeds:
    scenario_sampling: int
    mcts: int
    action_sampling: int


def _seed_from_sequence(
    sequence: np.random.SeedSequence,
) -> int:
    state = sequence.generate_state(
        1,
        dtype=np.uint64,
    )
    return int(state[0])


def _self_play_seeds(
    *,
    base_seed: int,
    iteration: int,
) -> _SelfPlaySeeds:
    root_sequence = np.random.SeedSequence(
        [
            int(base_seed),
            int(iteration),
        ]
    )

    (
        scenario_sequence,
        mcts_sequence,
        action_sequence,
    ) = root_sequence.spawn(3)

    return _SelfPlaySeeds(
        scenario_sampling=_seed_from_sequence(
            scenario_sequence
        ),
        mcts=_seed_from_sequence(
            mcts_sequence
        ),
        action_sampling=_seed_from_sequence(
            action_sequence
        ),
    )


_SELF_PLAY_EXPLORATION_NUMERIC_COLUMNS = (
    "selection_temperature",
    "policy_target_entropy",
    "policy_target_normalized_entropy",
    "mcts_legal_action_count",
    "mcts_considered_action_count",
    "mcts_visited_action_count",
    "mcts_action_coverage",
    "mcts_visited_action_coverage",
)


def _self_play_exploration_metrics(
    examples: pd.DataFrame,
) -> dict[str, int | float]:
    required_columns = (
        set(_SELF_PLAY_EXPLORATION_NUMERIC_COLUMNS)
        | {"selection_mode"}
    )
    missing_columns = sorted(
        required_columns - set(examples.columns)
    )

    if missing_columns:
        raise ValueError(
            "Self-play examples are missing exploration "
            "diagnostic columns: "
            + ", ".join(missing_columns)
        )

    if examples.empty:
        raise ValueError(
            "Cannot calculate self-play exploration metrics "
            "from an empty dataframe."
        )

    numeric = examples[
        list(_SELF_PLAY_EXPLORATION_NUMERIC_COLUMNS)
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric_values = numeric.to_numpy(
        dtype=np.float64
    )

    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric_values).all()
    ):
        raise ValueError(
            "Self-play exploration diagnostics must be finite."
        )

    selection_modes = (
        examples["selection_mode"]
        .astype(str)
        .str.strip()
    )
    invalid_modes = sorted(
        set(selection_modes)
        - {"sample", "argmax"}
    )

    if invalid_modes:
        raise ValueError(
            "Unsupported self-play selection modes: "
            + ", ".join(invalid_modes)
        )

    step_count = int(len(examples))
    sampled_steps = int(
        (selection_modes == "sample").sum()
    )

    return {
        "steps": step_count,
        "sampled_steps": sampled_steps,
        "sample_fraction": float(
            sampled_steps / step_count
        ),
        "mean_selection_temperature": float(
            numeric["selection_temperature"].mean()
        ),
        "mean_policy_target_entropy": float(
            numeric["policy_target_entropy"].mean()
        ),
        "mean_policy_target_normalized_entropy": float(
            numeric[
                "policy_target_normalized_entropy"
            ].mean()
        ),
        "mean_mcts_legal_action_count": float(
            numeric["mcts_legal_action_count"].mean()
        ),
        "mean_mcts_considered_action_count": float(
            numeric[
                "mcts_considered_action_count"
            ].mean()
        ),
        "mean_mcts_visited_action_count": float(
            numeric["mcts_visited_action_count"].mean()
        ),
        "mean_mcts_action_coverage": float(
            numeric["mcts_action_coverage"].mean()
        ),
        "min_mcts_action_coverage": float(
            numeric["mcts_action_coverage"].min()
        ),
        "mean_mcts_visited_action_coverage": float(
            numeric[
                "mcts_visited_action_coverage"
            ].mean()
        ),
        "min_mcts_visited_action_coverage": float(
            numeric[
                "mcts_visited_action_coverage"
            ].min()
        ),
    }


@dataclass(frozen=True, slots=True)
class IterationRequest:
    iteration: int
    config: SelfPlayConfig
    raw_config: Mapping[str, object]
    paths: SelfPlayPaths

    parent_checkpoint: Path

    pool_metadata: dict[str, Any]
    replay_buffer: RollingReplayBuffer

    def __post_init__(self) -> None:
        if int(self.iteration) <= 0:
            raise ValueError("iteration must be > 0")


@dataclass(frozen=True, slots=True)
class IterationResult:
    iteration: int
    accepted: bool
    status: str

    selected_scenario_ids: tuple[int, ...]

    raw_examples_csv: Path
    train_batch_csv: Path
    train_examples_csv: Path
    validation_examples_csv: Path
    split_metadata_path: Path
    candidate_checkpoint: Path
    metadata_path: Path

    parent_metrics: dict[str, object]
    candidate_metrics: dict[str, object]

    best_checkpoint: Path
    best_metrics: dict[str, object]

    pool_metadata: dict[str, Any]
    learning_curve_row: dict[str, object]


def _require_matching_evaluation_inputs(
    parent_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
) -> None:
    parent_run_info = parent_metrics.get("run_info")
    candidate_run_info = candidate_metrics.get("run_info")

    if not isinstance(parent_run_info, Mapping):
        raise ValueError(
            "Parent evaluation metrics are missing run_info."
        )

    if not isinstance(candidate_run_info, Mapping):
        raise ValueError(
            "Candidate evaluation metrics are missing run_info."
        )

    missing_fields = [
        field
        for field in _MATCHING_EVALUATION_FIELDS
        if field not in parent_run_info
        or field not in candidate_run_info
    ]

    if missing_fields:
        raise ValueError(
            "Evaluation run_info is incomplete: "
            + ", ".join(missing_fields)
        )

    mismatches = [
        field
        for field in _MATCHING_EVALUATION_FIELDS
        if parent_run_info[field]
        != candidate_run_info[field]
    ]

    if not mismatches:
        return

    details = "; ".join(
        (
            f"{field}: "
            f"parent={parent_run_info[field]!r}, "
            f"candidate={candidate_run_info[field]!r}"
        )
        for field in mismatches
    )

    raise ValueError(
        "Parent and candidate evaluations used different inputs: "
        + details
    )


def _metrics_for_policy_mode(
    metrics: Mapping[str, object],
    mode: PolicyMode,
    *,
    source: str,
) -> dict[str, object]:
    require_primary_policy_mode(
        metrics,
        mode,
        source=source,
    )

    task_config = metrics.get("task_config")
    if (
        not isinstance(task_config, Mapping)
        or task_config.get("primary_policy_mode") != mode.value
    ):
        raise ValueError(
            "Evaluation task_config primary policy mode mismatch for "
            f"{source}: expected {mode.value!r}."
        )

    mode_metrics = require_policy_mode_metrics(
        metrics,
        mode,
        source=source,
    )
    view = dict(metrics)
    view.update(mode_metrics)
    view["primary_policy_mode"] = mode.value
    return view


def _count_examples_csv(path: str | Path) -> int:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Examples CSV not found while counting rows: {path}"
        )

    try:
        examples = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Examples CSV has no readable columns: {path}"
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Could not parse examples CSV: {path}"
        ) from exc

    if examples.empty:
        raise ValueError(
            f"Examples CSV contains no rows: {path}"
        )

    return int(len(examples))


def _save_iteration_metadata(
    *,
    iteration: int,
    path: str | Path,
    accepted: bool,
    parent_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    train_batch_csv: str | Path,
    train_examples_csv: str | Path,
    validation_examples_csv: str | Path,
    split_metadata_path: str | Path,
    raw_examples_csv: str | Path | None,
    parent_metrics: dict[str, Any],
    metrics: dict[str, Any],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save reproducibility metadata for one self-play iteration.
    """

    path = Path(path)

    parent_checkpoint = Path(parent_checkpoint)
    candidate_checkpoint = Path(candidate_checkpoint)
    train_batch_csv = Path(train_batch_csv)
    train_examples_csv = Path(train_examples_csv)
    validation_examples_csv = Path(validation_examples_csv)
    split_metadata_path = Path(split_metadata_path)

    payload: dict[str, Any] = {
        "iteration": int(iteration),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": bool(accepted),
        "parent_checkpoint": str(parent_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "train_batch_csv": str(train_batch_csv),
        "train_examples_csv": str(train_examples_csv),
        "validation_examples_csv": str(validation_examples_csv),
        "split_metadata_path": str(split_metadata_path),
        "raw_examples_csv": None if raw_examples_csv is None else str(raw_examples_csv),
        "hashes": {},
        "parent_metrics": parent_metrics,
        "metrics": metrics,
        "config": config,
    }

    for name, file_path in {
        "parent_checkpoint_sha256": parent_checkpoint,
        "candidate_checkpoint_sha256": candidate_checkpoint,
        "train_batch_csv_sha256": train_batch_csv,
        "train_examples_csv_sha256": train_examples_csv,
        "validation_examples_csv_sha256": validation_examples_csv,
        "split_metadata_sha256": split_metadata_path,
    }.items():
        if file_path.exists():
            payload["hashes"][name] = sha256_file(file_path)

    if raw_examples_csv is not None:
        raw_examples_path = Path(raw_examples_csv)

        if raw_examples_path.exists():
            payload["hashes"]["raw_examples_csv_sha256"] = sha256_file(
                raw_examples_path
            )

    if extra is not None:
        payload["extra"] = extra

    save_json(payload, path)

    return path


def run_self_play_iteration(
    request: IterationRequest,
) -> IterationResult:
    iteration = int(request.iteration)
    config = request.config
    paths = request.paths
    metric_name = config.acceptance.metric

    iter_dir = paths.iteration_dir(iteration)
    iter_dir.mkdir(parents=True, exist_ok=True)

    parent_checkpoint = request.parent_checkpoint

    iteration_seed = int(config.seed) + iteration

    self_play_seeds = _self_play_seeds(
        base_seed=int(config.seed),
        iteration=iteration,
    )

    scenario_ids = sample_from_pool(
        pool_metadata=request.pool_metadata,
        n=config.n_scenarios_per_iteration,
        seed=self_play_seeds.scenario_sampling,
        current_iter=iteration,
        config=config.pool.curriculum,
    )

    selected_ids_path = iter_dir / "selected_scenario_ids.txt"
    selected_ids_path.write_text(
        "\n".join(str(value) for value in scenario_ids) + "\n",
        encoding="utf-8",
    )

    print(f"Sampled scenarios: {len(scenario_ids)}")
    print(f"Selected IDs:      {selected_ids_path}")

    raw_examples_csv = run_generate(
        project_root=paths.project_root,
        raw_dir=paths.pool_raw_dir,
        transitions_csv=paths.pool_transitions_csv,
        scenario_ids=scenario_ids,
        checkpoint=parent_checkpoint,
        output_dir=iter_dir / "raw",
        config=config.generation,
        physics_config=config.physics,
        mcts_seed=self_play_seeds.mcts,
        action_seed=self_play_seeds.action_sampling,
        iteration=iteration,
    )

    raw_examples_count = _count_examples_csv(raw_examples_csv)

    raw_examples_df = pd.read_csv(raw_examples_csv)

    exploration_metrics = (
        _self_play_exploration_metrics(
            raw_examples_df
        )
    )

    new_examples = request.replay_buffer.add_and_save_from_csv(
        examples_csv=raw_examples_csv,
        iteration=iteration,
    )

    configured_examples = config.training.examples_per_iteration
    examples_per_iteration = (
        len(request.replay_buffer)
        if configured_examples is None
        else configured_examples
    )

    train_batch_path = iter_dir / "train_batch.csv"
    train_examples_path = iter_dir / "train_examples.csv"
    validation_examples_path = iter_dir / "validation_examples.csv"
    split_metadata_path = iter_dir / "train_validation_split.json"

    train_batch_metadata, split_metadata = (
        prepare_physical_iteration_split(
            replay_buffer=request.replay_buffer,
            paths=paths,
            physics_config=config.physics,
            iteration=iteration,
            split_seed=int(config.seed),
            sampling_seed=iteration_seed,
            validation_fraction=(
                config.training.validation_fraction
            ),
            min_validation_lineages=(
                config.training.min_validation_scenarios
            ),
            n_examples=examples_per_iteration,
            fresh_fraction=float(
                config.replay_buffer.fresh_fraction
            ),
            train_batch_path=train_batch_path,
            train_examples_path=train_examples_path,
            validation_examples_path=validation_examples_path,
            metadata_path=split_metadata_path,
        )
    )

    candidate_checkpoint = run_train(
        project_root=paths.project_root,
        examples_csv=train_examples_path,
        validation_examples_csv=validation_examples_path,
        init_checkpoint=parent_checkpoint,
        output_dir=iter_dir,
        config=config.training,
        physics_config=config.physics,
        iteration=iteration,
        seed=iteration_seed,
    )

    evaluation_scenario_ids = tuple(
        load_scenario_ids(
            paths.eval_csv,
            limit=None,
        )
    )

    evaluation_dir = iter_dir / "evaluation"
    parent_evaluation_dir = evaluation_dir / "parent"
    candidate_evaluation_dir = evaluation_dir / "candidate"

    parent_metrics = run_evaluate(
        project_root=paths.project_root,
        checkpoint=parent_checkpoint,
        eval_csv=paths.eval_csv,
        eval_raw_dir=paths.eval_raw_dir,
        output_dir=parent_evaluation_dir,
        config=config.evaluation,
        physics_config=config.physics,
        scenario_ids=evaluation_scenario_ids,
    )

    candidate_metrics = run_evaluate(
        project_root=paths.project_root,
        checkpoint=candidate_checkpoint,
        eval_csv=paths.eval_csv,
        eval_raw_dir=paths.eval_raw_dir,
        output_dir=candidate_evaluation_dir,
        config=config.evaluation,
        physics_config=config.physics,
        scenario_ids=evaluation_scenario_ids,
    )

    parent_metrics_path = (
            parent_evaluation_dir
            / config.evaluation.output_json_name
    )
    candidate_metrics_path = (
            candidate_evaluation_dir
            / config.evaluation.output_json_name
    )

    require_metrics_pf_alg(
        parent_metrics,
        expected_pf_alg=config.evaluation.pf_alg,
        source=str(parent_metrics_path),
    )
    require_metrics_physics_config(
        parent_metrics,
        expected_physics_config=config.physics,
        source=str(parent_metrics_path),
    )

    require_metrics_pf_alg(
        candidate_metrics,
        expected_pf_alg=config.evaluation.pf_alg,
        source=str(candidate_metrics_path),
    )
    require_metrics_physics_config(
        candidate_metrics,
        expected_physics_config=config.physics,
        source=str(candidate_metrics_path),
    )

    _require_matching_evaluation_inputs(
        parent_metrics,
        candidate_metrics,
    )

    parent_ungated_metrics = _metrics_for_policy_mode(
        parent_metrics,
        PolicyMode.UNGATED,
        source=str(parent_metrics_path),
    )
    candidate_ungated_metrics = _metrics_for_policy_mode(
        candidate_metrics,
        PolicyMode.UNGATED,
        source=str(candidate_metrics_path),
    )

    parent_results_path = (
            parent_evaluation_dir
            / config.evaluation.output_csv_name
    )
    candidate_results_path = (
            candidate_evaluation_dir
            / config.evaluation.output_csv_name
    )

    comparison = compare_evaluation_results(
        parent_csv=parent_results_path,
        candidate_csv=candidate_results_path,
        policy_mode=PolicyMode.UNGATED.value,
        confidence_level=(
            config.acceptance.confidence_level
        ),
        bootstrap_samples=(
            config.acceptance.bootstrap_samples
        ),
        seed=iteration_seed,
    )

    comparison_path = (
            evaluation_dir
            / "comparison.json"
    )
    save_json(
        comparison,
        comparison_path,
    )

    aggregate_gates_passed = accept_candidate(
        new_metrics=candidate_ungated_metrics,
        best_metrics=parent_ungated_metrics,
        config=config.acceptance,
    )

    confidence_gates_passed = (
        passes_confidence_gates(
            comparison=comparison,
            config=config.acceptance,
        )
    )

    accepted = (
            aggregate_gates_passed
            and confidence_gates_passed
    )

    physically_secure_comparison = (
        comparison["metrics"]["physically_secure"]
    )

    status = "ACCEPTED" if accepted else "REJECTED"
    metadata_path = iter_dir / "metadata.json"

    _save_iteration_metadata(
        iteration=iteration,
        path=metadata_path,
        accepted=accepted,
        parent_checkpoint=parent_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        train_batch_csv=train_batch_path,
        train_examples_csv=train_examples_path,
        validation_examples_csv=validation_examples_path,
        split_metadata_path=split_metadata_path,
        raw_examples_csv=raw_examples_csv,
        parent_metrics=parent_metrics,
        metrics=candidate_metrics,
        config=dict(request.raw_config),
        extra={
            "status": status,
            "primary_policy_mode": PolicyMode.UNGATED.value,
            "metric_name": metric_name,
            "candidate_metric": candidate_ungated_metrics.get(metric_name),
            "best_metric_before": parent_ungated_metrics.get(metric_name),
            "n_evaluation_scenarios": len(evaluation_scenario_ids),
            "parent_evaluation_dir": str(parent_evaluation_dir),
            "candidate_evaluation_dir": str(candidate_evaluation_dir),
            "n_sampled_scenarios": len(scenario_ids),
            "n_raw_examples": raw_examples_count,
            "n_new_examples_loaded": len(new_examples),
            "train_batch_metadata": train_batch_metadata,
            "scenario_sampling_seed": int(
                self_play_seeds.scenario_sampling
            ),
            "mcts_seed": int(
                self_play_seeds.mcts
            ),
            "action_sampling_seed": int(
                self_play_seeds.action_sampling
            ),
            "self_play_exploration": (
                exploration_metrics
            ),
            "training_seed": int(iteration_seed),
            "validation_fraction": float(config.training.validation_fraction),
            "train_validation_split": split_metadata,
            "selected_scenario_ids_path": str(selected_ids_path),
            "pool_metadata_path": str(paths.pool_metadata),
            "pool_metadata_sha256_before_update": (
                sha256_file(paths.pool_metadata)
                if paths.pool_metadata.exists()
                else None
            ),
            "paired_comparison_path": str(
                comparison_path
            ),
            "paired_comparison_sha256": sha256_file(
                comparison_path
            ),
            "paired_comparison_scenarios": int(
                comparison["scenario_count"]
            ),
            "paired_comparison_policy_mode": str(
                comparison["policy_mode"]
            ),
            "paired_comparison_confidence_level": float(
                comparison["confidence_level"]
            ),
            "paired_comparison_bootstrap_samples": int(
                comparison["bootstrap_samples"]
            ),
            "aggregate_gates_passed": bool(
                aggregate_gates_passed
            ),
            "confidence_gates_passed": bool(
                confidence_gates_passed
            ),
            "paired_physically_secure_rate_difference": float(
                physically_secure_comparison[
                    "rate_difference"
                ]
            ),
            "paired_physically_secure_ci_lower": float(
                physically_secure_comparison[
                    "ci_lower"
                ]
            ),
            "paired_physically_secure_ci_upper": float(
                physically_secure_comparison[
                    "ci_upper"
                ]
            ),
        },
    )

    if accepted:
        best_state = promote_candidate(
            candidate_checkpoint=Path(candidate_checkpoint),
            candidate_metrics=candidate_metrics,
            paths=paths,
        )
        best_checkpoint = best_state.checkpoint
        best_metrics = dict(best_state.metrics)
    else:
        best_checkpoint = parent_checkpoint
        best_metrics = dict(parent_metrics)
        save_json(best_metrics, paths.best_metrics)

    pool_metadata = update_and_save_pool_metadata(
        pool_metadata=request.pool_metadata,
        episode_results=raw_examples_df,
        current_iter=iteration,
        path=paths.pool_metadata,
        selected_scenario_ids=scenario_ids,
        stale_after_iterations=(
            config.pool.curriculum.stale_after_iterations
        ),
    )

    candidate_metric = candidate_ungated_metrics.get(metric_name)
    best_ungated_metrics = _metrics_for_policy_mode(
        best_metrics,
        PolicyMode.UNGATED,
        source="best metrics after iteration",
    )
    best_metric_after = best_ungated_metrics.get(metric_name)

    row: dict[str, object] = {
        "iteration": int(iteration),
        "accepted": bool(accepted),
        "status": status,
        "primary_policy_mode": PolicyMode.UNGATED.value,
        "candidate_metric": candidate_metric,
        "best_metric_after": best_metric_after,
        "n_sampled_scenarios": int(len(scenario_ids)),
        "n_raw_examples": int(raw_examples_count),
        "n_train_examples": int(train_batch_metadata["n_examples"]),
        "n_fit_examples": int(split_metadata["train_examples"]),
        "n_validation_examples": int(split_metadata["validation_examples"]),
        "n_fit_scenarios": int(split_metadata["train_scenarios"]),
        "n_validation_scenarios": int(split_metadata["validation_scenarios"]),
        "training_seed": int(iteration_seed),
        "checkpoint_selection_metric": "validation_loss",
        "n_fresh": int(train_batch_metadata["n_fresh"]),
        "n_old": int(train_batch_metadata["n_old"]),
        "candidate_checkpoint": str(candidate_checkpoint),
        "best_checkpoint_after": str(best_checkpoint),
        "aggregate_gates_passed": bool(
            aggregate_gates_passed
        ),
        "confidence_gates_passed": bool(
            confidence_gates_passed
        ),
        "paired_scenario_count": int(
            comparison["scenario_count"]
        ),
        "paired_confidence_level": float(
            comparison["confidence_level"]
        ),
        "paired_bootstrap_samples": int(
            comparison["bootstrap_samples"]
        ),
        "physically_secure_rate_difference": float(
            physically_secure_comparison[
                "rate_difference"
            ]
        ),
        "physically_secure_ci_lower": float(
            physically_secure_comparison[
                "ci_lower"
            ]
        ),
        "physically_secure_ci_upper": float(
            physically_secure_comparison[
                "ci_upper"
            ]
        ),
    }

    for key, value in exploration_metrics.items():
        row[f"self_play_{key}"] = value

    for key, value in candidate_metrics.items():
        row[f"candidate_{key}"] = value

    for key, value in best_metrics.items():
        row[f"best_{key}"] = value


    return IterationResult(
        iteration=iteration,
        accepted=accepted,
        status=status,
        selected_scenario_ids=tuple(int(value) for value in scenario_ids),
        raw_examples_csv=raw_examples_csv,
        train_batch_csv=train_batch_path,
        train_examples_csv=train_examples_path,
        validation_examples_csv=validation_examples_path,
        split_metadata_path=split_metadata_path,
        candidate_checkpoint=candidate_checkpoint,
        metadata_path=metadata_path,
        parent_metrics=dict(parent_metrics),
        candidate_metrics=dict(candidate_metrics),
        best_checkpoint=best_checkpoint,
        best_metrics=best_metrics,
        pool_metadata=pool_metadata,
        learning_curve_row=row,
    )

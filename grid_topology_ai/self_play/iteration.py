from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.acceptance import (
    accept_candidate,
    require_metrics_pf_alg,
)
from grid_topology_ai.self_play.artifacts import save_json, sha256_file
from grid_topology_ai.self_play.checkpoint_state import promote_candidate
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.pool_sampling import sample_from_pool
from grid_topology_ai.self_play.pool_state import update_and_save_pool_metadata
from grid_topology_ai.self_play.replay import RollingReplayBuffer
from grid_topology_ai.self_play.stages import (
    run_evaluate,
    run_generate,
    run_train,
    split_examples_by_scenario,
)


@dataclass(frozen=True, slots=True)
class IterationRequest:
    iteration: int
    config: SelfPlayConfig
    raw_config: Mapping[str, object]
    paths: SelfPlayPaths

    parent_checkpoint: Path
    parent_metrics: Mapping[str, object]

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

    candidate_metrics: dict[str, object]

    best_checkpoint: Path
    best_metrics: dict[str, object]

    pool_metadata: dict[str, Any]
    learning_curve_row: dict[str, object]


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
    parent_metrics = dict(request.parent_metrics)

    iteration_seed = int(config.seed) + iteration

    scenario_ids = sample_from_pool(
        pool_metadata=request.pool_metadata,
        n=config.n_scenarios_per_iteration,
        seed=iteration_seed,
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
        base_seed=config.seed,
        iteration=iteration,
    )

    raw_examples_count = _count_examples_csv(raw_examples_csv)

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

    train_batch_metadata = request.replay_buffer.export_mixed_batch(
        output_path=train_batch_path,
        current_iteration=iteration,
        n_examples=examples_per_iteration,
        fresh_fraction=float(config.replay_buffer.fresh_fraction),
        seed=iteration_seed,
    )

    train_examples_path = iter_dir / "train_examples.csv"
    validation_examples_path = iter_dir / "validation_examples.csv"
    split_metadata_path = iter_dir / "train_validation_split.json"
    split_metadata = split_examples_by_scenario(
        examples_csv=train_batch_path,
        train_output_csv=train_examples_path,
        validation_output_csv=validation_examples_path,
        metadata_output_json=split_metadata_path,
        validation_fraction=config.training.validation_fraction,
        min_validation_scenarios=config.training.min_validation_scenarios,
        seed=iteration_seed,
    )

    candidate_checkpoint = run_train(
        project_root=paths.project_root,
        examples_csv=train_examples_path,
        validation_examples_csv=validation_examples_path,
        init_checkpoint=parent_checkpoint,
        output_dir=iter_dir,
        config=config.training,
        iteration=iteration,
        seed=iteration_seed,
    )

    metrics = run_evaluate(
        project_root=paths.project_root,
        checkpoint=candidate_checkpoint,
        eval_csv=paths.eval_csv,
        eval_raw_dir=paths.eval_raw_dir,
        output_dir=iter_dir,
        config=config.evaluation,
    )
    require_metrics_pf_alg(
        metrics,
        expected_pf_alg=config.evaluation.pf_alg,
        source=str(iter_dir / config.evaluation.output_json_name),
    )
    require_metrics_pf_alg(
        parent_metrics,
        expected_pf_alg=config.evaluation.pf_alg,
        source="parent/best metrics",
    )

    accepted = accept_candidate(
        new_metrics=metrics,
        best_metrics=parent_metrics,
        config=config.acceptance,
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
        metrics=metrics,
        config=dict(request.raw_config),
        extra={
            "status": status,
            "metric_name": metric_name,
            "candidate_metric": metrics.get(metric_name),
            "best_metric_before": parent_metrics.get(metric_name),
            "n_sampled_scenarios": len(scenario_ids),
            "n_raw_examples": raw_examples_count,
            "n_new_examples_loaded": len(new_examples),
            "train_batch_metadata": train_batch_metadata,
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
        },
    )

    if accepted:
        best_state = promote_candidate(
            candidate_checkpoint=Path(candidate_checkpoint),
            candidate_metrics=metrics,
            paths=paths,
        )
        best_checkpoint = best_state.checkpoint
        best_metrics = dict(best_state.metrics)
    else:
        best_checkpoint = parent_checkpoint
        best_metrics = parent_metrics

    raw_examples_df = pd.read_csv(raw_examples_csv)

    pool_metadata = update_and_save_pool_metadata(
        pool_metadata=request.pool_metadata,
        episode_results=raw_examples_df,
        current_iter=iteration,
        path=paths.pool_metadata,
        selected_scenario_ids=scenario_ids,
    )

    candidate_metric = metrics.get(metric_name)
    best_metric_after = best_metrics.get(metric_name)

    row: dict[str, object] = {
        "iteration": int(iteration),
        "accepted": bool(accepted),
        "status": status,
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
    }

    for key, value in metrics.items():
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
        candidate_metrics=metrics,
        best_checkpoint=best_checkpoint,
        best_metrics=best_metrics,
        pool_metadata=pool_metadata,
        learning_curve_row=row,
    )

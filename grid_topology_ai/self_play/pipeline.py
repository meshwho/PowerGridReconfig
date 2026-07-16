from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.artifacts import save_yaml
from grid_topology_ai.self_play.acceptance import require_metrics_pf_alg
from grid_topology_ai.self_play.checkpoint_state import initialize_best_state
from grid_topology_ai.self_play.completion import write_iteration_completion_marker
from grid_topology_ai.self_play.iteration import IterationRequest, run_self_play_iteration
from grid_topology_ai.self_play.learning_curve import (
    load_learning_curve,
    save_learning_curve,
    upsert_iteration_row,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.pool_state import initialize_pool_metadata
from grid_topology_ai.self_play.preflight import validate_resume_artifacts
from grid_topology_ai.self_play.replay import RollingReplayBuffer
from grid_topology_ai.self_play.run_state import resolve_run_state


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    config: SelfPlayConfig
    raw_config: Mapping[str, object]
    paths: SelfPlayPaths
    resume: bool = False


@dataclass(frozen=True, slots=True)
class PipelineResult:
    best_checkpoint: Path
    best_metrics: dict[str, object]

    start_iteration: int
    completed_iterations_before_run: tuple[int, ...]
    executed_iterations: tuple[int, ...]

    learning_curve_path: Path
    pool_metadata: dict[str, object]

    already_complete: bool


def _print_header(title: str) -> None:
    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)


def _format_metric(
    metrics: Mapping[str, object],
    metric_name: str,
) -> str:
    if metric_name not in metrics:
        return "n/a"

    value = metrics[metric_name]

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def run_self_play_pipeline(
    request: PipelineRequest,
) -> PipelineResult:
    config = request.config
    paths = request.paths

    paths.run_dir.mkdir(parents=True, exist_ok=True)

    run_state = resolve_run_state(
        run_dir=paths.run_dir,
        resume=request.resume,
    )
    start_iteration = run_state.start_iteration
    completed_iterations = run_state.completed_iterations

    save_yaml(
        payload=dict(request.raw_config),
        path=paths.resolved_config,
    )

    if request.resume and completed_iterations:
        validate_resume_artifacts(paths)

    best_state = initialize_best_state(paths=paths)
    best_checkpoint = best_state.checkpoint
    best_metrics = dict(best_state.metrics)
    require_metrics_pf_alg(
        best_metrics,
        expected_pf_alg=config.evaluation.pf_alg,
        source=str(paths.best_metrics),
    )

    pool_metadata = initialize_pool_metadata(
        transitions_csv=paths.pool_transitions_csv,
        path=paths.pool_metadata,
        current_iter=0,
        overwrite=False,
    )

    replay_buffer = RollingReplayBuffer(
        save_dir=paths.replay_dir,
        config=config.replay_buffer,
    )

    learning_curve_path = paths.learning_curve
    learning_curve = load_learning_curve(learning_curve_path)

    metric_name = config.acceptance.metric

    _print_header(f"Self-play loop: {config.run_name}")
    print(f"Project root:             {paths.project_root}")
    print(f"Resolved config copy:     {paths.resolved_config}")
    print(f"Iterations planned:       {config.n_iterations}")
    print(f"Resume mode:              {request.resume}")
    print(f"Completed iterations:     {completed_iterations}")
    print(f"Starting iteration:       {start_iteration}")
    print(f"Scenarios per iteration:  {config.n_scenarios_per_iteration}")
    print(f"Pool transitions:         {paths.pool_transitions_csv}")
    print(f"Pool raw dir:             {paths.pool_raw_dir}")
    print(f"Pool metadata:            {paths.pool_metadata}")
    print(f"Pool size:                {len(pool_metadata['scenarios'])}")
    print(f"Replay buffer size:       {len(replay_buffer)}")
    print(f"Best checkpoint:          {best_checkpoint}")
    print(f"Best metric {metric_name}:       {_format_metric(best_metrics, metric_name)}")

    executed_iterations: list[int] = []

    if start_iteration > config.n_iterations:
        _print_header("Self-play already complete")
        print(f"Completed iterations: {completed_iterations}")
        print(f"Configured total:     {config.n_iterations}")
        print(f"Best checkpoint:      {best_checkpoint}")
        print(
            f"Best {metric_name}:           "
            f"{_format_metric(best_metrics, metric_name)}"
        )
        return PipelineResult(
            best_checkpoint=best_checkpoint,
            best_metrics=best_metrics,
            start_iteration=start_iteration,
            completed_iterations_before_run=completed_iterations,
            executed_iterations=(),
            learning_curve_path=learning_curve_path,
            pool_metadata=pool_metadata,
            already_complete=True,
        )

    for iteration in range(start_iteration, config.n_iterations + 1):
        _print_header(f"Iteration {iteration} / {config.n_iterations}")

        result = run_self_play_iteration(
            IterationRequest(
                iteration=iteration,
                config=config,
                raw_config=request.raw_config,
                paths=paths,
                parent_checkpoint=best_checkpoint,
                parent_metrics=best_metrics,
                pool_metadata=pool_metadata,
                replay_buffer=replay_buffer,
            )
        )

        best_checkpoint = result.best_checkpoint
        best_metrics = dict(result.best_metrics)
        pool_metadata = result.pool_metadata

        learning_curve = upsert_iteration_row(
            rows=learning_curve,
            row=result.learning_curve_row,
        )

        save_learning_curve(
            rows=learning_curve,
            path=learning_curve_path,
        )

        completion_marker_path = paths.iteration_completion_marker(iteration)
        write_iteration_completion_marker(
            path=completion_marker_path,
            iteration=iteration,
            accepted=result.accepted,
            status=result.status,
            metadata_path=result.metadata_path,
            candidate_checkpoint=result.candidate_checkpoint,
            best_checkpoint_after=result.best_checkpoint,
            best_metrics_path=paths.best_metrics,
            pool_metadata_path=paths.pool_metadata,
            replay_manifest_path=paths.replay_manifest,
            replay_iteration_path=paths.replay_iteration_file(iteration),
            learning_curve_path=learning_curve_path,
        )

        executed_iterations.append(iteration)

        print("")
        print(
            f"[iter {iteration:03d}] {result.status} | "
            f"{metric_name}={_format_metric(result.candidate_metrics, metric_name)} | "
            f"best={_format_metric(best_metrics, metric_name)} | "
            f"fresh={result.learning_curve_row['n_fresh']} | "
            f"old={result.learning_curve_row['n_old']}"
        )

        print(f"Candidate checkpoint: {result.candidate_checkpoint}")
        print(f"Best checkpoint:      {best_checkpoint}")
        print(f"Metadata:             {result.metadata_path}")
        print(f"Learning curve:       {learning_curve_path}")
        print(f"Completion marker:    {completion_marker_path}")

    _print_header("Self-play complete")
    print(f"Final best checkpoint: {best_checkpoint}")
    print(f"Final best {metric_name}: {_format_metric(best_metrics, metric_name)}")
    print(f"Learning curve:        {learning_curve_path}")

    return PipelineResult(
        best_checkpoint=best_checkpoint,
        best_metrics=best_metrics,
        start_iteration=start_iteration,
        completed_iterations_before_run=completed_iterations,
        executed_iterations=tuple(executed_iterations),
        learning_curve_path=learning_curve_path,
        pool_metadata=pool_metadata,
        already_complete=False,
    )

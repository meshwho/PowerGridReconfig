from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from grid_topology_ai.self_play.artifacts import save_yaml

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.checkpoint_state import initialize_best_state
from grid_topology_ai.self_play.learning_curve import (
    load_learning_curve,
    save_learning_curve,
    upsert_iteration_row,
)
from grid_topology_ai.self_play.iteration import (
    IterationRequest,
    run_self_play_iteration,
)
from grid_topology_ai.self_play.paths import (
    SelfPlayPaths,
    discover_project_root,
)
from grid_topology_ai.self_play.plan import render_execution_plan
from grid_topology_ai.self_play.preflight import (
    validate_inputs,
    validate_resume_artifacts,
)

from grid_topology_ai.self_play.pool_metadata import initialize_pool_metadata
from grid_topology_ai.self_play.replay_buffer_v2 import (
    ReplayBuffer,
)
from grid_topology_ai.self_play.run_state import resolve_run_state


def print_header(title: str) -> None:
    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return data



def format_metric(metrics: dict[str, Any], metric_name: str) -> str:
    if metric_name not in metrics:
        return "n/a"

    value = metrics[metric_name]

    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def run_loop(
    *,
    config_path: str | Path,
    validate_only: bool = False,
    plan_only: bool = False,
    resume: bool = False,
) -> None:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    project_root = discover_project_root(config_path)

    config = SelfPlayConfig.from_mapping(cfg)
    paths = SelfPlayPaths.from_config(
        config=config,
        project_root=project_root,
    )

    if plan_only:
        rendered_plan = render_execution_plan(
            config=config,
            paths=paths,
            config_path=config_path,
        )
        print(rendered_plan)
        return

    warnings = validate_inputs(
        paths,
        require_bootstrap=not validate_only,
    )

    for warning in warnings:
        print(f"WARNING: {warning}")

    if validate_only:
        print_header("Self-play config validation")
        print("Config is valid.")
        print(f"Project root: {project_root}")
        print(f"Config:       {config_path}")
        return

    run_name = config.run_name
    checkpoint_dir = paths.run_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_iterations = config.n_iterations

    run_state = resolve_run_state(
        run_dir=paths.run_dir,
        resume=resume,
    )
    start_iteration = run_state.start_iteration
    completed_iterations = run_state.completed_iterations

    run_config_copy = paths.resolved_config

    save_yaml(
        payload=cfg,
        path=run_config_copy,
    )

    pool_transitions_csv = paths.pool_transitions_csv
    pool_raw_dir = paths.pool_raw_dir
    pool_metadata_path = paths.pool_metadata
    eval_csv = paths.eval_csv
    eval_raw_dir = paths.eval_raw_dir

    if resume and completed_iterations:
        validate_resume_artifacts(paths)

    best_state = initialize_best_state(paths=paths)
    best_checkpoint = best_state.checkpoint
    best_metrics = dict(best_state.metrics)

    pool_metadata = initialize_pool_metadata(
        transitions_csv=pool_transitions_csv,
        path=pool_metadata_path,
        current_iter=0,
        overwrite=False,
    )

    replay_buffer = ReplayBuffer(
        save_dir=paths.replay_dir,
        config=config.replay_buffer,
    )

    learning_curve_path = paths.learning_curve
    learning_curve = load_learning_curve(learning_curve_path)

    metric_name = config.acceptance.metric

    print_header(f"Self-play loop: {run_name}")
    print(f"Project root:             {project_root}")
    print(f"Config:                   {config_path}")
    print(f"Resolved config copy:     {run_config_copy}")
    print(f"Iterations planned:       {cfg['n_iterations']}")
    print(f"Resume mode:              {resume}")
    print(f"Completed iterations:     {completed_iterations}")
    print(f"Starting iteration:       {start_iteration}")
    print(f"Scenarios per iteration:  {cfg['n_scenarios_per_iteration']}")
    print(f"Pool transitions:         {pool_transitions_csv}")
    print(f"Pool raw dir:             {pool_raw_dir}")
    print(f"Pool metadata:            {pool_metadata_path}")
    print(f"Pool size:                {len(pool_metadata['scenarios'])}")
    print(f"Replay buffer size:       {len(replay_buffer)}")
    print(f"Best checkpoint:          {best_checkpoint}")
    print(f"Best metric {metric_name}:       {format_metric(best_metrics, metric_name)}")

    n_iterations = config.n_iterations

    if start_iteration > n_iterations:
        print_header("Self-play already complete")
        print(f"Completed iterations: {completed_iterations}")
        print(f"Configured total:     {n_iterations}")
        print(f"Best checkpoint:      {best_checkpoint}")
        print(
            f"Best {metric_name}:           "
            f"{format_metric(best_metrics, metric_name)}"
        )
        return

    for iteration in range(start_iteration, n_iterations + 1):
        print_header(f"Iteration {iteration} / {n_iterations}")

        result = run_self_play_iteration(
            IterationRequest(
                iteration=iteration,
                config=config,
                raw_config=cfg,
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

        print("")
        print(
            f"[iter {iteration:03d}] {result.status} | "
            f"{metric_name}={format_metric(result.candidate_metrics, metric_name)} | "
            f"best={format_metric(best_metrics, metric_name)} | "
            f"fresh={result.learning_curve_row['n_fresh']} | "
            f"old={result.learning_curve_row['n_old']}"
        )

        print(f"Candidate checkpoint: {result.candidate_checkpoint}")
        print(f"Best checkpoint:      {best_checkpoint}")
        print(f"Metadata:             {result.metadata_path}")
        print(f"Learning curve:       {learning_curve_path}")

    print_header("Self-play complete")
    print(f"Final best checkpoint: {best_checkpoint}")
    print(f"Final best {metric_name}: {format_metric(best_metrics, metric_name)}")
    print(f"Learning curve:        {learning_curve_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hybrid pool-guided self-play loop."
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to self_play_loop.yaml.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config and required paths, do not run self-play.",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Print the resolved self-play execution plan without running "
            "generation, training, or evaluation."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue after the last completed iteration. "
            "Refuse to continue if incomplete iteration "
            "directories are present."
        ),
    )

    args = parser.parse_args()

    run_loop(
        config_path=args.config,
        validate_only=bool(args.validate_only),
        plan_only=bool(args.plan_only),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()

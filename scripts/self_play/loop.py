from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from grid_topology_ai.self_play.pool_metadata import (
    initialize_pool_metadata,
    sample_from_pool,
    update_and_save_pool_metadata,
)
from grid_topology_ai.self_play.replay_buffer_v2 import (
    ReplayBuffer,
    ReplayBufferConfig,
)
from scripts.self_play.run_iteration import (
    accept_candidate,
    copy_if_accepted,
    discover_project_root,
    load_json,
    run_evaluate,
    run_generate,
    run_train,
    save_iteration_metadata,
    save_json,
    sha256_file,
)


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


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return project_root / path


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")


def require_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")


def validate_config(
    *,
    cfg: dict[str, Any],
    project_root: Path,
    require_runtime_artifacts: bool = True,
) -> None:
    required_top_level = [
        "run_name",
        "seed",
        "n_iterations",
        "n_scenarios_per_iteration",
        "pool",
        "eval_csv",
        "eval_raw_dir",
        "bootstrap_checkpoint",
        "bootstrap_eval_metrics",
        "checkpoint_dir",
        "best_checkpoint_path",
        "best_metrics_path",
        "replay_buffer",
        "generation",
        "training",
        "evaluation",
        "acceptance",
    ]

    missing = [
        key
        for key in required_top_level
        if key not in cfg
    ]

    if missing:
        raise ValueError(f"Missing top-level config keys: {missing}")

    pool_cfg = cfg["pool"]

    for key in ["transitions_csv", "raw_dir", "metadata_path"]:
        if key not in pool_cfg:
            raise ValueError(f"Missing pool.{key} in config.")

    require_file(
        as_project_path(project_root, pool_cfg["transitions_csv"]),
        "Pool transitions CSV",
    )

    require_dir(
        as_project_path(project_root, pool_cfg["raw_dir"]),
        "Pool raw directory",
    )

    require_file(
        as_project_path(project_root, cfg["eval_csv"]),
        "Eval transitions CSV",
    )

    require_dir(
        as_project_path(project_root, cfg["eval_raw_dir"]),
        "Eval raw directory",
    )

    bootstrap_checkpoint = as_project_path(
        project_root,
        cfg["bootstrap_checkpoint"],
    )

    bootstrap_eval_metrics = as_project_path(
        project_root,
        cfg["bootstrap_eval_metrics"],
    )

    if require_runtime_artifacts:
        require_file(
            bootstrap_checkpoint,
            "Bootstrap checkpoint",
        )

        require_file(
            bootstrap_eval_metrics,
            "Bootstrap eval metrics",
        )
    else:
        if not bootstrap_checkpoint.exists():
            print(
                "WARNING: Bootstrap checkpoint is missing. "
                f"This is allowed in --validate-only mode: {bootstrap_checkpoint}"
            )

        if not bootstrap_eval_metrics.exists():
            print(
                "WARNING: Bootstrap eval metrics are missing. "
                f"This is allowed in --validate-only mode: {bootstrap_eval_metrics}"
            )

    replay_cfg = cfg["replay_buffer"]

    if int(replay_cfg.get("max_size", 0)) <= 0:
        raise ValueError("replay_buffer.max_size must be positive.")

    if int(replay_cfg.get("min_size_to_train", 0)) <= 0:
        raise ValueError("replay_buffer.min_size_to_train must be positive.")

    fresh_fraction = float(replay_cfg.get("fresh_fraction", 0.70))

    if fresh_fraction < 0.0 or fresh_fraction > 1.0:
        raise ValueError("replay_buffer.fresh_fraction must be in [0, 1].")

    acceptance_metric = str(cfg["acceptance"].get("metric", ""))

    if not acceptance_metric:
        raise ValueError("acceptance.metric must not be empty.")


def initialize_best_checkpoint(
    *,
    project_root: Path,
    cfg: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """
    Initialize canonical self-play best checkpoint.

    If runs/self_play_v1/checkpoints/best.pt does not exist yet,
    copy bootstrap checkpoint there and copy bootstrap metrics.
    """

    bootstrap_checkpoint = as_project_path(
        project_root,
        cfg["bootstrap_checkpoint"],
    )
    bootstrap_metrics_path = as_project_path(
        project_root,
        cfg["bootstrap_eval_metrics"],
    )

    best_checkpoint_path = as_project_path(
        project_root,
        cfg["best_checkpoint_path"],
    )
    best_metrics_path = as_project_path(
        project_root,
        cfg["best_metrics_path"],
    )

    require_file(bootstrap_checkpoint, "Bootstrap checkpoint")
    require_file(bootstrap_metrics_path, "Bootstrap eval metrics")

    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not best_checkpoint_path.exists():
        print("Initializing self-play best checkpoint from bootstrap.")
        print(f"Bootstrap checkpoint: {bootstrap_checkpoint}")
        print(f"Best checkpoint:      {best_checkpoint_path}")

        shutil.copy2(
            bootstrap_checkpoint,
            best_checkpoint_path,
        )

    if not best_metrics_path.exists():
        print("Initializing self-play best metrics from bootstrap.")
        print(f"Bootstrap metrics: {bootstrap_metrics_path}")
        print(f"Best metrics:      {best_metrics_path}")

        shutil.copy2(
            bootstrap_metrics_path,
            best_metrics_path,
        )

    best_metrics = load_json(best_metrics_path)

    return best_checkpoint_path, best_metrics


def save_config_copy(
    *,
    cfg: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    return output_path


def load_or_initialize_learning_curve(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        return []

    df = pd.read_csv(path)

    if df.empty:
        return []

    return df.to_dict(orient="records")


def save_learning_curve(
    *,
    rows: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    # Stable field ordering: important columns first, then all metric columns.
    preferred = [
        "iteration",
        "accepted",
        "status",
        "candidate_metric",
        "best_metric_after",
        "n_sampled_scenarios",
        "n_raw_examples",
        "n_train_examples",
        "n_fresh",
        "n_old",
        "candidate_checkpoint",
        "best_checkpoint_after",
    ]

    keys = []

    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)

    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return path


def count_examples_csv(path: str | Path) -> int:
    path = Path(path)

    if not path.exists():
        return 0

    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def make_training_config(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    training_cfg = dict(cfg.get("training", {}))

    training_cfg.setdefault(
        "epochs",
        int(cfg.get("epochs_per_iteration", 10)),
    )

    return training_cfg


def update_best_metrics_file(
    *,
    metrics: dict[str, Any],
    path: str | Path,
) -> Path:
    path = Path(path)
    save_json(metrics, path)
    return path


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
) -> None:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    project_root = discover_project_root(config_path)

    validate_config(
        cfg=cfg,
        project_root=project_root,
        require_runtime_artifacts=not validate_only,
    )

    if validate_only:
        print_header("Self-play config validation")
        print("Config is valid.")
        print(f"Project root: {project_root}")
        print(f"Config:       {config_path}")
        return

    run_name = str(cfg["run_name"])
    seed = int(cfg.get("seed", 42))

    checkpoint_dir = as_project_path(
        project_root,
        cfg.get("checkpoint_dir", f"runs/{run_name}"),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_config_copy = checkpoint_dir / "self_play_loop.resolved.yaml"

    save_config_copy(
        cfg=cfg,
        output_path=run_config_copy,
    )

    pool_cfg = cfg["pool"]

    pool_transitions_csv = as_project_path(
        project_root,
        pool_cfg["transitions_csv"],
    )
    pool_raw_dir = as_project_path(
        project_root,
        pool_cfg["raw_dir"],
    )
    pool_metadata_path = as_project_path(
        project_root,
        pool_cfg["metadata_path"],
    )

    eval_csv = as_project_path(
        project_root,
        cfg["eval_csv"],
    )
    eval_raw_dir = as_project_path(
        project_root,
        cfg["eval_raw_dir"],
    )

    best_checkpoint, best_metrics = initialize_best_checkpoint(
        project_root=project_root,
        cfg=cfg,
    )

    best_metrics_path = as_project_path(
        project_root,
        cfg["best_metrics_path"],
    )

    pool_metadata = initialize_pool_metadata(
        transitions_csv=pool_transitions_csv,
        path=pool_metadata_path,
        current_iter=0,
        overwrite=False,
    )

    replay_cfg = ReplayBufferConfig(
        **dict(cfg.get("replay_buffer", {}))
    )

    replay_buffer = ReplayBuffer(
        save_dir=checkpoint_dir / "replay_buffer",
        config=replay_cfg,
    )

    learning_curve_path = checkpoint_dir / "learning_curve.csv"
    learning_curve = load_or_initialize_learning_curve(learning_curve_path)

    metric_name = str(cfg["acceptance"].get("metric", "solve_rate"))

    print_header(f"Self-play loop: {run_name}")
    print(f"Project root:             {project_root}")
    print(f"Config:                   {config_path}")
    print(f"Resolved config copy:     {run_config_copy}")
    print(f"Iterations planned:       {cfg['n_iterations']}")
    print(f"Scenarios per iteration:  {cfg['n_scenarios_per_iteration']}")
    print(f"Pool transitions:         {pool_transitions_csv}")
    print(f"Pool raw dir:             {pool_raw_dir}")
    print(f"Pool metadata:            {pool_metadata_path}")
    print(f"Pool size:                {len(pool_metadata['scenarios'])}")
    print(f"Replay buffer size:       {len(replay_buffer)}")
    print(f"Best checkpoint:          {best_checkpoint}")
    print(f"Best metric {metric_name}:       {format_metric(best_metrics, metric_name)}")

    n_iterations = int(cfg["n_iterations"])
    n_scenarios_per_iteration = int(cfg["n_scenarios_per_iteration"])

    for iteration in range(1, n_iterations + 1):
        print_header(f"Iteration {iteration} / {n_iterations}")

        iter_dir = checkpoint_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        parent_checkpoint = best_checkpoint
        parent_metrics = dict(best_metrics)

        iteration_seed = seed + iteration

        scenario_ids = sample_from_pool(
            pool_metadata=pool_metadata,
            n=n_scenarios_per_iteration,
            seed=iteration_seed,
        )

        selected_ids_path = iter_dir / "selected_scenario_ids.txt"
        selected_ids_path.write_text(
            "\n".join(str(value) for value in scenario_ids) + "\n",
            encoding="utf-8",
        )

        print(f"Sampled scenarios: {len(scenario_ids)}")
        print(f"Selected IDs:      {selected_ids_path}")

        generation_cfg = dict(cfg.get("generation", {}))
        generation_cfg.setdefault("seed", seed)

        raw_examples_csv = run_generate(
            project_root=project_root,
            raw_dir=pool_raw_dir,
            transitions_csv=pool_transitions_csv,
            scenario_ids=scenario_ids,
            checkpoint=parent_checkpoint,
            output_dir=iter_dir / "raw",
            config=generation_cfg,
            iteration=iteration,
        )

        raw_examples_count = count_examples_csv(raw_examples_csv)

        new_examples = replay_buffer.add_and_save_from_csv(
            examples_csv=raw_examples_csv,
            iteration=iteration,
        )

        examples_per_iteration = int(
            cfg.get("training", {}).get(
                "examples_per_iteration",
                len(replay_buffer),
            )
        )

        train_batch_path = iter_dir / "train_batch.csv"

        train_batch_metadata = replay_buffer.export_mixed_batch(
            output_path=train_batch_path,
            current_iteration=iteration,
            n_examples=examples_per_iteration,
            fresh_fraction=float(replay_cfg.fresh_fraction),
            seed=iteration_seed,
        )

        training_cfg = make_training_config(cfg)

        candidate_checkpoint = run_train(
            project_root=project_root,
            examples_csv=train_batch_path,
            init_checkpoint=parent_checkpoint,
            output_dir=iter_dir,
            config=training_cfg,
            iteration=iteration,
        )

        metrics = run_evaluate(
            project_root=project_root,
            checkpoint=candidate_checkpoint,
            eval_csv=eval_csv,
            eval_raw_dir=eval_raw_dir,
            output_dir=iter_dir,
            config=cfg["evaluation"],
        )

        accepted = accept_candidate(
            new_metrics=metrics,
            best_metrics=parent_metrics,
            policy=cfg["acceptance"],
        )

        status = "ACCEPTED" if accepted else "REJECTED"

        # Save metadata before overwriting canonical best checkpoint.
        save_iteration_metadata(
            iteration=iteration,
            path=iter_dir / "metadata.json",
            accepted=accepted,
            parent_checkpoint=parent_checkpoint,
            candidate_checkpoint=candidate_checkpoint,
            train_batch_csv=train_batch_path,
            raw_examples_csv=raw_examples_csv,
            metrics=metrics,
            config=cfg,
            extra={
                "status": status,
                "metric_name": metric_name,
                "candidate_metric": metrics.get(metric_name),
                "best_metric_before": parent_metrics.get(metric_name),
                "n_sampled_scenarios": len(scenario_ids),
                "n_raw_examples": raw_examples_count,
                "n_new_examples_loaded": len(new_examples),
                "train_batch_metadata": train_batch_metadata,
                "selected_scenario_ids_path": str(selected_ids_path),
                "pool_metadata_path": str(pool_metadata_path),
                "pool_metadata_sha256_before_update": (
                    sha256_file(pool_metadata_path)
                    if pool_metadata_path.exists()
                    else None
                ),
            },
        )

        if accepted:
            best_checkpoint = copy_if_accepted(
                candidate_checkpoint=candidate_checkpoint,
                best_checkpoint_path=best_checkpoint,
            )

            best_metrics = dict(metrics)

            update_best_metrics_file(
                metrics=best_metrics,
                path=best_metrics_path,
            )

        else:
            best_checkpoint = parent_checkpoint
            best_metrics = parent_metrics

        # Update pool metadata after generation, regardless of candidate acceptance.
        raw_examples_df = pd.read_csv(raw_examples_csv)

        pool_metadata = update_and_save_pool_metadata(
            pool_metadata=pool_metadata,
            episode_results=raw_examples_df,
            current_iter=iteration,
            path=pool_metadata_path,
        )

        candidate_metric = metrics.get(metric_name)
        best_metric_after = best_metrics.get(metric_name)

        row: dict[str, Any] = {
            "iteration": int(iteration),
            "accepted": bool(accepted),
            "status": status,
            "candidate_metric": candidate_metric,
            "best_metric_after": best_metric_after,
            "n_sampled_scenarios": int(len(scenario_ids)),
            "n_raw_examples": int(raw_examples_count),
            "n_train_examples": int(train_batch_metadata["n_examples"]),
            "n_fresh": int(train_batch_metadata["n_fresh"]),
            "n_old": int(train_batch_metadata["n_old"]),
            "candidate_checkpoint": str(candidate_checkpoint),
            "best_checkpoint_after": str(best_checkpoint),
        }

        for key, value in metrics.items():
            row[f"candidate_{key}"] = value

        for key, value in best_metrics.items():
            row[f"best_{key}"] = value

        # Avoid duplicate rows if the same iteration is rerun manually.
        learning_curve = [
            item
            for item in learning_curve
            if int(item.get("iteration", -1)) != int(iteration)
        ]

        learning_curve.append(row)

        save_learning_curve(
            rows=learning_curve,
            path=learning_curve_path,
        )

        print("")
        print(
            f"[iter {iteration:03d}] {status} | "
            f"{metric_name}={format_metric(metrics, metric_name)} | "
            f"best={format_metric(best_metrics, metric_name)} | "
            f"fresh={train_batch_metadata['n_fresh']} | "
            f"old={train_batch_metadata['n_old']}"
        )

        print(f"Candidate checkpoint: {candidate_checkpoint}")
        print(f"Best checkpoint:      {best_checkpoint}")
        print(f"Metadata:             {iter_dir / 'metadata.json'}")
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

    args = parser.parse_args()

    run_loop(
        config_path=args.config,
        validate_only=bool(args.validate_only),
    )


if __name__ == "__main__":
    main()
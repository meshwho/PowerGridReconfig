from __future__ import annotations

import json
from pathlib import Path

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.paths import SelfPlayPaths
from scripts.self_play.loop import (
    initialize_best_checkpoint,
    print_execution_plan,
)


def _config(tmp_path: Path) -> SelfPlayConfig:
    return SelfPlayConfig.from_mapping(
        {
            "run_name": "test_self_play",
            "seed": 7,
            "n_iterations": 2,
            "n_scenarios_per_iteration": 3,
            "epochs_per_iteration": 1,
            "pool": {
                "transitions_csv": "inputs/pool.csv",
                "raw_dir": "inputs/pool_raw",
                "metadata_path": "runs/test_self_play/inputs/pool_metadata.json",
            },
            "eval_csv": "inputs/eval.csv",
            "eval_raw_dir": "inputs/eval_raw",
            "bootstrap_checkpoint": "bootstrap/bootstrap.pt",
            "bootstrap_eval_metrics": "bootstrap/metrics.json",
            "checkpoint_dir": "runs/test_self_play",
            "best_checkpoint_path": "runs/test_self_play/checkpoints/best.pt",
            "best_metrics_path": "runs/test_self_play/checkpoints/best_metrics.json",
            "replay_buffer": {
                "max_size": 20,
                "min_size_to_train": 1,
                "fresh_fraction": 0.5,
                "random_seed": 7,
            },
            "generation": {
                "simulations": 2,
                "depth": 1,
                "max_steps": 4,
                "top_k": 2,
            },
            "training": {
                "examples_per_iteration": 5,
                "batch_size": 2,
                "learning_rate": 0.001,
                "device": "cpu",
            },
            "evaluation": {
                "simulations": 2,
                "depth": 1,
                "max_steps": 2,
                "top_k": 2,
                "device": "cpu",
            },
            "acceptance": {
                "metric": "solve_rate",
                "min_improvement": 0.0,
                "max_simple_solve_rate_drop": 0.05,
            },
        }
    )


def _paths(tmp_path: Path) -> SelfPlayPaths:
    return SelfPlayPaths.from_config(
        config=_config(tmp_path),
        project_root=tmp_path,
    )


def test_initialize_best_checkpoint_uses_resolved_paths(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.bootstrap_checkpoint.parent.mkdir(parents=True)
    paths.bootstrap_checkpoint.write_bytes(b"checkpoint")
    paths.bootstrap_metrics.write_text(
        json.dumps({"solve_rate": 0.75}),
        encoding="utf-8",
    )

    best_checkpoint, best_metrics = initialize_best_checkpoint(
        paths=paths,
    )

    assert best_checkpoint == paths.best_checkpoint
    assert paths.best_checkpoint.read_bytes() == b"checkpoint"
    assert paths.best_metrics.is_file()
    assert best_metrics["solve_rate"] == 0.75


def test_execution_plan_uses_typed_paths(
    tmp_path: Path,
    capsys,
) -> None:
    config = _config(tmp_path)
    paths = SelfPlayPaths.from_config(
        config=config,
        project_root=tmp_path,
    )
    paths.pool_transitions_csv.parent.mkdir(parents=True)
    paths.pool_transitions_csv.write_text(
        "scenario_id\n1\n2\n2\n",
        encoding="utf-8",
    )
    paths.eval_csv.write_text(
        "scenario_id\n3\n",
        encoding="utf-8",
    )
    paths.pool_raw_dir.mkdir(parents=True)
    paths.eval_raw_dir.mkdir(parents=True)

    print_execution_plan(
        config=config,
        paths=paths,
        config_path=tmp_path / "self_play.yaml",
    )

    output = capsys.readouterr().out
    assert str(paths.run_dir) in output
    assert str(paths.iteration_dir(1)) in output
    assert "No generation, training, evaluation" in output


def test_loop_module_has_no_removed_path_helpers() -> None:
    assert callable(initialize_best_checkpoint)
    assert callable(print_execution_plan)

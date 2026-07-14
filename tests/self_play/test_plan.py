from __future__ import annotations

from pathlib import Path

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.plan import render_execution_plan


def _config() -> SelfPlayConfig:
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
        config=_config(),
        project_root=tmp_path,
    )


def _write_plan_inputs(paths: SelfPlayPaths) -> None:
    paths.pool_transitions_csv.parent.mkdir(parents=True)
    paths.pool_transitions_csv.write_text(
        "scenario_id\n1\n2\n2\n",
        encoding="utf-8",
    )
    paths.eval_csv.write_text(
        "scenario_id\n3\n4\n",
        encoding="utf-8",
    )
    paths.pool_raw_dir.mkdir(parents=True)
    paths.eval_raw_dir.mkdir(parents=True)


def test_render_execution_plan_contains_resolved_values(
    tmp_path: Path,
) -> None:
    config = _config()
    paths = SelfPlayPaths.from_config(
        config=config,
        project_root=tmp_path,
    )
    _write_plan_inputs(paths)

    output = render_execution_plan(
        config=config,
        paths=paths,
        config_path=tmp_path / "self_play.yaml",
    )

    assert config.run_name in output
    assert str(paths.run_dir) in output
    assert str(paths.iteration_dir(1)) in output
    assert str(paths.learning_curve) in output
    assert "unique scenarios:         2" in output
    assert "unique eval scenarios:    2" in output
    assert "No generation, training, evaluation, or file creation was performed." in output


def test_render_execution_plan_does_not_create_artifacts(
    tmp_path: Path,
) -> None:
    config = _config()
    paths = SelfPlayPaths.from_config(
        config=config,
        project_root=tmp_path,
    )

    assert not paths.run_dir.exists()

    render_execution_plan(
        config=config,
        paths=paths,
        config_path=tmp_path / "self_play.yaml",
    )

    assert not paths.run_dir.exists()
    assert not paths.iteration_dir(1).exists()
    assert not paths.learning_curve.exists()
    assert not paths.best_checkpoint.exists()
    assert not paths.bootstrap_checkpoint.exists()
    assert not paths.bootstrap_metrics.exists()


def test_missing_optional_paths_are_reported(
    tmp_path: Path,
) -> None:
    config = _config()
    paths = SelfPlayPaths.from_config(
        config=config,
        project_root=tmp_path,
    )
    _write_plan_inputs(paths)

    output = render_execution_plan(
        config=config,
        paths=paths,
        config_path=tmp_path / "self_play.yaml",
    )

    assert "MISSING" in output

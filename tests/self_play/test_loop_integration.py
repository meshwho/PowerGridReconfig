from __future__ import annotations

from pathlib import Path

import pytest

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.paths import SelfPlayPaths
from scripts.self_play import loop as loop_module


def _config_mapping() -> dict[str, object]:
    return {
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


def test_plan_only_uses_imported_renderer(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cfg = _config_mapping()
    paths = SelfPlayPaths.from_config(
        config=SelfPlayConfig.from_mapping(cfg),
        project_root=tmp_path,
    )
    paths.pool_transitions_csv.parent.mkdir(parents=True)
    paths.pool_transitions_csv.write_text(
        "scenario_id\n1\n",
        encoding="utf-8",
    )
    paths.eval_csv.write_text(
        "scenario_id\n2\n",
        encoding="utf-8",
    )
    paths.pool_raw_dir.mkdir(parents=True)
    paths.eval_raw_dir.mkdir(parents=True)

    calls: list[tuple[SelfPlayConfig, SelfPlayPaths, Path]] = []

    def fake_render_execution_plan(
        *,
        config: SelfPlayConfig,
        paths: SelfPlayPaths,
        config_path: Path,
    ) -> str:
        calls.append((config, paths, config_path))
        return "rendered execution plan"

    monkeypatch.setattr(
        loop_module,
        "discover_project_root",
        lambda config_path: tmp_path,
    )
    monkeypatch.setattr(
        loop_module,
        "load_yaml",
        lambda config_path: cfg,
    )
    monkeypatch.setattr(
        loop_module,
        "render_execution_plan",
        fake_render_execution_plan,
    )

    loop_module.run_loop(
        config_path=tmp_path / "self_play.yaml",
        plan_only=True,
    )

    output = capsys.readouterr().out
    assert "rendered execution plan" in output
    assert len(calls) == 1
    assert calls[0][0].run_name == cfg["run_name"]
    assert calls[0][1].run_dir == paths.run_dir
    assert calls[0][2] == tmp_path / "self_play.yaml"


def test_loop_delegates_to_pipeline_package_api() -> None:
    source = Path(loop_module.__file__).read_text(encoding="utf-8")

    assert "run_self_play_pipeline" in source
    assert "run_generate(" not in source
    assert "run_train(" not in source
    assert "run_evaluate(" not in source


def test_normal_loop_delegates_to_pipeline(tmp_path: Path, monkeypatch) -> None:
    cfg = _config_mapping()
    captured = []
    monkeypatch.setattr(loop_module, "load_yaml", lambda config_path: cfg)
    monkeypatch.setattr(loop_module, "discover_project_root", lambda config_path: tmp_path)
    monkeypatch.setattr(loop_module, "validate_inputs", lambda paths, require_bootstrap: [])
    monkeypatch.setattr(loop_module, "run_self_play_pipeline", lambda request: captured.append(request))

    loop_module.run_loop(config_path=tmp_path / "self_play.yaml", resume=True)

    assert len(captured) == 1
    request = captured[0]
    assert request.config.run_name == cfg["run_name"]
    assert request.raw_config is cfg
    assert request.paths.run_dir == SelfPlayPaths.from_config(config=SelfPlayConfig.from_mapping(cfg), project_root=tmp_path).run_dir
    assert request.resume is True


def test_validate_only_does_not_run_pipeline(tmp_path: Path, monkeypatch) -> None:
    cfg = _config_mapping()
    monkeypatch.setattr(loop_module, "load_yaml", lambda config_path: cfg)
    monkeypatch.setattr(loop_module, "discover_project_root", lambda config_path: tmp_path)
    monkeypatch.setattr(loop_module, "validate_inputs", lambda paths, require_bootstrap: [])
    monkeypatch.setattr(loop_module, "run_self_play_pipeline", lambda request: pytest.fail("pipeline should not run"))

    loop_module.run_loop(config_path=tmp_path / "self_play.yaml", validate_only=True)


def test_plan_only_does_not_run_pipeline_or_preflight(tmp_path: Path, monkeypatch) -> None:
    cfg = _config_mapping()
    monkeypatch.setattr(loop_module, "load_yaml", lambda config_path: cfg)
    monkeypatch.setattr(loop_module, "discover_project_root", lambda config_path: tmp_path)
    monkeypatch.setattr(loop_module, "validate_inputs", lambda paths, require_bootstrap: pytest.fail("preflight should not run"))
    monkeypatch.setattr(loop_module, "run_self_play_pipeline", lambda request: pytest.fail("pipeline should not run"))
    monkeypatch.setattr(loop_module, "render_execution_plan", lambda **kwargs: "plan")

    loop_module.run_loop(config_path=tmp_path / "self_play.yaml", plan_only=True)


def test_loop_source_does_not_contain_iteration_orchestration() -> None:
    source = Path(loop_module.__file__).read_text(encoding="utf-8")

    assert "run_self_play_iteration" not in source
    assert "ReplayBuffer(" not in source
    assert "initialize_best_state" not in source
    assert "initialize_pool_metadata" not in source
    assert "save_learning_curve(" not in source

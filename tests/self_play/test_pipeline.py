from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play import pipeline as pipeline_module
from grid_topology_ai.self_play.iteration import IterationResult
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.pipeline import PipelineRequest, run_self_play_pipeline


def _raw_config(n_iterations: int = 2) -> dict[str, object]:
    return {
        "run_name": "test_self_play",
        "seed": 7,
        "n_iterations": n_iterations,
        "n_scenarios_per_iteration": 3,
        "epochs_per_iteration": 1,
        "pool": {"transitions_csv": "inputs/pool.csv", "raw_dir": "inputs/pool_raw", "metadata_path": "runs/test_self_play/inputs/pool_metadata.json"},
        "eval_csv": "inputs/eval.csv",
        "eval_raw_dir": "inputs/eval_raw",
        "bootstrap_checkpoint": "bootstrap/bootstrap.pt",
        "bootstrap_eval_metrics": "bootstrap/metrics.json",
        "checkpoint_dir": "runs/test_self_play",
        "best_checkpoint_path": "runs/test_self_play/checkpoints/best.pt",
        "best_metrics_path": "runs/test_self_play/checkpoints/best_metrics.json",
        "replay_buffer": {"max_size": 20, "min_size_to_train": 1, "fresh_fraction": 0.5, "random_seed": 7},
        "generation": {"simulations": 2, "depth": 1, "max_steps": 4, "top_k": 2},
        "training": {"examples_per_iteration": 5, "batch_size": 2, "learning_rate": 0.001, "device": "cpu"},
        "evaluation": {"simulations": 2, "depth": 1, "max_steps": 2, "top_k": 2, "device": "cpu"},
        "acceptance": {"metric": "solve_rate", "min_improvement": 0.0, "max_simple_solve_rate_drop": 0.05},
    }


def _request(tmp_path: Path, n_iterations: int = 2, resume: bool = False, raw: dict[str, object] | None = None) -> PipelineRequest:
    cfg = _raw_config(n_iterations) if raw is None else raw
    config = SelfPlayConfig.from_mapping(cfg)
    return PipelineRequest(config=config, raw_config=cfg, paths=SelfPlayPaths.from_config(config=config, project_root=tmp_path), resume=resume)


@dataclass(frozen=True, slots=True)
class _RunState:
    completed_iterations: tuple[int, ...]
    start_iteration: int


@dataclass(frozen=True, slots=True)
class _BestState:
    checkpoint: Path
    metrics: dict[str, object]


class _ReplayBuffer:
    def __init__(self, *, save_dir: Path, config: object) -> None:
        self.save_dir = save_dir
        self.config = config

    def __len__(self) -> int:
        return 0


def _patch_basics(monkeypatch, tmp_path: Path, calls: list[str] | None = None, *, start: int = 1, completed: tuple[int, ...] = ()) -> None:
    def note(name: str) -> None:
        if calls is not None:
            calls.append(name)

    monkeypatch.setattr(pipeline_module, "resolve_run_state", lambda **kwargs: (note("resolve_run_state") or _RunState(completed, start)))
    monkeypatch.setattr(pipeline_module, "save_yaml", lambda **kwargs: note("save_yaml"))
    monkeypatch.setattr(pipeline_module, "validate_resume_artifacts", lambda paths: note("validate_resume_artifacts"))
    monkeypatch.setattr(pipeline_module, "initialize_best_state", lambda **kwargs: (note("initialize_best_state") or _BestState(tmp_path / "best.pt", {"solve_rate": 0.1})))
    monkeypatch.setattr(pipeline_module, "initialize_pool_metadata", lambda **kwargs: (note("initialize_pool_metadata") or {"scenarios": []}))
    monkeypatch.setattr(pipeline_module, "ReplayBuffer", lambda **kwargs: (note("ReplayBuffer") or _ReplayBuffer(**kwargs)))
    monkeypatch.setattr(pipeline_module, "load_learning_curve", lambda path: (note("load_learning_curve") or []))
    monkeypatch.setattr(pipeline_module, "upsert_iteration_row", lambda *, rows, row: (calls.append(f"upsert {row['iteration']}") if calls is not None else None) or [*rows, row])
    monkeypatch.setattr(pipeline_module, "save_learning_curve", lambda *, rows, path: (calls.append(f"save {rows[-1]['iteration']}") if calls is not None and rows else None))


def _iteration_result(iteration: int, best: Path, metric: float, pool: dict[str, object]) -> IterationResult:
    return IterationResult(iteration=iteration, accepted=True, status="ACCEPTED", selected_scenario_ids=(), raw_examples_csv=Path("raw.csv"), train_batch_csv=Path("train.csv"), candidate_checkpoint=Path(f"candidate-{iteration}.pt"), metadata_path=Path("metadata.json"), candidate_metrics={"solve_rate": metric}, best_checkpoint=best, best_metrics={"solve_rate": metric}, pool_metadata=pool, learning_curve_row={"iteration": iteration, "n_fresh": 1, "n_old": 0})


def test_pipeline_request_is_frozen_and_slotted(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(FrozenInstanceError):
        request.resume = True  # type: ignore[misc]
    assert not hasattr(request, "__dict__")


def test_pipeline_initialization_order(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _patch_basics(monkeypatch, tmp_path, calls, start=3, completed=(1, 2))
    run_self_play_pipeline(_request(tmp_path, n_iterations=2))
    assert calls[:7] == ["resolve_run_state", "save_yaml", "initialize_best_state", "initialize_pool_metadata", "ReplayBuffer", "load_learning_curve"]


def test_pipeline_passes_shared_state_between_iterations(tmp_path: Path, monkeypatch) -> None:
    _patch_basics(monkeypatch, tmp_path)
    seen = []
    def fake_run(request):
        seen.append((request.iteration, request.parent_checkpoint, dict(request.parent_metrics), request.pool_metadata, request.replay_buffer))
        return _iteration_result(request.iteration, tmp_path / f"best-{request.iteration}.pt", request.iteration / 10, {"scenarios": [], "iter": request.iteration})
    monkeypatch.setattr(pipeline_module, "run_self_play_iteration", fake_run)
    run_self_play_pipeline(_request(tmp_path, n_iterations=3))
    assert seen[1][1] == tmp_path / "best-1.pt"
    assert seen[2][2] == {"solve_rate": 0.2}
    assert seen[2][3]["iter"] == 2
    assert len({id(item[4]) for item in seen}) == 1


def test_pipeline_uses_resolved_start_iteration(tmp_path: Path, monkeypatch) -> None:
    _patch_basics(monkeypatch, tmp_path, start=3, completed=(1, 2))
    iterations: list[int] = []
    monkeypatch.setattr(pipeline_module, "run_self_play_iteration", lambda request: iterations.append(request.iteration) or _iteration_result(request.iteration, tmp_path / "best.pt", 0.2, {"scenarios": []}))
    run_self_play_pipeline(_request(tmp_path, n_iterations=4))
    assert iterations == [3, 4]


def test_pipeline_validates_resume_artifacts(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _patch_basics(monkeypatch, tmp_path, calls, start=2, completed=(1,))
    monkeypatch.setattr(pipeline_module, "run_self_play_iteration", lambda request: _iteration_result(request.iteration, tmp_path / "best.pt", 0.2, {"scenarios": []}))
    run_self_play_pipeline(_request(tmp_path, n_iterations=1, resume=True))
    assert "validate_resume_artifacts" in calls


def test_pipeline_skips_resume_validation_without_completed_iterations(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _patch_basics(monkeypatch, tmp_path, calls, start=1, completed=())
    monkeypatch.setattr(pipeline_module, "run_self_play_iteration", lambda request: _iteration_result(request.iteration, tmp_path / "best.pt", 0.2, {"scenarios": []}))
    run_self_play_pipeline(_request(tmp_path, n_iterations=1, resume=True))
    assert "validate_resume_artifacts" not in calls


def test_pipeline_saves_learning_curve_after_each_iteration(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    _patch_basics(monkeypatch, tmp_path, calls)
    def fake_run(request):
        calls.append(f"iteration {request.iteration}")
        return _iteration_result(request.iteration, tmp_path / "best.pt", 0.2, {"scenarios": []})
    monkeypatch.setattr(pipeline_module, "run_self_play_iteration", fake_run)
    run_self_play_pipeline(_request(tmp_path, n_iterations=2))
    assert [c for c in calls if c.startswith(("iteration", "upsert", "save "))] == ["iteration 1", "upsert 1", "save 1", "iteration 2", "upsert 2", "save 2"]


def test_pipeline_returns_final_state(tmp_path: Path, monkeypatch) -> None:
    _patch_basics(monkeypatch, tmp_path, start=2, completed=(1,))
    monkeypatch.setattr(pipeline_module, "run_self_play_iteration", lambda request: _iteration_result(request.iteration, tmp_path / f"best-{request.iteration}.pt", 0.4, {"scenarios": [], "done": True}))
    result = run_self_play_pipeline(_request(tmp_path, n_iterations=2))
    assert result.best_checkpoint == tmp_path / "best-2.pt"
    assert result.best_metrics == {"solve_rate": 0.4}
    assert result.start_iteration == 2
    assert result.completed_iterations_before_run == (1,)
    assert result.executed_iterations == (2,)
    assert result.learning_curve_path.name == "learning_curve.csv"
    assert result.pool_metadata["done"] is True
    assert result.already_complete is False


def test_pipeline_returns_already_complete(tmp_path: Path, monkeypatch) -> None:
    _patch_basics(monkeypatch, tmp_path, start=3, completed=(1, 2))
    monkeypatch.setattr(pipeline_module, "run_self_play_iteration", lambda request: pytest.fail("iteration should not run"))
    result = run_self_play_pipeline(_request(tmp_path, n_iterations=2))
    assert result.executed_iterations == ()
    assert result.already_complete is True


def test_pipeline_does_not_mutate_raw_config(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_config(n_iterations=1)
    original = dict(raw)
    _patch_basics(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline_module, "save_yaml", lambda *, payload, path: payload.update({"mutated": True}))
    monkeypatch.setattr(pipeline_module, "run_self_play_iteration", lambda request: _iteration_result(request.iteration, tmp_path / "best.pt", 0.2, {"scenarios": []}))
    run_self_play_pipeline(_request(tmp_path, n_iterations=1, raw=raw))
    assert raw == original

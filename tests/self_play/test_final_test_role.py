from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grid_topology_ai.config import EvaluationConfig, SelfPlayConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play import final_test as final_test_module
from grid_topology_ai.self_play import pipeline as pipeline_module
from grid_topology_ai.self_play.final_test import (
    FINAL_TEST_EVALUATION_ROLE,
    FINAL_TEST_REPORT_SCHEMA_VERSION,
    FinalTestEvaluation,
    run_final_test_evaluation,
)
from grid_topology_ai.self_play.iteration import IterationResult
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.pipeline import (
    PipelineRequest,
    run_self_play_pipeline,
)


def _paths(tmp_path: Path) -> SelfPlayPaths:
    run_dir = tmp_path / "run"
    paths = SelfPlayPaths(
        project_root=tmp_path,
        run_dir=run_dir,
        pool_transitions_csv=tmp_path / "pool.csv",
        pool_raw_dir=tmp_path / "pool_raw",
        pool_metadata=run_dir / "pool.json",
        eval_csv=tmp_path / "eval.csv",
        eval_raw_dir=tmp_path / "eval_raw",
        final_test_csv=tmp_path / "final_test.csv",
        final_test_raw_dir=tmp_path / "final_raw",
        bootstrap_checkpoint=tmp_path / "bootstrap.pt",
        bootstrap_metrics=tmp_path / "bootstrap.json",
        best_checkpoint=run_dir / "best.pt",
        best_metrics=run_dir / "best_metrics.json",
    )
    paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.best_checkpoint.write_bytes(b"selected-best-checkpoint")
    paths.best_metrics.write_text(
        json.dumps({"selection_metric": 0.4}),
        encoding="utf-8",
    )
    paths.final_test_csv.write_text(
        "scenario_id\n7\n8\n",
        encoding="utf-8",
    )
    paths.final_test_raw_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _patch_evaluation_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        final_test_module,
        "require_metrics_pf_alg",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        final_test_module,
        "require_metrics_physics_config",
        lambda *args, **kwargs: None,
    )


def _write_fake_evaluation(
    *,
    output_dir: Path,
    config: EvaluationConfig,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / config.output_json_name).write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    (output_dir / config.output_csv_name).write_text(
        "scenario_id,solved\n7,1\n8,0\n",
        encoding="utf-8",
    )


def test_final_test_is_reporting_only_and_preserves_best_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    config = EvaluationConfig()
    _patch_evaluation_contracts(monkeypatch)
    captured: dict[str, object] = {}

    def fake_run_evaluate(**kwargs):
        captured.update(kwargs)
        metrics = {"physically_secure_rate_requested": 0.5}
        _write_fake_evaluation(
            output_dir=Path(kwargs["output_dir"]),
            config=kwargs["config"],
            metrics=metrics,
        )
        return metrics

    monkeypatch.setattr(
        final_test_module,
        "run_evaluate",
        fake_run_evaluate,
    )
    checkpoint_before = paths.best_checkpoint.read_bytes()
    metrics_before = paths.best_metrics.read_bytes()

    result = run_final_test_evaluation(
        paths=paths,
        checkpoint=paths.best_checkpoint,
        config=config,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert captured["checkpoint"] == paths.best_checkpoint
    assert captured["eval_csv"] == paths.final_test_csv
    assert captured["eval_raw_dir"] == paths.final_test_raw_dir
    assert captured["scenario_ids"] == (7, 8)
    assert report["schema_version"] == FINAL_TEST_REPORT_SCHEMA_VERSION
    assert report["evaluation_role"] == FINAL_TEST_EVALUATION_ROLE
    assert report["checkpoint_selection_allowed"] is False
    assert report["checkpoint_promotion_allowed"] is False
    assert report["checkpoint_selected_before_evaluation"] is True
    assert paths.best_checkpoint.read_bytes() == checkpoint_before
    assert paths.best_metrics.read_bytes() == metrics_before


def test_final_test_rejects_unselected_candidate_checkpoint(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")

    with pytest.raises(
        ValueError,
        match="selected best checkpoint",
    ):
        run_final_test_evaluation(
            paths=paths,
            checkpoint=candidate,
            config=EvaluationConfig(),
            physics_config=DEFAULT_PHYSICS_CONFIG,
        )


def test_final_test_detects_selection_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    config = EvaluationConfig()
    _patch_evaluation_contracts(monkeypatch)

    def fake_run_evaluate(**kwargs):
        metrics = {"physically_secure_rate_requested": 0.5}
        _write_fake_evaluation(
            output_dir=Path(kwargs["output_dir"]),
            config=kwargs["config"],
            metrics=metrics,
        )
        paths.best_metrics.write_text(
            json.dumps({"selection_metric": 1.0}),
            encoding="utf-8",
        )
        return metrics

    monkeypatch.setattr(
        final_test_module,
        "run_evaluate",
        fake_run_evaluate,
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint-selection metrics",
    ):
        run_final_test_evaluation(
            paths=paths,
            checkpoint=paths.best_checkpoint,
            config=config,
            physics_config=DEFAULT_PHYSICS_CONFIG,
        )


def _raw_config(tmp_path: Path) -> dict[str, object]:
    return {
        "run_name": "final_test_boundary",
        "seed": 7,
        "n_iterations": 1,
        "n_scenarios_per_iteration": 1,
        "epochs_per_iteration": 1,
        "pool": {
            "transitions_csv": "pool.csv",
            "raw_dir": "pool_raw",
            "metadata_path": "run/pool.json",
        },
        "eval_csv": "eval.csv",
        "eval_raw_dir": "eval_raw",
        "final_test_csv": "final.csv",
        "final_test_raw_dir": "final_raw",
        "bootstrap_checkpoint": "bootstrap.pt",
        "bootstrap_eval_metrics": "bootstrap.json",
        "checkpoint_dir": "run",
        "best_checkpoint_path": "run/best.pt",
        "best_metrics_path": "run/best_metrics.json",
        "replay_buffer": {
            "max_size": 20,
            "min_size_to_train": 1,
            "fresh_fraction": 0.5,
            "random_seed": 7,
        },
        "generation": {
            "simulations": 2,
            "depth": 1,
            "max_steps": 2,
            "top_k": 2,
        },
        "training": {
            "examples_per_iteration": 2,
            "batch_size": 1,
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
            "metric": "physically_secure_rate_requested",
            "min_improvement": 0.0,
        },
    }


def test_pipeline_runs_final_test_after_checkpoint_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_config(tmp_path)
    config = SelfPlayConfig.from_mapping(raw)
    paths = SelfPlayPaths.from_config(config, tmp_path)
    metric = config.acceptance.metric
    calls: list[str] = []

    monkeypatch.setattr(
        pipeline_module,
        "resolve_run_state",
        lambda **kwargs: SimpleNamespace(
            start_iteration=1,
            completed_iterations=(),
        ),
    )
    monkeypatch.setattr(pipeline_module, "save_yaml", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline_module,
        "initialize_best_state",
        lambda **kwargs: SimpleNamespace(
            checkpoint=paths.best_checkpoint,
            metrics={metric: 0.1},
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "require_metrics_pf_alg",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "require_metrics_physics_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "initialize_pool_metadata",
        lambda **kwargs: {"scenarios": []},
    )

    class Replay:
        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(
        pipeline_module,
        "RollingReplayBuffer",
        lambda **kwargs: Replay(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_learning_curve",
        lambda path: [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "upsert_iteration_row",
        lambda *, rows, row: [*rows, row],
    )
    monkeypatch.setattr(
        pipeline_module,
        "save_learning_curve",
        lambda **kwargs: None,
    )

    selected_checkpoint = tmp_path / "run" / "best-after.pt"

    def fake_iteration(request):
        calls.append("iteration")
        return IterationResult(
            iteration=1,
            accepted=True,
            status="ACCEPTED",
            selected_scenario_ids=(1,),
            raw_examples_csv=Path("raw.csv"),
            train_batch_csv=Path("batch.csv"),
            train_examples_csv=Path("train.csv"),
            validation_examples_csv=Path("validation.csv"),
            split_metadata_path=Path("split.json"),
            candidate_checkpoint=Path("candidate.pt"),
            metadata_path=Path("metadata.json"),
            parent_metrics={metric: 0.1},
            candidate_metrics={metric: 0.4},
            best_checkpoint=selected_checkpoint,
            best_metrics={metric: 0.4},
            pool_metadata={"scenarios": []},
            learning_curve_row={
                "iteration": 1,
                "n_fresh": 1,
                "n_old": 0,
            },
        )

    monkeypatch.setattr(
        pipeline_module,
        "run_self_play_iteration",
        fake_iteration,
    )
    monkeypatch.setattr(
        pipeline_module,
        "write_iteration_completion_marker",
        lambda **kwargs: calls.append("completion"),
    )

    def fake_final_test(**kwargs):
        calls.append("final-test")
        assert kwargs["checkpoint"] == selected_checkpoint
        return FinalTestEvaluation(
            metrics={metric: 0.99},
            metrics_path=tmp_path / "final_metrics.json",
            results_path=tmp_path / "final_results.csv",
            report_path=tmp_path / "final_report.json",
            checkpoint=selected_checkpoint,
        )

    monkeypatch.setattr(
        pipeline_module,
        "run_final_test_evaluation",
        fake_final_test,
    )

    result = run_self_play_pipeline(
        PipelineRequest(
            config=config,
            raw_config=raw,
            paths=paths,
        )
    )

    assert calls == ["iteration", "completion", "final-test"]
    assert result.best_checkpoint == selected_checkpoint
    assert result.best_metrics == {metric: 0.4}
    assert result.final_test_metrics == {metric: 0.99}


def test_final_test_selection_boundary_is_structural() -> None:
    final_test_source = Path(
        "grid_topology_ai/self_play/final_test.py"
    ).read_text(encoding="utf-8")
    iteration_source = Path(
        "grid_topology_ai/self_play/iteration.py"
    ).read_text(encoding="utf-8")
    pipeline_source = Path(
        "grid_topology_ai/self_play/pipeline.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "accept_candidate",
        "passes_confidence_gates",
        "promote_candidate",
    ):
        assert forbidden not in final_test_source
    assert "final_test" not in iteration_source
    assert '"checkpoint_selection_allowed": False' in final_test_source
    assert pipeline_source.rfind(
        "final_test = _run_reporting_only_final_test("
    ) > pipeline_source.index(
        "for iteration in range("
    )

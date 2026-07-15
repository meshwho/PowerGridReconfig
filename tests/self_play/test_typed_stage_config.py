from __future__ import annotations

import sys
from pathlib import Path

import pytest

from grid_topology_ai.config import (
    EvaluationConfig,
    GenerationConfig,
    TrainingConfig,
)
from grid_topology_ai.self_play.artifacts import save_json
from scripts.self_play import run_iteration


def test_run_generate_uses_generation_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = []

    def fake_generate(request):
        captured.append(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        examples_csv = request.output_dir / "examples.csv"
        examples_csv.write_text(
            "scenario_id,outcome_value_target\n1,0.5\n2,0.4\n",
            encoding="utf-8",
        )
        print("generated examples")
        return examples_csv

    monkeypatch.setattr(run_iteration, "generate_self_play_examples", fake_generate)
    original_cwd = Path.cwd()
    transitions_csv = tmp_path / "transitions.csv"
    transitions_csv.write_text("scenario_id\n1\n2\n3\n", encoding="utf-8")
    config = GenerationConfig(simulations=17, gamma=0.91)

    examples_csv = run_iteration.run_generate(
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        transitions_csv=transitions_csv,
        scenario_ids=[1, 2],
        checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "generated",
        config=config,
        base_seed=100,
        iteration=3,
    )

    request = captured[0]
    assert request.config is config
    assert request.seed == 103
    assert request.checkpoint == tmp_path / "best.pt"
    assert request.raw_dir == tmp_path / "raw"
    assert request.output_dir == tmp_path / "generated"
    assert request.clear_cache_between_scenarios is True
    selected_transitions = tmp_path / "generated" / "selected_transitions.csv"
    assert selected_transitions.exists()
    assert "3" not in selected_transitions.read_text(encoding="utf-8")
    assert examples_csv == tmp_path / "generated" / "examples.csv"
    assert (tmp_path / "generated" / "generate.log").exists()
    assert Path.cwd() == original_cwd


def test_run_train_uses_training_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = []

    def fake_train(request):
        captured.append(request)
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(b"checkpoint")
        print("trained model")
        return request.output_path

    monkeypatch.setattr(run_iteration, "train_graph_policy_value_model", fake_train)
    original_cwd = Path.cwd()
    config = TrainingConfig(epochs=4, batch_size=9, no_tensorboard=True)

    checkpoint = run_iteration.run_train(
        project_root=tmp_path,
        examples_csv=tmp_path / "examples.csv",
        init_checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "train",
        config=config,
        iteration=5,
    )

    request = captured[0]
    assert request.config is config
    assert request.project_root == tmp_path.resolve()
    assert request.examples_csv == tmp_path / "examples.csv"
    assert request.init_checkpoint == tmp_path / "best.pt"
    assert request.output_path == tmp_path / "train" / "candidate_checkpoint.pt"
    assert request.save_best is True
    assert request.use_amp is False
    assert request.normalize_features is True
    assert request.validation_examples_csv is None
    assert request.metrics_csv == tmp_path / "train" / "train_metrics.csv"
    assert request.run_name == "self_play_iter_005"
    assert checkpoint == tmp_path / "train" / "candidate_checkpoint.pt"
    assert (tmp_path / "train" / "train.log").exists()
    assert Path.cwd() == original_cwd


def test_run_evaluate_uses_evaluation_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = []

    def fake_evaluate(request):
        captured.append(request)
        request.output_csv.parent.mkdir(parents=True, exist_ok=True)
        request.output_csv.write_text("scenario_id,solved\n1,true\n", encoding="utf-8")
        metrics = {"solve_rate": 0.8, "source": "json"}
        save_json(metrics, request.output_json)
        print("evaluated checkpoint")
        return {"solve_rate": 0.1}

    monkeypatch.setattr(run_iteration, "evaluate_checkpoint", fake_evaluate)
    original_cwd = Path.cwd()
    config = EvaluationConfig(
        output_csv_name="custom_eval.csv",
        output_json_name="custom_metrics.json",
    )

    metrics = run_iteration.run_evaluate(
        project_root=tmp_path,
        checkpoint=tmp_path / "candidate.pt",
        eval_csv=tmp_path / "eval.csv",
        eval_raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "eval",
        config=config,
    )

    request = captured[0]
    assert request.config is config
    assert request.raw_dir == tmp_path / "raw"
    assert request.transitions_csv == tmp_path / "eval.csv"
    assert request.checkpoint == tmp_path / "candidate.pt"
    assert request.output_csv == tmp_path / "eval" / "custom_eval.csv"
    assert request.output_json == tmp_path / "eval" / "custom_metrics.json"
    assert request.quiet is True
    assert request.limit is None
    assert request.pf_alg == 1
    assert request.disable_cache is False
    assert request.leaf_penalty_weight == 0.10
    assert request.stop_policy == "no_hard_overloads"
    assert request.use_dc_screening is False
    assert metrics == {"solve_rate": 0.8, "source": "json"}
    assert (tmp_path / "eval" / "evaluate.log").exists()
    assert Path.cwd() == original_cwd


def test_stage_output_logs_exception_and_restores_streams(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_generate(request):
        print("before failure")
        raise RuntimeError("stage failed")

    monkeypatch.setattr(run_iteration, "generate_self_play_examples", fake_generate)
    original_cwd = Path.cwd()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    transitions_csv = tmp_path / "transitions.csv"
    transitions_csv.write_text("scenario_id\n1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stage failed"):
        run_iteration.run_generate(
            project_root=tmp_path,
            raw_dir=tmp_path / "raw",
            transitions_csv=transitions_csv,
            scenario_ids=[1],
            checkpoint=tmp_path / "best.pt",
            output_dir=tmp_path / "generated",
            config=GenerationConfig(),
            base_seed=1,
            iteration=1,
        )

    log_path = tmp_path / "generated" / "generate.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "stage failed" in log_text
    assert Path.cwd() == original_cwd
    print("stdout still works")
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert not sys.stdout.closed
    assert not sys.stderr.closed


def test_working_directory_restores_after_exception(tmp_path: Path) -> None:
    original_cwd = Path.cwd()

    with pytest.raises(RuntimeError, match="boom"):
        with run_iteration._working_directory(tmp_path):
            assert Path.cwd() == tmp_path
            raise RuntimeError("boom")

    assert Path.cwd() == original_cwd

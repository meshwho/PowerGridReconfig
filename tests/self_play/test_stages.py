from __future__ import annotations

from pathlib import Path

import pytest

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.self_play import stages
from grid_topology_ai.self_play.artifacts import save_json
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION


def _checkpoint_metadata(selector: str) -> dict[str, object]:
    return {
        "checkpoint_selection_metric": selector,
        "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    }


def test_run_evaluate_resolves_config_pf_alg(tmp_path: Path, monkeypatch) -> None:
    captured = []

    def fake_evaluate(request):
        captured.append(request)
        request.output_csv.parent.mkdir(parents=True, exist_ok=True)
        request.output_csv.write_text("scenario_id,solved\n1,true\n", encoding="utf-8")
        save_json({"solve_rate": 1.0, "pf_alg": 3}, request.output_json)
        return {"solve_rate": 1.0, "pf_alg": 3}

    monkeypatch.setattr(stages, "evaluate_checkpoint", fake_evaluate)
    stages.run_evaluate(
        project_root=tmp_path,
        checkpoint=tmp_path / "candidate.pt",
        eval_csv=tmp_path / "eval.csv",
        eval_raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "eval",
        config=EvaluationConfig(pf_alg=3),
    )

    assert captured[0].pf_alg is None
    assert captured[0].resolved_pf_alg == 3

import torch
from grid_topology_ai.config import TrainingConfig


def test_run_train_requires_validation_csv(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Validation"):
        stages.run_train(
            project_root=tmp_path,
            examples_csv=train_csv,
            validation_examples_csv=tmp_path / "missing.csv",
            init_checkpoint=tmp_path / "best.pt",
            output_dir=tmp_path / "train",
            config=TrainingConfig(),
            iteration=1,
            seed=8,
        )


def test_run_train_passes_validation_and_seed(tmp_path: Path, monkeypatch) -> None:
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    train_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    validation_csv.write_text("scenario_id\n2\n", encoding="utf-8")
    captured = []

    def fake_train(request):
        captured.append(request)
        torch.save(_checkpoint_metadata("validation_loss"), request.output_path)
        return request.output_path

    monkeypatch.setattr(stages, "train_graph_policy_value_model", fake_train)
    stages.run_train(
        project_root=tmp_path,
        examples_csv=train_csv,
        validation_examples_csv=validation_csv,
        init_checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "train",
        config=TrainingConfig(),
        iteration=1,
        seed=8,
    )
    request = captured[0]
    assert request.examples_csv == train_csv
    assert request.validation_examples_csv == validation_csv
    assert request.seed == 8
    assert request.save_best is True


def test_run_train_rejects_non_validation_selector(tmp_path: Path, monkeypatch) -> None:
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    train_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    validation_csv.write_text("scenario_id\n2\n", encoding="utf-8")

    def fake_train(request):
        torch.save(_checkpoint_metadata("training_loss"), request.output_path)
        return request.output_path

    monkeypatch.setattr(stages, "train_graph_policy_value_model", fake_train)
    with pytest.raises(RuntimeError, match="validation_loss"):
        stages.run_train(
            project_root=tmp_path,
            examples_csv=train_csv,
            validation_examples_csv=validation_csv,
            init_checkpoint=tmp_path / "best.pt",
            output_dir=tmp_path / "train",
            config=TrainingConfig(),
            iteration=1,
            seed=8,
        )

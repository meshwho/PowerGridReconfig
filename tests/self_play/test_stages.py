from __future__ import annotations

import json
from pathlib import Path

import pytest

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.self_play import stages
from grid_topology_ai.self_play.artifacts import save_json


def _checkpoint_metadata(selector: str) -> dict[str, object]:
    return {
        "checkpoint_selection_metric": selector,
        "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
        **physics_provenance(DEFAULT_PHYSICS_CONFIG),
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
            physics_config=DEFAULT_PHYSICS_CONFIG,
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
        physics_config=DEFAULT_PHYSICS_CONFIG,
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
            physics_config=DEFAULT_PHYSICS_CONFIG,
            iteration=1,
            seed=8,
        )


import pandas as pd


def _stage_rows(
    *, reason: str = "solved", solved: bool = True, count: int = 1
) -> list[dict[str, object]]:
    provenance = physics_provenance(DEFAULT_PHYSICS_CONFIG)
    return [
        {
            "state_path": "unused.npz",
            "mcts_policy_json": '{"0": 1.0}',
            "state_id": f"state-{step}",
            "scenario_id": 1,
            "step": step,
            "solved": solved,
            "done": True,
            "termination_reason": reason,
            "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
            "outcome_value_target_contract_version": (
                OUTCOME_VALUE_TARGET_CONTRACT_VERSION
            ),
            "physics_config_contract_version": provenance[
                "physics_config_contract_version"
            ],
            "physics_config": json.dumps(
                provenance["physics_config"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "physics_config_fingerprint": provenance[
                "physics_config_fingerprint"
            ],
        }
        for step in range(count)
    ]


@pytest.mark.parametrize(
    "reason, solved, expected",
    [
        ("solved", True, 0.95),
        ("handoff_to_redispatch", False, 0.0),
        ("max_steps_reached", False, -0.95),
    ],
)
def test_ensure_outcome_value_targets_valid_terminal_cases(
    tmp_path: Path, reason: str, solved: bool, expected: float
) -> None:
    examples_csv = tmp_path / "examples.csv"
    pd.DataFrame(_stage_rows(reason=reason, solved=solved, count=2)).to_csv(
        examples_csv, index=False
    )
    stages.ensure_outcome_value_targets(examples_csv, gamma=0.95)
    result = pd.read_csv(examples_csv)
    assert result.loc[1, "outcome_value_target"] == pytest.approx(expected)
    assert result.loc[0, "outcome_value_target"] == pytest.approx(
        expected * 0.95 if expected else 0.0
    )
    from grid_topology_ai.self_play.example_validation import (
        validate_example_contract_versions,
        validate_example_outcome_contracts,
    )

    validate_example_contract_versions(result, source_path=examples_csv)
    validate_example_outcome_contracts(result, source_path=examples_csv)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: None,
        lambda rows: rows.__setitem__(0, {**rows[0], "scenario_id": float("inf")}),
        lambda rows: rows.__setitem__(0, {**rows[0], "done": False}),
        lambda rows: rows.__setitem__(
            1, {**rows[1], "termination_reason": "max_steps_reached"}
        ),
        lambda rows: rows.__setitem__(1, {**rows[1], "step": 0}),
    ],
    ids=["gamma_nan", "scenario_inf", "incomplete", "mixed_outcomes", "duplicate_step"],
)
def test_ensure_outcome_value_targets_invalid_is_atomic(tmp_path: Path, mutate) -> None:
    rows = _stage_rows(count=2)
    mutate(rows)
    examples_csv = tmp_path / "examples.csv"
    pd.DataFrame(rows).to_csv(examples_csv, index=False)
    before = examples_csv.read_bytes()
    gamma = (
        float("nan")
        if mutate.__name__ == "<lambda>" and rows == _stage_rows(count=2)
        else 0.95
    )
    with pytest.raises(ValueError):
        stages.ensure_outcome_value_targets(examples_csv, gamma=gamma)
    assert examples_csv.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_ensure_outcome_value_targets_rejects_existing_positive_handoff_atomically(
    tmp_path: Path,
) -> None:
    rows = _stage_rows(reason="handoff_to_redispatch", solved=False)
    rows[0].update(
        outcome_value_target=0.95,
        outcome_value_target_contract_version=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
        outcome_class="handoff_to_redispatch",
        outcome_steps_to_terminal=1,
        outcome_value_target_mode="alphazero_discounted",
        outcome_gamma=0.95,
    )
    path = tmp_path / "examples.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        stages.ensure_outcome_value_targets(path, gamma=0.95)
    assert path.read_bytes() == before


def test_ensure_outcome_value_targets_rejects_legacy_outcome_version_atomically(
    tmp_path: Path,
) -> None:
    rows = _stage_rows()
    rows[0]["outcome_value_target_contract_version"] = 1
    path = tmp_path / "examples.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        stages.ensure_outcome_value_targets(path, gamma=0.95)
    assert path.read_bytes() == before


@pytest.mark.parametrize("gamma", [True, float("nan")])
def test_ensure_outcome_value_targets_rejects_invalid_gamma_atomically(
    tmp_path: Path, gamma: object
) -> None:
    path = tmp_path / "examples.csv"
    pd.DataFrame(_stage_rows()).to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="gamma"):
        stages.ensure_outcome_value_targets(path, gamma=gamma)  # type: ignore[arg-type]
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))

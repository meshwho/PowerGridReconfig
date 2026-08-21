from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from grid_topology_ai.config import (
    EvaluationConfig,
    GenerationConfig,
    TrainingConfig,
)
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.physics.objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.self_play import stages
from grid_topology_ai.self_play.artifacts import save_json
from grid_topology_ai.state.schema import BRANCH_FEATURE_COLUMNS, BUS_FEATURE_COLUMNS
from grid_topology_ai.topology_actions import (
    STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    action_layout_to_list,
)
from tests.topology_contract_helpers import (
    TEST_ACTION_SPACE_CONFIG,
    test_action_layout,
)


def _checkpoint_metadata(selector: str) -> dict[str, object]:
    num_bus_features = len(BUS_FEATURE_COLUMNS)
    num_branch_features = len(BRANCH_FEATURE_COLUMNS)
    return {
        "checkpoint_selection_metric": selector,
        "physics_config": DEFAULT_PHYSICS_CONFIG.to_dict(),
        "topology_action_config": TEST_ACTION_SPACE_CONFIG.to_contract_dict(),
        "action_layout": action_layout_to_list(test_action_layout((0,))),
        "policy_layout": STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
        "model_type": "graph_policy_value_net_v2",
        "topology_cardinality_independent": True,
        "model_state_dict": {},
        "num_bus_features": num_bus_features,
        "num_branch_features": num_branch_features,
        "hidden_dim": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "bus_feature_mean": np.zeros(num_bus_features, dtype=np.float32),
        "bus_feature_std": np.ones(num_bus_features, dtype=np.float32),
        "branch_feature_mean": np.zeros(num_branch_features, dtype=np.float32),
        "branch_feature_std": np.ones(num_branch_features, dtype=np.float32),
    }


def test_run_evaluate_resolves_config_pf_alg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def fake_evaluate(request):
        captured.append(request)
        request.output_csv.parent.mkdir(parents=True, exist_ok=True)
        request.output_csv.write_text(
            "scenario_id,solved\n1,true\n",
            encoding="utf-8",
        )
        save_json(
            {"solve_rate": 1.0, "pf_alg": 3},
            request.output_json,
        )
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
    assert captured[0].project_root == tmp_path


def test_run_generate_returns_complete_generator_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions_csv = tmp_path / "transitions.csv"
    transitions_csv.write_text(
        "scenario_id\n1\n",
        encoding="utf-8",
    )
    generated_csv = tmp_path / "generation" / "examples.csv"
    captured = []

    def fake_generate(request):
        captured.append(request)
        generated_csv.parent.mkdir(parents=True, exist_ok=True)
        generated_csv.write_text(
            "outcome_value_target,outcome_gamma\n1.0,1.0\n",
            encoding="utf-8",
        )
        return generated_csv

    monkeypatch.setattr(
        stages,
        "generate_self_play_examples",
        fake_generate,
    )

    result = stages.run_generate(
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        transitions_csv=transitions_csv,
        scenario_ids=[1],
        checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "generation",
        config=GenerationConfig(),
        physics_config=DEFAULT_PHYSICS_CONFIG,
        mcts_seed=7,
        action_seed=8,
        iteration=2,
    )

    assert result == generated_csv
    assert captured[0].iteration == 2
    assert captured[0].scenario_ids is None
    assert captured[0].transitions_csv == (
        tmp_path / "generation" / "selected_transitions.csv"
    )
    assert not hasattr(stages, "ensure_outcome_value_targets")


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


def test_run_train_passes_validation_and_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    train_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    validation_csv.write_text("scenario_id\n2\n", encoding="utf-8")
    captured = []

    def fake_train(request):
        captured.append(request)
        torch.save(
            _checkpoint_metadata("validation_loss"),
            request.output_path,
        )
        return request.output_path

    monkeypatch.setattr(
        stages,
        "train_graph_policy_value_model",
        fake_train,
    )
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


def test_run_train_rejects_non_validation_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    train_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    validation_csv.write_text("scenario_id\n2\n", encoding="utf-8")

    def fake_train(request):
        torch.save(
            _checkpoint_metadata("training_loss"),
            request.output_path,
        )
        return request.output_path

    monkeypatch.setattr(
        stages,
        "train_graph_policy_value_model",
        fake_train,
    )

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

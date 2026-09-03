from __future__ import annotations

import numpy as np
import pytest
import torch

from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.model import GraphPolicyValueNetV2
from grid_topology_ai.evaluator import (
    NeuralPolicyValueEvaluator,
    require_checkpoint_contracts,
)
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, BUS_FEATURE_COLUMNS
from grid_topology_ai.actions import (
    STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    action_layout_to_list,
)
from tests.topology_contract_helpers import (
    TEST_ACTION_SPACE_CONFIG,
    test_action_layout,
)


def _checkpoint_payload() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    num_bus_features = len(BUS_FEATURE_COLUMNS)
    num_branch_features = len(BRANCH_FEATURE_COLUMNS)
    bus_mean = np.arange(num_bus_features, dtype=np.float32) + 10.0
    bus_std = np.arange(num_bus_features, dtype=np.float32) + 1.0
    branch_mean = np.arange(num_branch_features, dtype=np.float32) + 100.0
    branch_std = np.arange(num_branch_features, dtype=np.float32) + 1.0

    model = GraphPolicyValueNetV2(
        num_bus_features=num_bus_features,
        num_branch_features=num_branch_features,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    payload: dict[str, object] = {
        "physics_config": DEFAULT_PHYSICS_CONFIG.to_dict(),
        "topology_action_config": TEST_ACTION_SPACE_CONFIG.to_contract_dict(),
        "action_layout": action_layout_to_list(test_action_layout((10, 20))),
        "policy_layout": STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
        "model_type": "graph_policy_value_net_v2",
        "topology_cardinality_independent": True,
        "model_state_dict": model.state_dict(),
        "num_bus_features": num_bus_features,
        "num_branch_features": num_branch_features,
        "hidden_dim": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "bus_feature_mean": bus_mean,
        "bus_feature_std": bus_std,
        "branch_feature_mean": branch_mean,
        "branch_feature_std": branch_std,
    }
    arrays = {
        "bus_feature_mean": bus_mean,
        "bus_feature_std": bus_std,
        "branch_feature_mean": branch_mean,
        "branch_feature_std": branch_std,
    }
    return payload, arrays


def test_graph_v2_evaluator_loads_checkpoint_normalization_arrays(tmp_path) -> None:
    payload, arrays = _checkpoint_payload()
    checkpoint_path = tmp_path / "candidate.pt"
    torch.save(payload, checkpoint_path)

    evaluator = NeuralPolicyValueEvaluator(checkpoint_path, device="cpu")

    np.testing.assert_array_equal(
        evaluator.bus_feature_mean,
        arrays["bus_feature_mean"],
    )
    np.testing.assert_array_equal(
        evaluator.bus_feature_std,
        arrays["bus_feature_std"],
    )
    np.testing.assert_array_equal(
        evaluator.branch_feature_mean,
        arrays["branch_feature_mean"],
    )
    np.testing.assert_array_equal(
        evaluator.branch_feature_std,
        arrays["branch_feature_std"],
    )


def test_current_graph_checkpoint_contract_is_complete() -> None:
    payload, _ = _checkpoint_payload()

    physics = require_checkpoint_contracts(payload, source="checkpoint.pt")

    assert physics == DEFAULT_PHYSICS_CONFIG


@pytest.mark.parametrize(
    "field",
    [
        "model_type",
        "topology_cardinality_independent",
        "policy_layout",
        "model_state_dict",
        "hidden_dim",
        "num_layers",
        "dropout",
        "num_bus_features",
        "num_branch_features",
        "bus_feature_mean",
        "bus_feature_std",
        "branch_feature_mean",
        "branch_feature_std",
        "topology_action_config",
        "action_layout",
        "physics_config",
    ],
)
def test_graph_checkpoint_rejects_missing_required_field(field: str) -> None:
    payload, _ = _checkpoint_payload()
    del payload[field]

    with pytest.raises(ValueError, match="missing required fields"):
        require_checkpoint_contracts(payload, source="checkpoint.pt")


def test_graph_checkpoint_rejects_non_current_model_type() -> None:
    payload, _ = _checkpoint_payload()
    payload["model_type"] = "graph_policy_value_net"

    with pytest.raises(ValueError, match="model_type"):
        require_checkpoint_contracts(payload, source="checkpoint.pt")


def test_graph_checkpoint_requires_cardinality_independence() -> None:
    payload, _ = _checkpoint_payload()
    payload["topology_cardinality_independent"] = False

    with pytest.raises(ValueError, match="topology-cardinality independent"):
        require_checkpoint_contracts(payload, source="checkpoint.pt")

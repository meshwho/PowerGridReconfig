from __future__ import annotations

import numpy as np
import torch

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.models.graph_self_play_dataset import (
    GRAPH_BATCHING_CONTRACT_VERSION,
)
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.state_schema import BRANCH_FEATURE_COLUMNS, BUS_FEATURE_COLUMNS
from tests.topology_contract_helpers import checkpoint_topology_fields


def test_graph_v2_evaluator_loads_checkpoint_normalization_arrays(tmp_path) -> None:
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
    checkpoint_path = tmp_path / "candidate.pt"
    torch.save(
        {
            "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
            "graph_batching_contract_version": GRAPH_BATCHING_CONTRACT_VERSION,
            "topology_cardinality_independent": True,
            "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
            "outcome_objective_version": OUTCOME_OBJECTIVE_VERSION,
            "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
            **physics_provenance(DEFAULT_PHYSICS_CONFIG),
            **checkpoint_topology_fields((10, 20)),
            "model_type": "graph_policy_value_net_v2",
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
        },
        checkpoint_path,
    )

    evaluator = NeuralPolicyValueEvaluator(checkpoint_path, device="cpu")

    np.testing.assert_array_equal(evaluator.bus_feature_mean, bus_mean)
    np.testing.assert_array_equal(evaluator.bus_feature_std, bus_std)
    np.testing.assert_array_equal(evaluator.branch_feature_mean, branch_mean)
    np.testing.assert_array_equal(evaluator.branch_feature_std, branch_std)

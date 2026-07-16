from __future__ import annotations

import numpy as np
import torch

from grid_topology_ai.models.graph_policy_value_net import GraphPolicyValueNet
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator


def test_graph_evaluator_loads_checkpoint_normalization_arrays(tmp_path):
    model = GraphPolicyValueNet(
        num_bus_features=3,
        num_branch_features=4,
        num_actions=3,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    checkpoint_path = tmp_path / "candidate.pt"
    torch.save(
        {
            "model_type": "graph_policy_value_net",
            "model_state_dict": model.state_dict(),
            "num_bus_features": 3,
            "num_branch_features": 4,
            "num_buses": 2,
            "num_branches": 2,
            "num_actions": 3,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "bus_feature_mean": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "bus_feature_std": np.array([2.0, 4.0, 5.0], dtype=np.float32),
            "branch_feature_mean": np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
            "branch_feature_std": np.array([10.0, 20.0, 25.0, 40.0], dtype=np.float32),
        },
        checkpoint_path,
    )

    evaluator = NeuralPolicyValueEvaluator(checkpoint_path, device="cpu")

    np.testing.assert_array_equal(evaluator.bus_feature_mean, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    np.testing.assert_array_equal(evaluator.bus_feature_std, np.array([2.0, 4.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(evaluator.branch_feature_mean, np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32))
    np.testing.assert_array_equal(evaluator.branch_feature_std, np.array([10.0, 20.0, 25.0, 40.0], dtype=np.float32))

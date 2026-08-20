from __future__ import annotations

import numpy as np
import torch

from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.training.graph_policy_value import _forward_graph_model


class _RecordingV2(GraphPolicyValueNetV2):
    def __init__(self) -> None:
        super().__init__(
            num_bus_features=2,
            num_branch_features=3,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        )
        self.forward_kwargs: dict[str, torch.Tensor] = {}

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return torch.zeros(1, 3), torch.zeros(1)


def test_training_forward_passes_physical_edge_mask() -> None:
    model = _RecordingV2()
    edge_active_mask = torch.tensor([True, False])
    action_mask = torch.tensor([[True, True, False]])

    _forward_graph_model(
        model,
        bus_features=torch.zeros(2, 2),
        branch_features=torch.zeros(2, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_active_mask=edge_active_mask,
        action_mask=action_mask,
        node_batch=torch.zeros(2, dtype=torch.long),
        edge_batch=torch.zeros(2, dtype=torch.long),
    )

    assert torch.equal(model.forward_kwargs["edge_active_mask"], edge_active_mask)
    assert torch.equal(model.forward_kwargs["action_mask"], action_mask)


def _state() -> GridFMState:
    return GridFMState(
        scenario_id=7,
        load_scenario_idx=0.0,
        bus_features=np.zeros((2, 2), dtype=np.float32),
        branch_features=np.zeros((2, 3), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        branch_ids=np.array([10, 20], dtype=np.int64),
        branch_status=np.array([1.0, 0.0], dtype=np.float32),
        metrics={},
        outaged_branch_ids=[20],
    )


def test_neural_evaluator_builds_mask_from_branch_status() -> None:
    model = _RecordingV2()
    evaluator = object.__new__(NeuralPolicyValueEvaluator)
    evaluator.model = model
    evaluator.device = torch.device("cpu")
    evaluator.num_bus_features = 2
    evaluator.num_branch_features = 3
    evaluator.bus_feature_mean = np.zeros(2, dtype=np.float32)
    evaluator.bus_feature_std = np.ones(2, dtype=np.float32)
    evaluator.branch_feature_mean = np.zeros(3, dtype=np.float32)
    evaluator.branch_feature_std = np.ones(3, dtype=np.float32)

    evaluator._evaluate_graph(
        _state(),
        np.array([True, True, False], dtype=bool),
    )

    assert torch.equal(
        model.forward_kwargs["edge_active_mask"],
        torch.tensor([True, False]),
    )

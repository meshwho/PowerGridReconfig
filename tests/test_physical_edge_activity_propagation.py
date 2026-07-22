from __future__ import annotations

import numpy as np
import torch

from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.models.graph_policy_value_net import GraphPolicyValueNet
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.training.graph_policy_value import _forward_graph_model


class _RecordingV2(GraphPolicyValueNetV2):
    def __init__(self) -> None:
        super().__init__(
            num_bus_features=2,
            num_branch_features=3,
            num_actions=3,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        )
        self.forward_kwargs: dict[str, torch.Tensor] = {}

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        batch_size = int(kwargs["bus_features"].shape[0])
        return torch.zeros(batch_size, 3), torch.zeros(batch_size)


class _RecordingV1(GraphPolicyValueNet):
    def __init__(self) -> None:
        super().__init__(
            num_bus_features=2,
            num_branch_features=3,
            num_actions=3,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        )
        self.forward_kwargs: dict[str, torch.Tensor] = {}

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        batch_size = int(kwargs["bus_features"].shape[0])
        return torch.zeros(batch_size, 3), torch.zeros(batch_size)


def _graph_inputs() -> dict[str, torch.Tensor]:
    return {
        "bus_features": torch.zeros(1, 2, 2),
        "branch_features": torch.zeros(1, 2, 3),
        "edge_index": torch.tensor([[[0, 1], [1, 0]]]),
        "edge_active_mask": torch.tensor([[True, False]]),
        "action_mask": torch.tensor([[True, True, False]]),
    }


def test_training_forward_passes_physical_mask_only_to_v2():
    inputs = _graph_inputs()
    model_v2 = _RecordingV2()
    model_v1 = _RecordingV1()

    _forward_graph_model(model_v2, **inputs)
    _forward_graph_model(model_v1, **inputs)

    assert torch.equal(
        model_v2.forward_kwargs["edge_active_mask"],
        inputs["edge_active_mask"],
    )
    assert "edge_active_mask" not in model_v1.forward_kwargs
    assert torch.equal(
        model_v2.forward_kwargs["action_mask"],
        inputs["action_mask"],
    )
    assert torch.equal(
        model_v1.forward_kwargs["action_mask"],
        inputs["action_mask"],
    )


class _EvaluatorRecordingV2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs: dict[str, torch.Tensor] = {}

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return torch.zeros(1, 3), torch.zeros(1)


class _EvaluatorRecordingV1(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs: dict[str, torch.Tensor] = {}

    def forward(
        self,
        *,
        bus_features,
        branch_features,
        edge_index,
        action_mask,
    ):
        self.forward_kwargs = {
            "bus_features": bus_features,
            "branch_features": branch_features,
            "edge_index": edge_index,
            "action_mask": action_mask,
        }
        return torch.zeros(1, 3), torch.zeros(1)


def _evaluator(model_type: str, model: torch.nn.Module):
    evaluator = object.__new__(NeuralPolicyValueEvaluator)
    evaluator.model_type = model_type
    evaluator.model = model
    evaluator.device = torch.device("cpu")
    evaluator.num_buses = 2
    evaluator.num_branches = 2
    evaluator.num_bus_features = 2
    evaluator.num_branch_features = 3
    evaluator.bus_feature_mean = np.zeros(2, dtype=np.float32)
    evaluator.bus_feature_std = np.ones(2, dtype=np.float32)
    evaluator.branch_feature_mean = np.zeros(3, dtype=np.float32)
    evaluator.branch_feature_std = np.ones(3, dtype=np.float32)
    return evaluator


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


def test_neural_evaluator_builds_v2_mask_from_branch_status():
    model = _EvaluatorRecordingV2()
    evaluator = _evaluator("graph_policy_value_net_v2", model)
    action_mask = np.array([True, True, False], dtype=bool)

    evaluator._evaluate_graph(_state(), action_mask)

    assert torch.equal(
        model.forward_kwargs["edge_active_mask"],
        torch.tensor([[True, False]]),
    )
    assert torch.equal(
        model.forward_kwargs["action_mask"],
        torch.tensor([[True, True, False]]),
    )


def test_neural_evaluator_keeps_v1_forward_contract():
    model = _EvaluatorRecordingV1()
    evaluator = _evaluator("graph_policy_value_net", model)
    evaluator._evaluate_graph(
        _state(),
        np.array([True, True, False], dtype=bool),
    )

    assert "edge_active_mask" not in model.forward_kwargs

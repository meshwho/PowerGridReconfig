import pytest
import torch

from grid_topology_ai.models.graph_policy_value_net_v2 import (
    GraphPolicyValueNetV2,
    ResidualEdgeMessagePassingV2,
)


def reference_aggregate_messages(
    messages: torch.Tensor,
    target_indices: torch.Tensor,
    num_nodes: int,
    edge_active_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference active-edge mean aggregation using explicit Python loops."""

    batch_size, _, hidden_dim = messages.shape
    aggregated = messages.new_zeros(batch_size, num_nodes, hidden_dim)
    counts = messages.new_zeros(batch_size, num_nodes, 1)

    for batch_idx in range(batch_size):
        active = edge_active_mask[batch_idx]
        if not bool(active.any()):
            continue

        active_targets = target_indices[batch_idx][active].long()
        active_messages = messages[batch_idx][active]
        aggregated[batch_idx].index_add_(
            dim=0,
            index=active_targets,
            source=active_messages,
        )
        counts[batch_idx].index_add_(
            dim=0,
            index=active_targets,
            source=messages.new_ones(int(active.sum().item()), 1),
        )

    return aggregated / counts.clamp_min(1.0)


def test_vectorized_aggregate_matches_reference():
    torch.manual_seed(42)

    batch_size = 5
    num_edges = 17
    num_nodes = 9
    hidden_dim = 13

    messages = torch.randn(batch_size, num_edges, hidden_dim)
    target_indices = torch.randint(
        low=0,
        high=num_nodes,
        size=(batch_size, num_edges),
    )
    edge_active_mask = torch.rand(batch_size, num_edges) > 0.35
    edge_active_mask[:, 0] = True
    edge_active_mask[:, 1] = False

    expected = reference_aggregate_messages(
        messages=messages,
        target_indices=target_indices,
        num_nodes=num_nodes,
        edge_active_mask=edge_active_mask,
    )

    actual = ResidualEdgeMessagePassingV2._aggregate_messages(
        messages=messages,
        target_indices=target_indices,
        num_nodes=num_nodes,
        edge_active_mask=edge_active_mask,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_inactive_edge_does_not_dilute_degree_normalization():
    messages = torch.tensor(
        [[[6.0, 12.0], [1000.0, 2000.0]]],
        dtype=torch.float32,
    )
    target_indices = torch.tensor([[1, 1]], dtype=torch.long)
    edge_active_mask = torch.tensor([[True, False]], dtype=torch.bool)

    aggregated = ResidualEdgeMessagePassingV2._aggregate_messages(
        messages=messages,
        target_indices=target_indices,
        num_nodes=3,
        edge_active_mask=edge_active_mask,
    )

    expected = torch.tensor(
        [[[0.0, 0.0], [6.0, 12.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(aggregated, expected)


def test_graph_policy_value_net_v2_forward_shapes_and_mask():
    torch.manual_seed(123)

    batch_size = 3
    num_buses = 8
    num_branches = 11
    num_bus_features = 4
    num_branch_features = 6
    num_actions = num_branches + 1

    model = GraphPolicyValueNetV2(
        num_bus_features=num_bus_features,
        num_branch_features=num_branch_features,
        num_actions=num_actions,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )

    bus_features = torch.randn(batch_size, num_buses, num_bus_features)
    branch_features = torch.randn(batch_size, num_branches, num_branch_features)
    source = torch.randint(0, num_buses, size=(batch_size, num_branches))
    target = torch.randint(0, num_buses, size=(batch_size, num_branches))
    edge_index = torch.stack([source, target], dim=1)
    edge_active_mask = torch.ones(batch_size, num_branches, dtype=torch.bool)
    action_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)

    # Active physical lines may still be unavailable as switching actions.
    action_mask[:, 3] = False
    action_mask[:, 7] = False

    policy_logits, value = model(
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        edge_active_mask=edge_active_mask,
        action_mask=action_mask,
    )

    assert policy_logits.shape == (batch_size, num_actions)
    assert value.shape == (batch_size,)
    assert torch.isfinite(value).all()
    assert (policy_logits[:, 3] < -1e20).all()
    assert (policy_logits[:, 7] < -1e20).all()

    predicted_action = torch.argmax(policy_logits, dim=1)
    assert (predicted_action != 3).all()
    assert (predicted_action != 7).all()


def test_graph_policy_value_net_v2_rejects_wrong_edge_mask_shape():
    model = GraphPolicyValueNetV2(
        num_bus_features=4,
        num_branch_features=6,
        num_actions=6,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    bus_features = torch.randn(2, 4, 4)
    branch_features = torch.randn(2, 5, 6)
    source = torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 0]])
    target = torch.tensor([[1, 2, 3, 0, 2], [1, 2, 3, 0, 2]])
    edge_index = torch.stack([source, target], dim=1)
    wrong_edge_active_mask = torch.ones(2, 4, dtype=torch.bool)
    action_mask = torch.ones(2, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="edge_active_mask has 4 edges"):
        model(
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            edge_active_mask=wrong_edge_active_mask,
            action_mask=action_mask,
        )


def test_graph_policy_value_net_v2_rejects_valid_action_for_inactive_edge():
    model = GraphPolicyValueNetV2(
        num_bus_features=3,
        num_branch_features=4,
        num_actions=4,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    bus_features = torch.randn(1, 3, 3)
    branch_features = torch.randn(1, 3, 4)
    edge_index = torch.tensor(
        [[[0, 1, 2], [1, 2, 0]]],
        dtype=torch.long,
    )
    edge_active_mask = torch.tensor([[True, False, True]])
    action_mask = torch.tensor([[True, True, True, True]])

    with pytest.raises(ValueError, match="physically inactive branch action"):
        model(
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            edge_active_mask=edge_active_mask,
            action_mask=action_mask,
        )

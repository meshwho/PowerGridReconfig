import torch

from grid_topology_ai.models.graph_policy_value_net_v2 import (
    GraphPolicyValueNetV2,
    ResidualEdgeMessagePassingV2,
)


def reference_aggregate_messages(messages, target_indices, num_nodes):
    """
    Reference implementation that matches the old Python-loop logic.

    messages:
        [batch_size, num_edges, hidden_dim]

    target_indices:
        [batch_size, num_edges]
    """

    batch_size, _, hidden_dim = messages.shape

    aggregated = messages.new_zeros(batch_size, num_nodes, hidden_dim)
    counts = messages.new_zeros(batch_size, num_nodes, 1)
    ones = messages.new_ones(batch_size, target_indices.shape[1], 1)

    for batch_idx in range(batch_size):
        aggregated[batch_idx].index_add_(
            dim=0,
            index=target_indices[batch_idx].long(),
            source=messages[batch_idx],
        )

        counts[batch_idx].index_add_(
            dim=0,
            index=target_indices[batch_idx].long(),
            source=ones[batch_idx],
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

    expected = reference_aggregate_messages(
        messages=messages,
        target_indices=target_indices,
        num_nodes=num_nodes,
    )

    actual = ResidualEdgeMessagePassingV2._aggregate_messages(
        messages=messages,
        target_indices=target_indices,
        num_nodes=num_nodes,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


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

    action_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)

    # Disable some branch actions.
    # Action 0 is stop/handoff and remains valid.
    action_mask[:, 3] = False
    action_mask[:, 7] = False

    policy_logits, value = model(
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        action_mask=action_mask,
    )

    assert policy_logits.shape == (batch_size, num_actions)
    assert value.shape == (batch_size,)

    assert torch.isfinite(value).all()

    # Invalid actions must be strongly masked.
    assert (policy_logits[:, 3] < -1e20).all()
    assert (policy_logits[:, 7] < -1e20).all()

    predicted_action = torch.argmax(policy_logits, dim=1)

    assert (predicted_action != 3).all()
    assert (predicted_action != 7).all()

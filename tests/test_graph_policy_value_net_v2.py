import pytest
import torch

from grid_topology_ai.model import (
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

    _, hidden_dim = messages.shape
    aggregated = messages.new_zeros(num_nodes, hidden_dim)
    counts = messages.new_zeros(num_nodes, 1)
    for edge_index, target in enumerate(target_indices.tolist()):
        if not bool(edge_active_mask[edge_index]):
            continue
        aggregated[target] += messages[edge_index]
        counts[target] += 1.0
    return aggregated / counts.clamp_min(1.0)


def test_vectorized_aggregate_matches_reference():
    torch.manual_seed(42)
    num_edges = 17
    num_nodes = 9
    hidden_dim = 13
    messages = torch.randn(num_edges, hidden_dim)
    target_indices = torch.randint(0, num_nodes, size=(num_edges,))
    edge_active_mask = torch.rand(num_edges) > 0.35
    edge_active_mask[0] = True
    edge_active_mask[1] = False

    expected = reference_aggregate_messages(
        messages,
        target_indices,
        num_nodes,
        edge_active_mask,
    )
    actual = ResidualEdgeMessagePassingV2._aggregate_messages(
        messages,
        target_indices,
        num_nodes,
        edge_active_mask,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_inactive_edge_does_not_dilute_degree_normalization():
    messages = torch.tensor(
        [[6.0, 12.0], [1000.0, 2000.0]],
        dtype=torch.float32,
    )
    target_indices = torch.tensor([1, 1], dtype=torch.long)
    edge_active_mask = torch.tensor([True, False], dtype=torch.bool)

    aggregated = ResidualEdgeMessagePassingV2._aggregate_messages(
        messages,
        target_indices,
        3,
        edge_active_mask,
    )
    expected = torch.tensor(
        [[0.0, 0.0], [6.0, 12.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(aggregated, expected)


def _packed_batch(
    *,
    batch_size: int,
    num_buses: int,
    num_branches: int,
    num_bus_features: int,
    num_branch_features: int,
):
    buses = torch.randn(batch_size * num_buses, num_bus_features)
    branches = torch.randn(batch_size * num_branches, num_branch_features)
    node_batch = torch.arange(batch_size).repeat_interleave(num_buses)
    edge_batch = torch.arange(batch_size).repeat_interleave(num_branches)

    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for graph_index in range(batch_size):
        offset = graph_index * num_buses
        sources.append(torch.randint(0, num_buses, size=(num_branches,)) + offset)
        targets.append(torch.randint(0, num_buses, size=(num_branches,)) + offset)
    edge_index = torch.stack(
        [torch.cat(sources), torch.cat(targets)],
        dim=0,
    )
    return buses, branches, edge_index, node_batch, edge_batch


def test_graph_policy_value_net_v2_forward_shapes_and_mask():
    torch.manual_seed(123)
    batch_size = 3
    num_buses = 8
    num_branches = 11
    num_bus_features = 4
    num_branch_features = 6

    model = GraphPolicyValueNetV2(
        num_bus_features=num_bus_features,
        num_branch_features=num_branch_features,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )
    bus_features, branch_features, edge_index, node_batch, edge_batch = _packed_batch(
        batch_size=batch_size,
        num_buses=num_buses,
        num_branches=num_branches,
        num_bus_features=num_bus_features,
        num_branch_features=num_branch_features,
    )
    edge_active_mask = torch.ones(batch_size * num_branches, dtype=torch.bool)
    action_mask = torch.ones(batch_size, num_branches + 1, dtype=torch.bool)
    action_mask[:, 3] = False
    action_mask[:, 7] = False

    policy_logits, value = model(
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        edge_active_mask=edge_active_mask,
        action_mask=action_mask,
        node_batch=node_batch,
        edge_batch=edge_batch,
    )

    assert policy_logits.shape == (batch_size, num_branches + 1)
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
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    bus_features = torch.randn(4, 4)
    branch_features = torch.randn(5, 6)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="edge_active_mask"):
        model(
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            edge_active_mask=torch.ones(4, dtype=torch.bool),
            action_mask=torch.ones(6, dtype=torch.bool),
        )


def test_policy_can_select_a_closeable_inactive_branch():
    model = GraphPolicyValueNetV2(
        num_bus_features=3,
        num_branch_features=4,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    bus_features = torch.randn(3, 3)
    branch_features = torch.randn(3, 4)
    edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    edge_active_mask = torch.tensor([True, False, True])
    action_mask = torch.tensor([True, True, True, True])

    policy_logits, value = model(
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        edge_active_mask=edge_active_mask,
        action_mask=action_mask,
    )

    assert torch.isfinite(policy_logits[0, 2])
    assert torch.isfinite(value).all()


def test_model_rejects_dense_batch_inputs():
    model = GraphPolicyValueNetV2(
        num_bus_features=3,
        num_branch_features=4,
        hidden_dim=16,
        num_layers=1,
    )
    with pytest.raises(ValueError, match="bus_features"):
        model(
            bus_features=torch.randn(2, 3, 3),
            branch_features=torch.randn(2, 3, 4),
            edge_index=torch.zeros(2, 2, 3, dtype=torch.long),
            edge_active_mask=torch.ones(2, 3, dtype=torch.bool),
        )


def test_model_rejects_one_based_edge_index():
    model = GraphPolicyValueNetV2(
        num_bus_features=3,
        num_branch_features=4,
        hidden_dim=16,
        num_layers=1,
    )
    with pytest.raises(ValueError, match="zero-based"):
        model(
            bus_features=torch.randn(3, 3),
            branch_features=torch.randn(3, 4),
            edge_index=torch.tensor([[1, 2, 3], [2, 3, 1]]),
            edge_active_mask=torch.ones(3, dtype=torch.bool),
        )

import torch

from grid_topology_ai.models.graph_policy_value_net_v2 import (
    GraphPolicyValueNetV2,
    ResidualEdgeMessagePassingV2,
)


class _SelectFeature(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = int(index)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., self.index : self.index + 1]


class _SliceProjection(torch.nn.Module):
    def __init__(self, start: int, width: int) -> None:
        super().__init__()
        self.start = int(start)
        self.width = int(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., self.start : self.start + self.width]


class _ZeroAttention(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(*values.shape[:-1], 1)


def _context_probe_model() -> GraphPolicyValueNetV2:
    """
    Build a deterministic V2 model whose policy/value heads expose edge mean.

    This isolates physical pooling semantics from random weight initialization.
    """

    hidden_dim = 2
    edge_mean_start = hidden_dim * 2
    branch_global_start = hidden_dim * 5

    model = GraphPolicyValueNetV2(
        num_bus_features=hidden_dim,
        num_branch_features=hidden_dim,
        num_actions=3,
        hidden_dim=hidden_dim,
        num_layers=0,
        dropout=0.0,
    )

    model.bus_encoder = torch.nn.Identity()
    model.branch_encoder = torch.nn.Identity()
    model.overload_attention = _ZeroAttention()
    model.global_projection = _SliceProjection(
        start=edge_mean_start,
        width=hidden_dim,
    )
    model.branch_policy_head = _SelectFeature(branch_global_start)
    model.stop_policy_head = _SelectFeature(edge_mean_start)
    model.value_head = _SelectFeature(edge_mean_start)
    model.eval()

    return model


def _context_probe_inputs() -> dict[str, torch.Tensor]:
    return {
        "bus_features": torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]],
        ),
        "branch_features": torch.tensor(
            [[[1.0, 0.0], [3.0, 0.0]]],
        ),
        "edge_index": torch.tensor(
            [[[0, 1], [1, 2]]],
            dtype=torch.long,
        ),
    }


def test_action_legality_does_not_change_value_context():
    model = _context_probe_model()
    inputs = _context_probe_inputs()
    edge_active_mask = torch.tensor([[True, True]])

    all_actions = torch.tensor([[True, True, True]])
    restricted_actions = torch.tensor([[True, True, False]])

    with torch.no_grad():
        all_logits, all_value = model(
            **inputs,
            edge_active_mask=edge_active_mask,
            action_mask=all_actions,
        )
        restricted_logits, restricted_value = model(
            **inputs,
            edge_active_mask=edge_active_mask,
            action_mask=restricted_actions,
        )

    torch.testing.assert_close(all_value, restricted_value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        all_logits[:, :2],
        restricted_logits[:, :2],
        rtol=0.0,
        atol=0.0,
    )
    assert restricted_logits[0, 2] == torch.finfo(restricted_logits.dtype).min


def test_active_non_switchable_bridge_changes_value():
    model = _context_probe_model()
    inputs = _context_probe_inputs()
    edge_active_mask = torch.tensor([[True, True]])

    # Both lines form a radial chain, so each is a physical bridge.
    # The second bridge remains active but is not a legal switching action.
    action_mask = torch.tensor([[True, True, False]])

    changed_branch_features = inputs["branch_features"].clone()
    changed_branch_features[0, 1, 0] = 9.0

    with torch.no_grad():
        original_logits, original_value = model(
            **inputs,
            edge_active_mask=edge_active_mask,
            action_mask=action_mask,
        )
        changed_logits, changed_value = model(
            bus_features=inputs["bus_features"],
            branch_features=changed_branch_features,
            edge_index=inputs["edge_index"],
            edge_active_mask=edge_active_mask,
            action_mask=action_mask,
        )

    torch.testing.assert_close(original_value, torch.tensor([2.0]))
    torch.testing.assert_close(changed_value, torch.tensor([5.0]))
    torch.testing.assert_close(original_logits[:, :2], torch.tensor([[2.0, 2.0]]))
    torch.testing.assert_close(changed_logits[:, :2], torch.tensor([[5.0, 5.0]]))


def test_inactive_edge_features_do_not_change_policy_or_value():
    model = _context_probe_model()
    inputs = _context_probe_inputs()
    edge_active_mask = torch.tensor([[True, False]])
    action_mask = torch.tensor([[True, True, False]])

    changed_branch_features = inputs["branch_features"].clone()
    changed_branch_features[0, 1] = torch.tensor([10000.0, -10000.0])

    with torch.no_grad():
        original_logits, original_value = model(
            **inputs,
            edge_active_mask=edge_active_mask,
            action_mask=action_mask,
        )
        changed_logits, changed_value = model(
            bus_features=inputs["bus_features"],
            branch_features=changed_branch_features,
            edge_index=inputs["edge_index"],
            edge_active_mask=edge_active_mask,
            action_mask=action_mask,
        )

    torch.testing.assert_close(
        original_logits,
        changed_logits,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        original_value,
        changed_value,
        rtol=0.0,
        atol=0.0,
    )


def test_all_inactive_edges_produce_finite_zero_edge_context():
    model = _context_probe_model()
    inputs = _context_probe_inputs()

    with torch.no_grad():
        policy_logits, value = model(
            **inputs,
            edge_active_mask=torch.tensor([[False, False]]),
            action_mask=torch.tensor([[True, False, False]]),
        )

    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()
    torch.testing.assert_close(policy_logits[:, 0], torch.tensor([0.0]))
    torch.testing.assert_close(value, torch.tensor([0.0]))


def test_masked_edge_matches_physically_removed_edge_in_message_passing():
    torch.manual_seed(2026)

    hidden_dim = 6
    layer = ResidualEdgeMessagePassingV2(
        hidden_dim=hidden_dim,
        dropout=0.0,
    )
    layer.eval()

    node_embeddings = torch.randn(1, 4, hidden_dim)
    edge_embeddings = torch.randn(1, 3, hidden_dim)
    edge_index = torch.tensor(
        [[[0, 1, 2], [1, 2, 3]]],
        dtype=torch.long,
    )
    edge_active_mask = torch.tensor([[True, False, True]])
    active_positions = torch.tensor([0, 2])

    with torch.no_grad():
        full_nodes, full_edges = layer(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            edge_index=edge_index,
            edge_active_mask=edge_active_mask,
        )
        pruned_nodes, pruned_edges = layer(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings[:, active_positions],
            edge_index=edge_index[:, :, active_positions],
            edge_active_mask=torch.ones(1, 2, dtype=torch.bool),
        )

    torch.testing.assert_close(
        full_nodes,
        pruned_nodes,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        full_edges[:, active_positions],
        pruned_edges,
        rtol=1e-5,
        atol=1e-6,
    )

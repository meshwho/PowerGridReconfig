from __future__ import annotations

import pytest
import torch

from grid_topology_ai.training import graph_policy_value as training


def test_segment_softmax_handles_amp_promoted_exp(monkeypatch) -> None:
    scores = torch.tensor(
        [1.0, 2.0, -1.0, 3.0],
        dtype=torch.float16,
    )
    batch_index = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    original_exp = torch.exp

    def promoted_exp(values: torch.Tensor) -> torch.Tensor:
        return original_exp(values.float())

    monkeypatch.setattr(torch, "exp", promoted_exp)

    weights = training.GraphPolicyValueNetV2._segment_softmax(
        scores,
        batch_index,
        batch_size=2,
    )

    assert weights.dtype == scores.dtype
    assert torch.isfinite(weights).all()

    sums = torch.zeros(2, dtype=torch.float32)
    sums.index_add_(0, batch_index, weights.float())
    torch.testing.assert_close(
        sums,
        torch.ones_like(sums),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the autocast regression test.",
)
def test_graph_v2_forward_and_backward_with_cuda_amp() -> None:
    device = torch.device("cuda")
    model = training.GraphPolicyValueNetV2(
        num_bus_features=3,
        num_branch_features=2,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    ).to(device)

    bus_features = torch.randn(
        2,
        4,
        3,
        device=device,
    )
    branch_features = torch.randn(
        2,
        3,
        2,
        device=device,
    )
    edge_index = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3]],
            [[0, 1, 2], [1, 2, 3]],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_active_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ],
        device=device,
    )
    action_mask = torch.ones(
        2,
        4,
        dtype=torch.bool,
        device=device,
    )

    with torch.amp.autocast("cuda", dtype=torch.float16):
        policy_logits, value = model(
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            edge_active_mask=edge_active_mask,
            action_mask=action_mask,
        )
        loss = policy_logits.float().mean() + value.float().mean()

    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()

    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

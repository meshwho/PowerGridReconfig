from __future__ import annotations

import torch
from torch import nn


class SimplePolicyValueNet(nn.Module):
    """
    Simple AlphaZero-like policy-value network.

    This is not the final model.

    Final version:
        GNN/GAT encoder over buses and branches.

    Current version:
        MLP over aggregated state features.

    Outputs:
        policy_logits over actions
        value in [-1, 1]
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.policy_head = nn.Linear(hidden_dim, num_actions)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        state_vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(state_vector)

        policy_logits = self.policy_head(features)

        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)

        value = self.value_head(features).squeeze(-1)

        return policy_logits, value
from __future__ import annotations

import torch
from torch import nn


class MLPBlock(nn.Module):
    """
    Small MLP block with LayerNorm and optional Dropout.

    This file intentionally avoids torch_geometric / torch_scatter.
    It is pure PyTorch and should work on Windows, CPU, CUDA and AMP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        final_activation: bool = True,
    ):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        if final_activation:
            layers.extend(
                [
                    nn.ReLU(),
                    nn.LayerNorm(output_dim),
                ]
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualEdgeMessagePassingV2(nn.Module):
    """
    Edge-aware residual message passing layer.

    Compared with the old layer, this version:
    - uses gated residual updates;
    - updates both node and edge embeddings;
    - keeps edge embeddings central, because actions are branch actions.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        message_input_dim = hidden_dim * 3

        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.message_gate = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _gather_nodes(
        node_embeddings: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, hidden_dim = node_embeddings.shape

        expanded_indices = indices.unsqueeze(-1).expand(
            batch_size,
            indices.shape[1],
            hidden_dim,
        )

        return torch.gather(
            node_embeddings,
            dim=1,
            index=expanded_indices,
        )

    @staticmethod
    def _aggregate_messages(
            messages: torch.Tensor,
            target_indices: torch.Tensor,
            num_nodes: int,
            edge_active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate only physically active edge messages into node embeddings.

        Parameters
        ----------
        messages:
            Shape [batch_size, num_edges, hidden_dim].

        target_indices:
            Shape [batch_size, num_edges].

        edge_active_mask:
            Physical branch activity mask.
            Shape [batch_size, num_edges].
            True means that the branch physically exists in the current topology.

        Returns
        -------
        torch.Tensor
            Shape [batch_size, num_nodes, hidden_dim].
        """

        batch_size, num_edges, hidden_dim = messages.shape

        if edge_active_mask.shape != (batch_size, num_edges):
            raise ValueError(
                "edge_active_mask must have shape "
                f"({batch_size}, {num_edges}), "
                f"got {tuple(edge_active_mask.shape)}"
            )

        index = target_indices.long().unsqueeze(-1).expand(
            batch_size,
            num_edges,
            hidden_dim,
        )

        active = edge_active_mask.to(dtype=messages.dtype).unsqueeze(-1)

        aggregated = messages.new_zeros(
            batch_size,
            num_nodes,
            hidden_dim,
        )

        aggregated.scatter_add_(
            dim=1,
            index=index,
            src=messages * active,
        )

        count_index = target_indices.long().unsqueeze(-1)

        counts = messages.new_zeros(
            batch_size,
            num_nodes,
            1,
        )

        counts.scatter_add_(
            dim=1,
            index=count_index,
            src=active,
        )

        return aggregated / counts.clamp_min(1.0)

    def _directional_messages(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        source_node = self._gather_nodes(
            node_embeddings=node_embeddings,
            indices=source_indices,
        )

        target_node = self._gather_nodes(
            node_embeddings=node_embeddings,
            indices=target_indices,
        )

        message_input = torch.cat(
            [
                source_node,
                target_node,
                edge_embeddings,
            ],
            dim=-1,
        )

        raw_message = self.message_mlp(message_input)
        gate = self.message_gate(message_input)

        return raw_message * gate

    def forward(
            self,
            node_embeddings: torch.Tensor,
            edge_embeddings: torch.Tensor,
            edge_index: torch.Tensor,
            edge_active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = node_embeddings.shape

        source = edge_index[:, 0, :].long()
        target = edge_index[:, 1, :].long()

        forward_messages = self._directional_messages(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            source_indices=source,
            target_indices=target,
        )

        reverse_messages = self._directional_messages(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            source_indices=target,
            target_indices=source,
        )

        forward_aggregated = self._aggregate_messages(
            messages=forward_messages,
            target_indices=target,
            num_nodes=num_nodes,
            edge_active_mask=edge_active_mask,
        )

        reverse_aggregated = self._aggregate_messages(
            messages=reverse_messages,
            target_indices=source,
            num_nodes=num_nodes,
            edge_active_mask=edge_active_mask,
        )

        aggregated = 0.5 * (forward_aggregated + reverse_aggregated)

        node_input = torch.cat(
            [
                node_embeddings,
                aggregated,
            ],
            dim=-1,
        )

        node_delta = self.node_update(node_input)
        node_gate = self.node_gate(node_input)

        new_node_embeddings = self.node_norm(
            node_embeddings + node_gate * node_delta
        )

        source_node = self._gather_nodes(
            node_embeddings=new_node_embeddings,
            indices=source,
        )

        target_node = self._gather_nodes(
            node_embeddings=new_node_embeddings,
            indices=target,
        )

        edge_input = torch.cat(
            [
                source_node,
                target_node,
                torch.abs(source_node - target_node),
                edge_embeddings,
            ],
            dim=-1,
        )

        edge_delta = self.edge_update(edge_input)
        edge_gate = self.edge_gate(edge_input)

        new_edge_embeddings = self.edge_norm(
            edge_embeddings + edge_gate * edge_delta
        )

        return new_node_embeddings, new_edge_embeddings


class GraphPolicyValueNetV2(nn.Module):
    """
    Edge-centric graph policy-value network V2.

    The action space is the same as V1:
        action 0      - handoff / stop;
        action k > 0  - switch off branch with branch_pos = k - 1.

    Forward output is also the same:
        policy_logits, value
    """

    def __init__(
        self,
        num_bus_features: int,
        num_branch_features: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.model_type = "graph_policy_value_net_v2"

        self.num_bus_features = int(num_bus_features)
        self.num_branch_features = int(num_branch_features)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        if self.num_actions < 2:
            raise ValueError(
                "num_actions must be at least 2: action 0 + branch actions."
            )

        self.num_branch_actions = self.num_actions - 1

        self.bus_encoder = nn.Sequential(
            nn.LayerNorm(num_bus_features),
            nn.Linear(num_bus_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.branch_encoder = nn.Sequential(
            nn.LayerNorm(num_branch_features),
            nn.Linear(num_branch_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.layers = nn.ModuleList(
            [
                ResidualEdgeMessagePassingV2(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Learned overload / severity attention.
        # This does not rely on hard-coded feature indices.
        self.overload_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        # Global context:
        # node_mean, node_max, edge_mean, edge_max, overload_pool.
        self.global_dim = hidden_dim * 5

        self.global_projection = nn.Sequential(
            nn.Linear(self.global_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Branch policy receives:
        # source, target, edge, |source-target|, source*target, global, overload_pool.
        branch_head_input_dim = hidden_dim * 7

        self.branch_policy_head = nn.Sequential(
            nn.Linear(branch_head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

        self.stop_policy_head = nn.Sequential(
            nn.Linear(self.global_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.global_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _ensure_batched_features(
        tensor: torch.Tensor,
        expected_rank_without_batch: int,
        name: str,
    ) -> torch.Tensor:
        if tensor.dim() == expected_rank_without_batch:
            return tensor.unsqueeze(0)

        if tensor.dim() == expected_rank_without_batch + 1:
            return tensor

        raise ValueError(
            f"{name} has invalid shape {tuple(tensor.shape)}. "
            f"Expected rank {expected_rank_without_batch} or "
            f"{expected_rank_without_batch + 1}."
        )

    @staticmethod
    def _normalize_edge_index(
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        edge_index = edge_index.long()

        min_index = int(edge_index.min().detach().cpu().item())
        max_index = int(edge_index.max().detach().cpu().item())

        if min_index >= 0 and max_index < num_nodes:
            return edge_index

        if min_index >= 1 and max_index <= num_nodes:
            return edge_index - 1

        raise ValueError(
            "edge_index contains bus indices that cannot be mapped to "
            f"0..{num_nodes - 1}. "
            f"min={min_index}, max={max_index}, num_nodes={num_nodes}"
        )

    @staticmethod
    def _gather_nodes(
        node_embeddings: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, hidden_dim = node_embeddings.shape

        expanded_indices = indices.unsqueeze(-1).expand(
            batch_size,
            indices.shape[1],
            hidden_dim,
        )

        return torch.gather(
            node_embeddings,
            dim=1,
            index=expanded_indices,
        )

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor | None,
        dim: int,
    ) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=dim)

        mask_float = mask.to(dtype=values.dtype).unsqueeze(-1)
        summed = (values * mask_float).sum(dim=dim)
        count = mask_float.sum(dim=dim).clamp_min(1.0)

        return summed / count

    @staticmethod
    def _masked_max(
        values: torch.Tensor,
        mask: torch.Tensor | None,
        dim: int,
    ) -> torch.Tensor:
        if mask is None:
            return values.max(dim=dim).values

        mask_value = torch.finfo(values.dtype).min

        masked_values = values.masked_fill(
            ~mask.unsqueeze(-1),
            mask_value,
        )

        out = masked_values.max(dim=dim).values

        no_valid = ~mask.any(dim=dim)

        if bool(no_valid.any()):
            out = out.clone()
            out[no_valid] = 0.0

        return out

    def _overload_focused_pool(
        self,
        edge_embeddings: torch.Tensor,
        branch_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Learned severity pooling over branches.

        It lets the model focus on the most operationally important branches
        without relying on fixed feature indices.
        """

        scores = self.overload_attention(edge_embeddings).squeeze(-1)

        if branch_mask is not None:
            scores = scores.masked_fill(
                ~branch_mask,
                torch.finfo(scores.dtype).min,
            )

            no_valid = ~branch_mask.any(dim=1)

            if bool(no_valid.any()):
                scores = scores.clone()
                scores[no_valid] = 0.0

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        if branch_mask is not None:
            weights = weights * branch_mask.to(dtype=edge_embeddings.dtype).unsqueeze(-1)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return (edge_embeddings * weights).sum(dim=1)

    def _build_contexts(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action_mask is None:
            branch_mask = None
        else:
            branch_mask = action_mask[:, 1:].bool()

        node_mean = node_embeddings.mean(dim=1)
        node_max = node_embeddings.max(dim=1).values

        edge_mean = self._masked_mean(
            values=edge_embeddings,
            mask=branch_mask,
            dim=1,
        )

        edge_max = self._masked_max(
            values=edge_embeddings,
            mask=branch_mask,
            dim=1,
        )

        overload_pool = self._overload_focused_pool(
            edge_embeddings=edge_embeddings,
            branch_mask=branch_mask,
        )

        global_embedding = torch.cat(
            [
                node_mean,
                node_max,
                edge_mean,
                edge_max,
                overload_pool,
            ],
            dim=-1,
        )

        global_projected = self.global_projection(global_embedding)

        return global_embedding, global_projected, overload_pool

    def forward(
        self,
        bus_features: torch.Tensor,
        branch_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_active_mask: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bus_features = self._ensure_batched_features(
            tensor=bus_features,
            expected_rank_without_batch=2,
            name="bus_features",
        )

        branch_features = self._ensure_batched_features(
            tensor=branch_features,
            expected_rank_without_batch=2,
            name="branch_features",
        )

        edge_index = self._ensure_batched_features(
            tensor=edge_index,
            expected_rank_without_batch=2,
            name="edge_index",
        )

        edge_active_mask = self._ensure_batched_features(
            tensor=edge_active_mask,
            expected_rank_without_batch=1,
            name="edge_active_mask",
        ).bool()

        if action_mask is not None:
            action_mask = self._ensure_batched_features(
                tensor=action_mask,
                expected_rank_without_batch=1,
                name="action_mask",
            ).bool()

        batch_size, num_nodes, _ = bus_features.shape
        _, num_edges, _ = branch_features.shape

        if edge_active_mask.shape[0] != batch_size:
            raise ValueError(
                "edge_active_mask batch dimension does not match "
                f"bus_features: {edge_active_mask.shape[0]} != {batch_size}"
            )

        if edge_index.shape[1] != 2:
            raise ValueError(
                f"edge_index must have shape [batch, 2, num_edges], "
                f"got {tuple(edge_index.shape)}"
            )

        if edge_index.shape[2] != num_edges:
            raise ValueError(
                f"edge_index num_edges={edge_index.shape[2]} does not match "
                f"branch_features num_edges={num_edges}"
            )

        if edge_active_mask.shape[1] != num_edges:
            raise ValueError(
                f"edge_active_mask has {edge_active_mask.shape[1]} edges, "
                f"but branch_features has {num_edges}."
            )

        if num_edges != self.num_branch_actions:
            raise ValueError(
                f"Model expects {self.num_branch_actions} branch actions, "
                f"but input has {num_edges} branches."
            )

        if action_mask is not None and action_mask.shape[1] != self.num_actions:
            raise ValueError(
                f"action_mask has {action_mask.shape[1]} actions, "
                f"but model expects {self.num_actions}."
            )

        if action_mask is not None:
            invalid_physical_actions = (
                action_mask[:, 1:] & ~edge_active_mask
            )

            if bool(invalid_physical_actions.any()):
                raise ValueError(
                    "action_mask marks a physically inactive branch action "
                    "as valid."
                )

        edge_index = self._normalize_edge_index(
            edge_index=edge_index,
            num_nodes=num_nodes,
        )

        node_embeddings = self.bus_encoder(bus_features)
        edge_embeddings = self.branch_encoder(branch_features)

        for layer in self.layers:
            node_embeddings, edge_embeddings = layer(
                node_embeddings=node_embeddings,
                edge_embeddings=edge_embeddings,
                edge_index=edge_index,
                edge_active_mask=edge_active_mask,
            )

        source = edge_index[:, 0, :]
        target = edge_index[:, 1, :]

        source_node = self._gather_nodes(
            node_embeddings=node_embeddings,
            indices=source,
        )

        target_node = self._gather_nodes(
            node_embeddings=node_embeddings,
            indices=target,
        )

        global_embedding, global_projected, overload_pool = self._build_contexts(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            action_mask=action_mask,
        )

        global_repeated = global_projected.unsqueeze(1).expand(
            batch_size,
            num_edges,
            self.hidden_dim,
        )

        overload_repeated = overload_pool.unsqueeze(1).expand(
            batch_size,
            num_edges,
            self.hidden_dim,
        )

        branch_repr = torch.cat(
            [
                source_node,
                target_node,
                edge_embeddings,
                torch.abs(source_node - target_node),
                source_node * target_node,
                global_repeated,
                overload_repeated,
            ],
            dim=-1,
        )

        branch_logits = self.branch_policy_head(branch_repr).squeeze(-1)

        stop_value_input = torch.cat(
            [
                global_embedding,
                global_projected,
            ],
            dim=-1,
        )

        stop_logit = self.stop_policy_head(stop_value_input)

        policy_logits = torch.cat(
            [
                stop_logit,
                branch_logits,
            ],
            dim=1,
        )

        if action_mask is not None:
            mask_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(~action_mask, mask_value)

        value = self.value_head(stop_value_input).squeeze(-1)

        return policy_logits, value
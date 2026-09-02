from __future__ import annotations

import torch
from torch import nn


class MLPBlock(nn.Module):
    """Small MLP block with LayerNorm and optional dropout."""

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
            layers.extend([nn.ReLU(), nn.LayerNorm(output_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualEdgeMessagePassingV2(nn.Module):
    """Edge-aware residual message passing for packed graph tensors."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
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
    def _aggregate_messages(
        messages: torch.Tensor,
        target_indices: torch.Tensor,
        num_nodes: int,
        edge_active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if messages.ndim != 2:
            raise ValueError("messages must have shape [num_edges, hidden_dim].")
        num_edges, hidden_dim = messages.shape
        if target_indices.shape != (num_edges,):
            raise ValueError(
                f"target_indices must have shape {(num_edges,)}, "
                f"got {tuple(target_indices.shape)}."
            )
        if edge_active_mask.shape != (num_edges,):
            raise ValueError(
                f"edge_active_mask must have shape {(num_edges,)}, "
                f"got {tuple(edge_active_mask.shape)}."
            )

        active = edge_active_mask.to(dtype=messages.dtype).unsqueeze(-1)
        targets = target_indices.long()
        aggregated = messages.new_zeros(num_nodes, hidden_dim)
        aggregated.index_add_(0, targets, messages * active)
        counts = messages.new_zeros(num_nodes, 1)
        counts.index_add_(0, targets, active)
        return aggregated / counts.clamp_min(1.0)

    def _directional_messages(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        source_node = node_embeddings[source_indices.long()]
        target_node = node_embeddings[target_indices.long()]
        message_input = torch.cat(
            [source_node, target_node, edge_embeddings],
            dim=-1,
        )
        return self.message_mlp(message_input) * self.message_gate(message_input)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_embeddings.ndim != 2:
            raise ValueError("node_embeddings must have shape [num_nodes, hidden_dim].")
        if edge_embeddings.ndim != 2:
            raise ValueError("edge_embeddings must have shape [num_edges, hidden_dim].")
        num_nodes = int(node_embeddings.shape[0])
        num_edges = int(edge_embeddings.shape[0])
        if edge_index.shape != (2, num_edges):
            raise ValueError("edge_index must have shape (2, num_edges).")
        if edge_active_mask.shape != (num_edges,):
            raise ValueError("edge_active_mask must have shape (num_edges,).")

        source = edge_index[0].long()
        target = edge_index[1].long()
        forward_messages = self._directional_messages(
            node_embeddings,
            edge_embeddings,
            source,
            target,
        )
        reverse_messages = self._directional_messages(
            node_embeddings,
            edge_embeddings,
            target,
            source,
        )
        forward_aggregated = self._aggregate_messages(
            forward_messages,
            target,
            num_nodes,
            edge_active_mask,
        )
        reverse_aggregated = self._aggregate_messages(
            reverse_messages,
            source,
            num_nodes,
            edge_active_mask,
        )
        aggregated = 0.5 * (forward_aggregated + reverse_aggregated)

        node_input = torch.cat([node_embeddings, aggregated], dim=-1)
        node_delta = self.node_update(node_input)
        node_gate = self.node_gate(node_input)
        new_node_embeddings = self.node_norm(
            node_embeddings + node_gate * node_delta
        )

        source_node = new_node_embeddings[source]
        target_node = new_node_embeddings[target]
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
    """Edge-centric graph policy-value network V2 for packed graph tensors."""

    def __init__(
        self,
        num_bus_features: int,
        num_branch_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_type = "graph_policy_value_net_v2"
        self.num_bus_features = int(num_bus_features)
        self.num_branch_features = int(num_branch_features)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        if self.num_bus_features <= 0:
            raise ValueError("num_bus_features must be positive.")
        if self.num_branch_features <= 0:
            raise ValueError("num_branch_features must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

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
        self.overload_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

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
    def _validate_grouped_batch_vector(
        batch_index: torch.Tensor,
        *,
        item_count: int,
        name: str,
    ) -> tuple[int, torch.Tensor]:
        if batch_index.ndim != 1 or batch_index.shape != (item_count,):
            raise ValueError(
                f"{name} must contain one graph ID per item; "
                f"expected {(item_count,)}, got {tuple(batch_index.shape)}."
            )
        if item_count <= 0:
            raise ValueError(f"{name} must not be empty.")

        batch_index = batch_index.long()
        minimum = int(batch_index.min().item())
        maximum = int(batch_index.max().item())
        if minimum != 0:
            raise ValueError(f"{name} graph IDs must start from 0.")
        batch_size = maximum + 1
        counts = torch.bincount(batch_index, minlength=batch_size)
        expected = torch.arange(
            batch_size,
            device=batch_index.device,
            dtype=torch.long,
        ).repeat_interleave(counts)
        if not torch.equal(batch_index, expected):
            raise ValueError(f"{name} must group items contiguously by graph.")
        return batch_size, counts

    def _prepare_packed_inputs(
        self,
        *,
        bus_features: torch.Tensor,
        branch_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_active_mask: torch.Tensor,
        action_mask: torch.Tensor | None,
        node_batch: torch.Tensor | None,
        edge_batch: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        int,
        torch.Tensor,
    ]:
        if bus_features.ndim != 2:
            raise ValueError(
                "bus_features must have shape [total_nodes, num_bus_features]."
            )
        if branch_features.ndim != 2:
            raise ValueError(
                "branch_features must have shape [total_edges, num_branch_features]."
            )
        if bus_features.shape[1] != self.num_bus_features:
            raise ValueError(
                f"bus_features width mismatch: expected {self.num_bus_features}, "
                f"got {bus_features.shape[1]}."
            )
        if branch_features.shape[1] != self.num_branch_features:
            raise ValueError(
                f"branch_features width mismatch: expected {self.num_branch_features}, "
                f"got {branch_features.shape[1]}."
            )

        num_nodes = int(bus_features.shape[0])
        num_edges = int(branch_features.shape[0])
        if num_nodes <= 0:
            raise ValueError("Graph batch must contain at least one node.")
        if num_edges <= 0:
            raise ValueError("Graph batch must contain at least one edge.")
        if edge_index.shape != (2, num_edges):
            raise ValueError(
                f"edge_index must have shape {(2, num_edges)}, "
                f"got {tuple(edge_index.shape)}."
            )
        if edge_active_mask.shape != (num_edges,):
            raise ValueError(
                f"edge_active_mask must have shape {(num_edges,)}, "
                f"got {tuple(edge_active_mask.shape)}."
            )

        edge_index = edge_index.long()
        edge_active_mask = edge_active_mask.bool()
        if node_batch is None and edge_batch is None:
            node_batch = torch.zeros(
                num_nodes,
                dtype=torch.long,
                device=bus_features.device,
            )
            edge_batch = torch.zeros(
                num_edges,
                dtype=torch.long,
                device=branch_features.device,
            )
        elif node_batch is None or edge_batch is None:
            raise ValueError("Packed batches require both node_batch and edge_batch.")

        node_batch = node_batch.long()
        edge_batch = edge_batch.long()
        node_batch_size, node_counts = self._validate_grouped_batch_vector(
            node_batch,
            item_count=num_nodes,
            name="node_batch",
        )
        edge_batch_size, edge_counts = self._validate_grouped_batch_vector(
            edge_batch,
            item_count=num_edges,
            name="edge_batch",
        )
        if node_batch_size != edge_batch_size:
            raise ValueError(
                "node_batch and edge_batch describe different graph counts."
            )
        batch_size = node_batch_size
        if bool((node_counts <= 0).any()):
            raise ValueError("Every graph must contain at least one node.")
        if bool((edge_counts <= 0).any()):
            raise ValueError("Every graph must contain at least one edge.")

        minimum_node_index = int(edge_index.min().item())
        maximum_node_index = int(edge_index.max().item())
        if minimum_node_index < 0 or maximum_node_index >= num_nodes:
            raise ValueError(
                "edge_index must use the zero-based packed node index space."
            )
        source = edge_index[0]
        target = edge_index[1]
        if not torch.equal(node_batch[source], edge_batch):
            raise ValueError(
                "An edge source belongs to a different graph than the edge itself."
            )
        if not torch.equal(node_batch[target], edge_batch):
            raise ValueError(
                "An edge target belongs to a different graph than the edge itself."
            )

        max_edges = int(edge_counts.max().item())
        if action_mask is not None:
            if action_mask.ndim == 1:
                if batch_size != 1:
                    raise ValueError(
                        "A 1D action_mask is valid only for a single graph."
                    )
                action_mask = action_mask.unsqueeze(0)
            action_mask = action_mask.bool()
            expected_shape = (batch_size, max_edges + 1)
            if action_mask.shape != expected_shape:
                raise ValueError(
                    f"action_mask must have shape {expected_shape}, "
                    f"got {tuple(action_mask.shape)}."
                )
            if not bool(action_mask.any(dim=1).all()):
                raise ValueError("Every graph must contain at least one legal action.")
            for graph_index, edge_count in enumerate(edge_counts.tolist()):
                if bool(action_mask[graph_index, int(edge_count) + 1 :].any()):
                    raise ValueError(
                        f"action_mask padding must be false for graph {graph_index}."
                    )

        return (
            bus_features,
            branch_features,
            edge_index,
            edge_active_mask,
            node_batch,
            edge_batch,
            action_mask,
            batch_size,
            edge_counts,
        )

    @staticmethod
    def _segment_mean(
        values: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if values.ndim != 2 or batch_index.shape != (values.shape[0],):
            raise ValueError("Invalid tensors for segmented mean.")
        result = values.new_zeros(batch_size, values.shape[1])
        result.index_add_(0, batch_index.long(), values)
        counts = values.new_zeros(batch_size, 1)
        counts.index_add_(
            0,
            batch_index.long(),
            values.new_ones(values.shape[0], 1),
        )
        return result / counts.clamp_min(1.0)

    @staticmethod
    def _segment_max(
        values: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if values.ndim != 2 or batch_index.shape != (values.shape[0],):
            raise ValueError("Invalid tensors for segmented max.")
        result = values.new_full(
            (batch_size, values.shape[1]),
            torch.finfo(values.dtype).min,
        )
        expanded_index = batch_index.long().unsqueeze(-1).expand_as(values)
        result.scatter_reduce_(
            dim=0,
            index=expanded_index,
            src=values,
            reduce="amax",
            include_self=True,
        )
        counts = torch.bincount(batch_index.long(), minlength=batch_size)
        empty_graphs = counts == 0
        if bool(empty_graphs.any()):
            result = result.clone()
            result[empty_graphs] = 0.0
        return result

    @staticmethod
    def _segment_softmax(
        scores: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if scores.ndim != 1 or batch_index.shape != scores.shape:
            raise ValueError("Invalid tensors for segmented softmax.")
        work_dtype = (
            torch.float32
            if scores.dtype in (torch.float16, torch.bfloat16)
            else scores.dtype
        )
        work_scores = scores.to(dtype=work_dtype)
        index = batch_index.long()
        maxima = work_scores.new_full(
            (batch_size,),
            torch.finfo(work_scores.dtype).min,
        )
        maxima.scatter_reduce_(
            dim=0,
            index=index,
            src=work_scores,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(work_scores - maxima[index])
        denominators = exponentials.new_zeros(batch_size)
        denominators.index_add_(0, index, exponentials)
        return (
            exponentials / denominators[index].clamp_min(1e-12)
        ).to(dtype=scores.dtype)

    @classmethod
    def _segment_active_mean(
        cls,
        values: torch.Tensor,
        batch_index: torch.Tensor,
        active_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if active_mask.shape != (values.shape[0],):
            raise ValueError("active_mask must match the number of values.")
        active_values = values[active_mask]
        active_batch = batch_index[active_mask]
        if active_values.shape[0] == 0:
            return values.new_zeros(batch_size, values.shape[1])
        return cls._segment_mean(active_values, active_batch, batch_size)

    @classmethod
    def _segment_active_max(
        cls,
        values: torch.Tensor,
        batch_index: torch.Tensor,
        active_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if active_mask.shape != (values.shape[0],):
            raise ValueError("active_mask must match the number of values.")
        active_values = values[active_mask]
        active_batch = batch_index[active_mask]
        if active_values.shape[0] == 0:
            return values.new_zeros(batch_size, values.shape[1])
        return cls._segment_max(active_values, active_batch, batch_size)

    def _overload_focused_pool(
        self,
        *,
        edge_embeddings: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_active_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        scores = self.overload_attention(edge_embeddings).squeeze(-1)
        mask_value = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~edge_active_mask, mask_value)
        weights = self._segment_softmax(masked_scores, edge_batch, batch_size)
        weights = weights * edge_active_mask.to(dtype=weights.dtype)
        weighted_edges = edge_embeddings * weights.unsqueeze(-1)
        overload_pool = edge_embeddings.new_zeros(batch_size, self.hidden_dim)
        overload_pool.index_add_(0, edge_batch, weighted_edges)
        return overload_pool

    def _build_contexts(
        self,
        *,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        node_batch: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_active_mask: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_mean = self._segment_mean(node_embeddings, node_batch, batch_size)
        node_max = self._segment_max(node_embeddings, node_batch, batch_size)
        edge_mean = self._segment_active_mean(
            edge_embeddings,
            edge_batch,
            edge_active_mask,
            batch_size,
        )
        edge_max = self._segment_active_max(
            edge_embeddings,
            edge_batch,
            edge_active_mask,
            batch_size,
        )
        overload_pool = self._overload_focused_pool(
            edge_embeddings=edge_embeddings,
            edge_batch=edge_batch,
            edge_active_mask=edge_active_mask,
            batch_size=batch_size,
        )
        global_embedding = torch.cat(
            [node_mean, node_max, edge_mean, edge_max, overload_pool],
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
        node_batch: torch.Tensor | None = None,
        edge_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one graph or a packed variable-size graph batch."""

        (
            bus_features,
            branch_features,
            edge_index,
            edge_active_mask,
            node_batch,
            edge_batch,
            action_mask,
            batch_size,
            edge_counts,
        ) = self._prepare_packed_inputs(
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            edge_active_mask=edge_active_mask,
            action_mask=action_mask,
            node_batch=node_batch,
            edge_batch=edge_batch,
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

        source = edge_index[0].long()
        target = edge_index[1].long()
        source_node = node_embeddings[source]
        target_node = node_embeddings[target]
        global_embedding, global_projected, overload_pool = self._build_contexts(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            node_batch=node_batch,
            edge_batch=edge_batch,
            edge_active_mask=edge_active_mask,
            batch_size=batch_size,
        )
        global_repeated = global_projected[edge_batch]
        overload_repeated = overload_pool[edge_batch]
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
        branch_logits_flat = self.branch_policy_head(branch_repr).squeeze(-1)

        max_edges = int(edge_counts.max().item())
        edge_offsets = torch.cumsum(
            torch.cat([edge_counts.new_zeros(1), edge_counts[:-1]]),
            dim=0,
        )
        edge_positions = (
            torch.arange(
                branch_logits_flat.shape[0],
                device=edge_batch.device,
                dtype=torch.long,
            )
            - edge_offsets[edge_batch]
        )
        if bool((edge_positions < 0).any()) or bool(
            (edge_positions >= edge_counts[edge_batch]).any()
        ):
            raise RuntimeError("Computed an invalid local edge position.")

        mask_value = torch.finfo(branch_logits_flat.dtype).min
        branch_logits = branch_logits_flat.new_full(
            (batch_size, max_edges),
            mask_value,
        )
        branch_logits[edge_batch, edge_positions] = branch_logits_flat

        stop_value_input = torch.cat(
            [global_embedding, global_projected],
            dim=-1,
        )
        stop_logit = self.stop_policy_head(stop_value_input)
        policy_logits = torch.cat([stop_logit, branch_logits], dim=1)
        if action_mask is not None:
            if action_mask.shape != policy_logits.shape:
                raise ValueError(
                    "action_mask shape does not match dynamic policy output. "
                    f"Expected {tuple(policy_logits.shape)}, "
                    f"got {tuple(action_mask.shape)}."
                )
            policy_logits = policy_logits.masked_fill(~action_mask, mask_value)

        value = self.value_head(stop_value_input).squeeze(-1)
        return policy_logits, value

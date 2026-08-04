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
    def _gather_dense_nodes(
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
    def _gather_nodes(
        node_embeddings: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if node_embeddings.ndim == 2:
            return node_embeddings[
                indices.long()
            ]

        if node_embeddings.ndim == 3:
            return (
                ResidualEdgeMessagePassingV2
                ._gather_dense_nodes(
                    node_embeddings,
                    indices,
                )
            )

        raise ValueError(
            "node_embeddings must have shape "
            "[num_nodes, hidden_dim] or "
            "[batch, num_nodes, hidden_dim]."
        )

    @staticmethod
    def _aggregate_dense_messages(
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

    @staticmethod
    def _aggregate_packed_messages(
        messages: torch.Tensor,
        target_indices: torch.Tensor,
        num_nodes: int,
        edge_active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate active edge messages for one disconnected packed batch.

        All graphs share one node index space. edge_index has already been
        shifted by collate_graph_samples, so ordinary index_add_ is sufficient.
        """

        if messages.ndim != 2:
            raise ValueError(
                "Packed messages must have shape "
                "[num_edges, hidden_dim]."
            )

        num_edges, hidden_dim = (
            messages.shape
        )

        if target_indices.shape != (
            num_edges,
        ):
            raise ValueError(
                "Packed target_indices must have "
                f"shape {(num_edges,)}, got "
                f"{tuple(target_indices.shape)}."
            )

        if edge_active_mask.shape != (
            num_edges,
        ):
            raise ValueError(
                "Packed edge_active_mask must have "
                f"shape {(num_edges,)}, got "
                f"{tuple(edge_active_mask.shape)}."
            )

        active = edge_active_mask.to(
            dtype=messages.dtype
        ).unsqueeze(-1)

        aggregated = messages.new_zeros(
            num_nodes,
            hidden_dim,
        )

        aggregated.index_add_(
            0,
            target_indices.long(),
            messages * active,
        )

        counts = messages.new_zeros(
            num_nodes,
            1,
        )

        counts.index_add_(
            0,
            target_indices.long(),
            active,
        )

        return (
            aggregated
            / counts.clamp_min(1.0)
        )

    @staticmethod
    def _aggregate_messages(
        messages: torch.Tensor,
        target_indices: torch.Tensor,
        num_nodes: int,
        edge_active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if messages.ndim == 2:
            return (
                ResidualEdgeMessagePassingV2
                ._aggregate_packed_messages(
                    messages=messages,
                    target_indices=target_indices,
                    num_nodes=num_nodes,
                    edge_active_mask=(
                        edge_active_mask
                    ),
                )
            )

        if messages.ndim == 3:
            return (
                ResidualEdgeMessagePassingV2
                ._aggregate_dense_messages(
                    messages=messages,
                    target_indices=target_indices,
                    num_nodes=num_nodes,
                    edge_active_mask=(
                        edge_active_mask
                    ),
                )
            )

        raise ValueError(
            "messages must have shape "
            "[num_edges, hidden_dim] or "
            "[batch, num_edges, hidden_dim]."
        )

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
        if node_embeddings.ndim == 2:
            num_nodes = int(
                node_embeddings.shape[0]
            )

            if edge_embeddings.ndim != 2:
                raise ValueError(
                    "Packed edge_embeddings must "
                    "be 2D."
                )

            if edge_index.shape != (
                    2,
                    edge_embeddings.shape[0],
            ):
                raise ValueError(
                    "Packed edge_index must have "
                    "shape (2, num_edges)."
                )

            source = edge_index[0].long()
            target = edge_index[1].long()

        elif node_embeddings.ndim == 3:
            if edge_embeddings.ndim != 3:
                raise ValueError(
                    "Dense edge_embeddings must "
                    "be 3D."
                )

            _, num_nodes, _ = (
                node_embeddings.shape
            )

            if edge_index.shape != (
                    node_embeddings.shape[0],
                    2,
                    edge_embeddings.shape[1],
            ):
                raise ValueError(
                    "Dense edge_index must have "
                    "shape "
                    "[batch, 2, num_edges]."
                )

            source = (
                edge_index[:, 0, :]
                .long()
            )
            target = (
                edge_index[:, 1, :]
                .long()
            )

        else:
            raise ValueError(
                "node_embeddings must be 2D "
                "packed tensors or 3D dense "
                "tensors."
            )

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
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        num_actions: int | None = None,
    ):
        super().__init__()

        self.model_type = "graph_policy_value_net_v2"

        self.num_bus_features = int(
            num_bus_features
        )
        self.num_branch_features = int(
            num_branch_features
        )
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        # Retained only as reference metadata for old callers.
        # Graph V2 forward must never use it to constrain topology size.
        self.reference_num_actions = (
            None
            if num_actions is None
            else int(num_actions)
        )

        if self.num_bus_features <= 0:
            raise ValueError(
                "num_bus_features must be positive."
            )

        if self.num_branch_features <= 0:
            raise ValueError(
                "num_branch_features must be positive."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        if (
                self.reference_num_actions is not None
                and self.reference_num_actions < 2
        ):
            raise ValueError(
                "Reference num_actions must contain "
                "at least stop plus one branch action."
            )

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
    def _validate_batch_vector(
            batch_index: torch.Tensor,
            *,
            item_count: int,
            name: str,
    ) -> int:
        """
        Validate node_batch or edge_batch and return graph count.
        """

        if batch_index.ndim != 1:
            raise ValueError(
                f"{name} must be 1D, got "
                f"{tuple(batch_index.shape)}."
            )

        if batch_index.shape != (item_count,):
            raise ValueError(
                f"{name} must contain one graph ID "
                f"per item. Expected {(item_count,)}, "
                f"got {tuple(batch_index.shape)}."
            )

        if item_count <= 0:
            raise ValueError(
                f"{name} must not be empty."
            )

        if batch_index.dtype != torch.long:
            batch_index = batch_index.long()

        minimum = int(batch_index.min().item())
        maximum = int(batch_index.max().item())

        if minimum != 0:
            raise ValueError(
                f"{name} graph IDs must start from 0, "
                f"got minimum {minimum}."
            )

        batch_size = maximum + 1

        observed_ids = torch.unique(
            batch_index,
            sorted=True,
        )

        expected_ids = torch.arange(
            batch_size,
            device=batch_index.device,
            dtype=torch.long,
        )

        if not torch.equal(
                observed_ids,
                expected_ids,
        ):
            raise ValueError(
                f"{name} graph IDs must be contiguous "
                "from 0."
            )

        return batch_size

    @classmethod
    def _require_grouped_batch_vector(
            cls,
            batch_index: torch.Tensor,
            *,
            item_count: int,
            name: str,
    ) -> tuple[int, torch.Tensor]:
        """
        Validate a packed batch vector and require graph-contiguous items.

        collate_graph_samples stores all nodes and edges of graph 0 first,
        followed by graph 1, and so on. Keeping this order makes local edge
        positions deterministic and preserves action-to-branch alignment.
        """

        batch_index = batch_index.long()

        batch_size = cls._validate_batch_vector(
            batch_index,
            item_count=item_count,
            name=name,
        )

        counts = torch.bincount(
            batch_index,
            minlength=batch_size,
        )

        expected = torch.arange(
            batch_size,
            device=batch_index.device,
            dtype=torch.long,
        ).repeat_interleave(counts)

        if not torch.equal(
                batch_index,
                expected,
        ):
            raise ValueError(
                f"{name} must group items contiguously "
                "by graph."
            )

        return batch_size, counts

    def _pack_dense_inputs(
            self,
            *,
            bus_features: torch.Tensor,
            branch_features: torch.Tensor,
            edge_index: torch.Tensor,
            edge_active_mask: torch.Tensor,
            action_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Convert an equal-size dense batch to the packed representation.

        This compatibility path keeps existing Graph V2 callers and tests working
        while the training pipeline migrates to collate_graph_samples.
        """

        if bus_features.ndim != 3:
            raise ValueError(
                "Dense bus_features must have shape "
                "[batch, num_nodes, num_bus_features]."
            )

        if branch_features.ndim != 3:
            raise ValueError(
                "Dense branch_features must have shape "
                "[batch, num_edges, num_branch_features]."
            )

        batch_size = int(
            bus_features.shape[0]
        )
        num_nodes = int(
            bus_features.shape[1]
        )
        num_edges = int(
            branch_features.shape[1]
        )

        if batch_size <= 0:
            raise ValueError(
                "Dense graph batch must not be empty."
            )

        if num_nodes <= 0:
            raise ValueError(
                "Every graph must contain at least one node."
            )

        if num_edges <= 0:
            raise ValueError(
                "Every graph must contain at least one edge."
            )

        if branch_features.shape[0] != batch_size:
            raise ValueError(
                "Bus and branch batch dimensions "
                "do not match."
            )

        if (
                bus_features.shape[2]
                != self.num_bus_features
        ):
            raise ValueError(
                "bus_features width mismatch: expected "
                f"{self.num_bus_features}, got "
                f"{bus_features.shape[2]}."
            )

        if (
                branch_features.shape[2]
                != self.num_branch_features
        ):
            raise ValueError(
                "branch_features width mismatch: expected "
                f"{self.num_branch_features}, got "
                f"{branch_features.shape[2]}."
            )

        expected_edge_index_shape = (
            batch_size,
            2,
            num_edges,
        )

        if edge_index.shape != (
                expected_edge_index_shape
        ):
            raise ValueError(
                "edge_index must have shape "
                "[batch, 2, num_edges], got "
                f"{tuple(edge_index.shape)}."
            )

        if edge_active_mask.shape != (
                batch_size,
                num_edges,
        ):
            observed_edges = (
                int(edge_active_mask.shape[-1])
                if edge_active_mask.ndim > 0
                else 0
            )

            raise ValueError(
                f"edge_active_mask has {observed_edges} "
                f"edges, but branch_features has "
                f"{num_edges}."
            )

        normalized_edge_parts: list[
            torch.Tensor
        ] = []

        for graph_index in range(batch_size):
            normalized_edge_index = (
                self._normalize_edge_index(
                    edge_index=edge_index[
                        graph_index
                    ],
                    num_nodes=num_nodes,
                )
            )

            normalized_edge_parts.append(
                normalized_edge_index
                + graph_index * num_nodes
            )

        node_batch = torch.arange(
            batch_size,
            device=bus_features.device,
            dtype=torch.long,
        ).repeat_interleave(num_nodes)

        edge_batch = torch.arange(
            batch_size,
            device=branch_features.device,
            dtype=torch.long,
        ).repeat_interleave(num_edges)

        packed_action_mask = action_mask

        if packed_action_mask is not None:
            if packed_action_mask.ndim == 1:
                packed_action_mask = (
                    packed_action_mask.unsqueeze(0)
                )

            expected_action_shape = (
                batch_size,
                num_edges + 1,
            )

            if packed_action_mask.shape != (
                    expected_action_shape
            ):
                raise ValueError(
                    "action_mask must have shape "
                    f"{expected_action_shape}, got "
                    f"{tuple(packed_action_mask.shape)}."
                )

            packed_action_mask = (
                packed_action_mask.bool()
            )

        return (
            bus_features.reshape(
                -1,
                self.num_bus_features,
            ),
            branch_features.reshape(
                -1,
                self.num_branch_features,
            ),
            torch.cat(
                normalized_edge_parts,
                dim=1,
            ),
            edge_active_mask.reshape(
                -1
            ).bool(),
            node_batch,
            edge_batch,
            packed_action_mask,
        )

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
        """
        Normalize dense, single-graph and packed inputs to one representation.
        """

        if bus_features.ndim == 3:
            if (
                    node_batch is not None
                    or edge_batch is not None
            ):
                raise ValueError(
                    "Dense graph inputs must not provide "
                    "node_batch or edge_batch."
                )

            (
                bus_features,
                branch_features,
                edge_index,
                edge_active_mask,
                node_batch,
                edge_batch,
                action_mask,
            ) = self._pack_dense_inputs(
                bus_features=bus_features,
                branch_features=branch_features,
                edge_index=edge_index,
                edge_active_mask=edge_active_mask,
                action_mask=action_mask,
            )

        elif bus_features.ndim == 2:
            if branch_features.ndim != 2:
                raise ValueError(
                    "Packed branch_features must be 2D."
                )

            if (
                    bus_features.shape[1]
                    != self.num_bus_features
            ):
                raise ValueError(
                    "bus_features width mismatch: expected "
                    f"{self.num_bus_features}, got "
                    f"{bus_features.shape[1]}."
                )

            if (
                    branch_features.shape[1]
                    != self.num_branch_features
            ):
                raise ValueError(
                    "branch_features width mismatch: expected "
                    f"{self.num_branch_features}, got "
                    f"{branch_features.shape[1]}."
                )

            num_nodes = int(
                bus_features.shape[0]
            )
            num_edges = int(
                branch_features.shape[0]
            )

            if num_nodes <= 0:
                raise ValueError(
                    "Packed graph batch must contain "
                    "at least one node."
                )

            if num_edges <= 0:
                raise ValueError(
                    "Packed graph batch must contain "
                    "at least one edge."
                )

            if edge_index.shape != (
                    2,
                    num_edges,
            ):
                raise ValueError(
                    "Packed edge_index must have shape "
                    f"{(2, num_edges)}, got "
                    f"{tuple(edge_index.shape)}."
                )

            if edge_active_mask.shape != (
                    num_edges,
            ):
                observed_edges = (
                    int(edge_active_mask.shape[-1])
                    if edge_active_mask.ndim > 0
                    else 0
                )

                raise ValueError(
                    f"edge_active_mask has "
                    f"{observed_edges} edges, but "
                    f"branch_features has {num_edges}."
                )

            edge_index = edge_index.long()
            edge_active_mask = (
                edge_active_mask.bool()
            )

            if (
                    node_batch is None
                    and edge_batch is None
            ):
                # One graph evaluated outside a DataLoader.
                edge_index = (
                    self._normalize_edge_index(
                        edge_index=edge_index,
                        num_nodes=num_nodes,
                    )
                )

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

            elif (
                    node_batch is None
                    or edge_batch is None
            ):
                raise ValueError(
                    "Packed batches require both "
                    "node_batch and edge_batch."
                )

            if action_mask is not None:
                if action_mask.ndim == 1:
                    action_mask = (
                        action_mask.unsqueeze(0)
                    )

                action_mask = action_mask.bool()

        else:
            raise ValueError(
                "bus_features must be a packed 2D "
                "tensor or a dense 3D tensor."
            )

        assert node_batch is not None
        assert edge_batch is not None

        node_batch = node_batch.long()
        edge_batch = edge_batch.long()
        edge_index = edge_index.long()
        edge_active_mask = (
            edge_active_mask.bool()
        )

        node_batch_size, node_counts = (
            self._require_grouped_batch_vector(
                node_batch,
                item_count=int(
                    bus_features.shape[0]
                ),
                name="node_batch",
            )
        )

        edge_batch_size, edge_counts = (
            self._require_grouped_batch_vector(
                edge_batch,
                item_count=int(
                    branch_features.shape[0]
                ),
                name="edge_batch",
            )
        )

        if node_batch_size != edge_batch_size:
            raise ValueError(
                "node_batch and edge_batch describe "
                "different graph counts."
            )

        batch_size = node_batch_size

        if bool((node_counts <= 0).any()):
            raise ValueError(
                "Every packed graph must contain "
                "at least one node."
            )

        if bool((edge_counts <= 0).any()):
            raise ValueError(
                "Every packed graph must contain "
                "at least one edge."
            )

        if edge_index.numel() == 0:
            raise ValueError(
                "edge_index must not be empty."
            )

        minimum_node_index = int(
            edge_index.min().item()
        )
        maximum_node_index = int(
            edge_index.max().item()
        )

        if (
                minimum_node_index < 0
                or maximum_node_index
                >= bus_features.shape[0]
        ):
            raise ValueError(
                "Packed edge_index contains node "
                "indices outside the packed node tensor."
            )

        source = edge_index[0]
        target = edge_index[1]

        if not torch.equal(
                node_batch[source],
                edge_batch,
        ):
            raise ValueError(
                "An edge source belongs to a different "
                "graph than the edge itself."
            )

        if not torch.equal(
                node_batch[target],
                edge_batch,
        ):
            raise ValueError(
                "An edge target belongs to a different "
                "graph than the edge itself."
            )

        max_edges = int(
            edge_counts.max().item()
        )

        if action_mask is not None:
            expected_action_shape = (
                batch_size,
                max_edges + 1,
            )

            if action_mask.shape != (
                    expected_action_shape
            ):
                raise ValueError(
                    "action_mask must have shape "
                    f"{expected_action_shape}, got "
                    f"{tuple(action_mask.shape)}."
                )

            if not bool(
                    action_mask.any(dim=1).all()
            ):
                raise ValueError(
                    "Every graph must contain at least "
                    "one legal action."
                )

            for graph_index, edge_count in enumerate(
                    edge_counts.tolist()
            ):
                padding = action_mask[
                          graph_index,
                          int(edge_count) + 1:,
                          ]

                if bool(padding.any()):
                    raise ValueError(
                        "action_mask padding must be "
                        f"false for graph {graph_index}."
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
        """
        Compute an independent mean for every graph in a packed batch.
        """

        if values.ndim != 2:
            raise ValueError(
                "Segmented mean expects values with "
                "shape [items, features]."
            )

        if batch_index.shape != (
                values.shape[0],
        ):
            raise ValueError(
                "batch_index must match the number "
                "of values."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        result = values.new_zeros(
            batch_size,
            values.shape[1],
        )

        result.index_add_(
            0,
            batch_index.long(),
            values,
        )

        counts = values.new_zeros(
            batch_size,
            1,
        )

        counts.index_add_(
            0,
            batch_index.long(),
            values.new_ones(
                values.shape[0],
                1,
            ),
        )

        return (
                result
                / counts.clamp_min(1.0)
        )

    @staticmethod
    def _segment_max(
            values: torch.Tensor,
            batch_index: torch.Tensor,
            batch_size: int,
    ) -> torch.Tensor:
        """
        Compute an independent feature-wise maximum for every graph.
        """

        if values.ndim != 2:
            raise ValueError(
                "Segmented max expects values with "
                "shape [items, features]."
            )

        if batch_index.shape != (
                values.shape[0],
        ):
            raise ValueError(
                "batch_index must match the number "
                "of values."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        result = values.new_full(
            (
                batch_size,
                values.shape[1],
            ),
            torch.finfo(values.dtype).min,
        )

        expanded_index = (
            batch_index
            .long()
            .unsqueeze(-1)
            .expand_as(values)
        )

        result.scatter_reduce_(
            dim=0,
            index=expanded_index,
            src=values,
            reduce="amax",
            include_self=True,
        )

        counts = torch.bincount(
            batch_index.long(),
            minlength=batch_size,
        )

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
        """
        Apply softmax independently to the items of every graph.
        """

        if scores.ndim != 1:
            raise ValueError(
                "Segmented softmax expects a 1D "
                "score tensor."
            )

        if batch_index.shape != scores.shape:
            raise ValueError(
                "batch_index must match scores."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        maxima = scores.new_full(
            (batch_size,),
            torch.finfo(scores.dtype).min,
        )

        maxima.scatter_reduce_(
            dim=0,
            index=batch_index.long(),
            src=scores,
            reduce="amax",
            include_self=True,
        )

        shifted_scores = (
                scores
                - maxima[batch_index.long()]
        )

        exponentials = torch.exp(
            shifted_scores
        )

        denominators = scores.new_zeros(
            batch_size
        )

        denominators.index_add_(
            0,
            batch_index.long(),
            exponentials,
        )

        return (
                exponentials
                / denominators[
                    batch_index.long()
                ].clamp_min(1e-12)
        )

    @classmethod
    def _segment_active_mean(
            cls,
            values: torch.Tensor,
            batch_index: torch.Tensor,
            active_mask: torch.Tensor,
            batch_size: int,
    ) -> torch.Tensor:
        if active_mask.shape != (
                values.shape[0],
        ):
            raise ValueError(
                "active_mask must match the number "
                "of values."
            )

        active_values = values[
            active_mask
        ]
        active_batch = batch_index[
            active_mask
        ]

        if active_values.shape[0] == 0:
            return values.new_zeros(
                batch_size,
                values.shape[1],
            )

        return cls._segment_mean(
            active_values,
            active_batch,
            batch_size,
        )

    @classmethod
    def _segment_active_max(
            cls,
            values: torch.Tensor,
            batch_index: torch.Tensor,
            active_mask: torch.Tensor,
            batch_size: int,
    ) -> torch.Tensor:
        if active_mask.shape != (
                values.shape[0],
        ):
            raise ValueError(
                "active_mask must match the number "
                "of values."
            )

        active_values = values[
            active_mask
        ]
        active_batch = batch_index[
            active_mask
        ]

        if active_values.shape[0] == 0:
            return values.new_zeros(
                batch_size,
                values.shape[1],
            )

        return cls._segment_max(
            active_values,
            active_batch,
            batch_size,
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
            *,
            edge_embeddings: torch.Tensor,
            edge_batch: torch.Tensor,
            edge_active_mask: torch.Tensor,
            batch_size: int,
    ) -> torch.Tensor:
        """
        Pool learned edge severity independently for every packed graph.

        Only physically active edges contribute. A graph without active edges
        receives a zero overload context.
        """

        if edge_embeddings.ndim != 2:
            raise ValueError(
                "Packed edge embeddings must be 2D."
            )

        num_edges = int(
            edge_embeddings.shape[0]
        )

        if edge_batch.shape != (
                num_edges,
        ):
            raise ValueError(
                "edge_batch must match edge embeddings."
            )

        if edge_active_mask.shape != (
                num_edges,
        ):
            raise ValueError(
                "edge_active_mask must match edge "
                "embeddings."
            )

        scores = self.overload_attention(
            edge_embeddings
        ).squeeze(-1)

        mask_value = torch.finfo(
            scores.dtype
        ).min

        masked_scores = scores.masked_fill(
            ~edge_active_mask,
            mask_value,
        )

        weights = self._segment_softmax(
            masked_scores,
            edge_batch,
            batch_size,
        )

        weights = (
                weights
                * edge_active_mask.to(
            dtype=weights.dtype
        )
        )

        weighted_edges = (
                edge_embeddings
                * weights.unsqueeze(-1)
        )

        overload_pool = edge_embeddings.new_zeros(
            batch_size,
            self.hidden_dim,
        )

        overload_pool.index_add_(
            0,
            edge_batch,
            weighted_edges,
        )

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Build one physical global context for every packed graph.

        Action legality is intentionally not used here. Value prediction depends
        on the physical topology, while action_mask only constrains policy output.
        """

        node_mean = self._segment_mean(
            node_embeddings,
            node_batch,
            batch_size,
        )

        node_max = self._segment_max(
            node_embeddings,
            node_batch,
            batch_size,
        )

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

        overload_pool = (
            self._overload_focused_pool(
                edge_embeddings=edge_embeddings,
                edge_batch=edge_batch,
                edge_active_mask=(
                    edge_active_mask
                ),
                batch_size=batch_size,
            )
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

        global_projected = (
            self.global_projection(
                global_embedding
            )
        )

        return (
            global_embedding,
            global_projected,
            overload_pool,
        )

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
        """
        Evaluate one graph, a legacy dense batch, or a packed variable-size batch.

        Packed representation:
            bus_features       [total_nodes, bus_feature_dim]
            branch_features    [total_edges, branch_feature_dim]
            edge_index         [2, total_edges]
            edge_active_mask   [total_edges]
            node_batch         [total_nodes]
            edge_batch         [total_edges]

        Policy output remains dense:
            policy_logits      [batch_size, 1 + max_edges_in_batch]
        """

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

        node_embeddings = self.bus_encoder(
            bus_features
        )

        edge_embeddings = self.branch_encoder(
            branch_features
        )

        for layer in self.layers:
            (
                node_embeddings,
                edge_embeddings,
            ) = layer(
                node_embeddings=node_embeddings,
                edge_embeddings=edge_embeddings,
                edge_index=edge_index,
                edge_active_mask=edge_active_mask,
            )

        source = edge_index[0].long()
        target = edge_index[1].long()

        source_node = node_embeddings[source]
        target_node = node_embeddings[target]

        (
            global_embedding,
            global_projected,
            overload_pool,
        ) = self._build_contexts(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            node_batch=node_batch,
            edge_batch=edge_batch,
            edge_active_mask=edge_active_mask,
            batch_size=batch_size,
        )

        global_repeated = global_projected[
            edge_batch
        ]

        overload_repeated = overload_pool[
            edge_batch
        ]

        branch_repr = torch.cat(
            [
                source_node,
                target_node,
                edge_embeddings,
                torch.abs(
                    source_node - target_node
                ),
                source_node * target_node,
                global_repeated,
                overload_repeated,
            ],
            dim=-1,
        )

        branch_logits_flat = (
            self.branch_policy_head(
                branch_repr
            ).squeeze(-1)
        )

        max_edges = int(
            edge_counts.max().item()
        )

        edge_offsets = torch.cumsum(
            torch.cat(
                [
                    edge_counts.new_zeros(1),
                    edge_counts[:-1],
                ]
            ),
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

        if bool(
                (
                        edge_positions < 0
                ).any()
        ):
            raise RuntimeError(
                "Computed a negative local edge "
                "position."
            )

        if bool(
                (
                        edge_positions
                        >= edge_counts[edge_batch]
                ).any()
        ):
            raise RuntimeError(
                "Computed an out-of-range local edge "
                "position."
            )

        mask_value = torch.finfo(
            branch_logits_flat.dtype
        ).min

        branch_logits = (
            branch_logits_flat.new_full(
                (
                    batch_size,
                    max_edges,
                ),
                mask_value,
            )
        )

        branch_logits[
            edge_batch,
            edge_positions,
        ] = branch_logits_flat

        stop_value_input = torch.cat(
            [
                global_embedding,
                global_projected,
            ],
            dim=-1,
        )

        stop_logit = self.stop_policy_head(
            stop_value_input
        )

        policy_logits = torch.cat(
            [
                stop_logit,
                branch_logits,
            ],
            dim=1,
        )

        if action_mask is not None:
            if action_mask.shape != (
                    policy_logits.shape
            ):
                raise ValueError(
                    "action_mask shape does not match "
                    "dynamic policy output. "
                    f"Expected "
                    f"{tuple(policy_logits.shape)}, got "
                    f"{tuple(action_mask.shape)}."
                )

            policy_logits = (
                policy_logits.masked_fill(
                    ~action_mask,
                    mask_value,
                )
            )

        value = self.value_head(
            stop_value_input
        ).squeeze(-1)

        return policy_logits, value
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """
    Small reusable MLP block.

    This is intentionally simple and dependency-free.
    We do not use PyTorch Geometric here, so the model works on Windows,
    CPU, CUDA and AMP without torch-scatter / torch-sparse issues.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]

        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.extend(
            [
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
            ]
        )

        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeMessagePassingLayer(nn.Module):
    """
    Lightweight GAT-like message passing layer.

    This layer updates:
    - node embeddings using messages sent along branches;
    - edge embeddings using endpoint node embeddings.

    It is not torch_geometric.nn.GATConv.
    It is a pure PyTorch implementation designed for this project.

    Why this design:
    - no torch-scatter dependency;
    - stable on Windows;
    - works with CUDA and AMP;
    - keeps branch/edge features central, which is important because actions
      are branch switching actions.
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

        self.attention_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _gather_nodes(
        node_embeddings: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather node embeddings for every edge endpoint.

        node_embeddings:
            [batch, num_nodes, hidden_dim]

        indices:
            [batch, num_edges]

        returns:
            [batch, num_edges, hidden_dim]
        """

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
    ) -> torch.Tensor:
        """
        Aggregate edge messages into target nodes.

        messages:
            [batch, num_edges, hidden_dim]

        target_indices:
            [batch, num_edges]

        returns:
            [batch, num_nodes, hidden_dim]
        """

        batch_size, _, hidden_dim = messages.shape

        aggregated = messages.new_zeros(
            batch_size,
            num_nodes,
            hidden_dim,
        )

        counts = messages.new_zeros(
            batch_size,
            num_nodes,
            1,
        )

        ones = messages.new_ones(
            batch_size,
            target_indices.shape[1],
            1,
        )

        for batch_idx in range(batch_size):
            aggregated[batch_idx].index_add_(
                dim=0,
                index=target_indices[batch_idx],
                source=messages[batch_idx],
            )

            counts[batch_idx].index_add_(
                dim=0,
                index=target_indices[batch_idx],
                source=ones[batch_idx],
            )

        aggregated = aggregated / counts.clamp_min(1.0)

        return aggregated

    def _directional_messages(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted messages from source nodes to target nodes.
        """

        source_node = self._gather_nodes(
            node_embeddings=node_embeddings,
            indices=source_indices,
        )

        target_node = self._gather_nodes(
            node_embeddings=node_embeddings,
            indices=target_indices,
        )

        message_input = torch.cat(
            [source_node, target_node, edge_embeddings],
            dim=-1,
        )

        raw_message = self.message_mlp(message_input)

        # Sigmoid attention is used instead of softmax-by-target-node.
        # This keeps the implementation dependency-free and stable.
        attention = torch.sigmoid(self.attention_mlp(message_input))

        return raw_message * attention

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One message passing step.

        node_embeddings:
            [batch, num_nodes, hidden_dim]

        edge_embeddings:
            [batch, num_edges, hidden_dim]

        edge_index:
            [batch, 2, num_edges]
        """

        batch_size, num_nodes, _ = node_embeddings.shape

        source = edge_index[:, 0, :].long()
        target = edge_index[:, 1, :].long()

        # Forward direction: source -> target.
        forward_messages = self._directional_messages(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            source_indices=source,
            target_indices=target,
        )

        forward_aggregated = self._aggregate_messages(
            messages=forward_messages,
            target_indices=target,
            num_nodes=num_nodes,
        )

        # Reverse direction: target -> source.
        # Power-grid branches are physically usable in both directions for
        # information propagation, even when active power has a sign.
        reverse_messages = self._directional_messages(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            source_indices=target,
            target_indices=source,
        )

        reverse_aggregated = self._aggregate_messages(
            messages=reverse_messages,
            target_indices=source,
            num_nodes=num_nodes,
        )

        aggregated = 0.5 * (forward_aggregated + reverse_aggregated)

        node_delta = self.node_update(
            torch.cat([node_embeddings, aggregated], dim=-1)
        )

        new_node_embeddings = self.node_norm(node_embeddings + node_delta)

        source_node = self._gather_nodes(
            node_embeddings=new_node_embeddings,
            indices=source,
        )

        target_node = self._gather_nodes(
            node_embeddings=new_node_embeddings,
            indices=target,
        )

        edge_delta = self.edge_update(
            torch.cat([source_node, target_node, edge_embeddings], dim=-1)
        )

        new_edge_embeddings = self.edge_norm(edge_embeddings + edge_delta)

        return new_node_embeddings, new_edge_embeddings


class GraphPolicyValueNet(nn.Module):
    """
    Graph policy-value network for topology switching.

    This model is added alongside SimplePolicyValueNet.
    It does not replace or delete the MLP baseline.

    Inputs
    ------
    bus_features:
        [batch, num_buses, num_bus_features]

    branch_features:
        [batch, num_branches, num_branch_features]

    edge_index:
        [batch, 2, num_branches]

    action_mask:
        [batch, 1 + num_branches]

    Outputs
    -------
    policy_logits:
        [batch, 1 + num_branches]

        action 0:
            stop / handoff to redispatch

        action k > 0:
            switch off branch with branch_pos = k - 1

    value:
        [batch]

        value in [-1, 1]
    """

    def __init__(
        self,
        num_bus_features: int,
        num_branch_features: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.model_type = "graph_policy_value_net"

        self.num_bus_features = int(num_bus_features)
        self.num_branch_features = int(num_branch_features)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        if self.num_actions < 2:
            raise ValueError(
                "num_actions must be at least 2: action 0 + at least one branch action."
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
                EdgeMessagePassingLayer(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        branch_head_input_dim = hidden_dim * 5

        self.branch_policy_head = nn.Sequential(
            nn.Linear(branch_head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

        global_dim = hidden_dim * 4

        self.stop_policy_head = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _ensure_batched_features(
        tensor: torch.Tensor,
        expected_rank_without_batch: int,
        name: str,
    ) -> torch.Tensor:
        """
        Add batch dimension when a single graph is passed.
        """

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
        """
        Ensure edge_index is zero-based.

        GridFM / power-grid data can sometimes use 1-based bus numbering.
        This helper handles the common case:
            edge labels 1..N -> convert to 0..N-1.

        If indices are already 0-based, they are left unchanged.
        """

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
        """
        Gather node embeddings for edge endpoints.
        """

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
        """
        Mean with optional boolean mask.

        values:
            [batch, items, hidden_dim]

        mask:
            [batch, items]
        """

        if mask is None:
            return values.mean(dim=dim)

        mask_float = mask.to(dtype=values.dtype).unsqueeze(-1)
        summed = (values * mask_float).sum(dim=dim)
        count = mask_float.sum(dim=dim).clamp_min(1.0)

        return summed / count

    def _build_global_embedding(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Build graph-level embedding for stop policy and value head.
        """

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

        if branch_mask is None:
            edge_max = edge_embeddings.max(dim=1).values
        else:
            mask_value = torch.finfo(edge_embeddings.dtype).min
            masked_edges = edge_embeddings.masked_fill(
                ~branch_mask.unsqueeze(-1),
                mask_value,
            )
            edge_max = masked_edges.max(dim=1).values

            # If all branch actions are masked for a sample, max would be a
            # huge negative value. Replace it with zeros.
            no_valid_branch = ~branch_mask.any(dim=1)
            if bool(no_valid_branch.any()):
                edge_max = edge_max.clone()
                edge_max[no_valid_branch] = 0.0

        return torch.cat(
            [node_mean, node_max, edge_mean, edge_max],
            dim=-1,
        )

    def forward(
        self,
        bus_features: torch.Tensor,
        branch_features: torch.Tensor,
        edge_index: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        """

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

        if action_mask is not None:
            action_mask = self._ensure_batched_features(
                tensor=action_mask,
                expected_rank_without_batch=1,
                name="action_mask",
            ).bool()

        batch_size, num_nodes, _ = bus_features.shape
        _, num_edges, _ = branch_features.shape

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

        if num_edges != self.num_branch_actions:
            raise ValueError(
                f"Model was initialized for {self.num_branch_actions} branch actions, "
                f"but input has {num_edges} branches."
            )

        if action_mask is not None and action_mask.shape[1] != self.num_actions:
            raise ValueError(
                f"action_mask has {action_mask.shape[1]} actions, "
                f"but model expects {self.num_actions}."
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

        branch_repr = torch.cat(
            [
                source_node,
                target_node,
                edge_embeddings,
                torch.abs(source_node - target_node),
                source_node * target_node,
            ],
            dim=-1,
        )

        branch_logits = self.branch_policy_head(branch_repr).squeeze(-1)

        global_embedding = self._build_global_embedding(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            action_mask=action_mask,
        )

        stop_logit = self.stop_policy_head(global_embedding)

        policy_logits = torch.cat(
            [stop_logit, branch_logits],
            dim=1,
        )

        if action_mask is not None:
            # dtype-safe mask value for CUDA AMP / float16.
            mask_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(~action_mask, mask_value)

        value = self.value_head(global_embedding).squeeze(-1)

        return policy_logits, value
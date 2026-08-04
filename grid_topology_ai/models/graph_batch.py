from __future__ import annotations

from typing import Any

import torch


GRAPH_BATCHING_CONTRACT_VERSION = 1

_REQUIRED_TENSOR_FIELDS = (
    "bus_features",
    "branch_features",
    "edge_index",
    "edge_active_mask",
    "action_mask",
    "target_policy",
    "target_value",
)

_REQUIRED_METADATA_FIELDS = (
    "scenario_id",
    "step",
    "state_id",
)


def _normalize_edge_index(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    """
    Convert one graph's edge_index to zero-based node positions.

    State artifacts normally use zero-based node positions. One-based indices
    are also accepted for compatibility with the existing Graph V2 contract.
    """

    if edge_index.ndim != 2:
        raise ValueError(
            "edge_index must be 2D, "
            f"got {tuple(edge_index.shape)}."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape (2, num_edges), "
            f"got {tuple(edge_index.shape)}."
        )

    if edge_index.shape[1] <= 0:
        raise ValueError(
            "Every graph must contain at least one edge."
        )

    edge_index = edge_index.long()

    minimum = int(edge_index.min().item())
    maximum = int(edge_index.max().item())

    if minimum >= 0 and maximum < num_nodes:
        return edge_index

    if minimum >= 1 and maximum <= num_nodes:
        return edge_index - 1

    raise ValueError(
        "edge_index contains node indices outside the graph: "
        f"min={minimum}, max={maximum}, num_nodes={num_nodes}."
    )


def _require_sample_fields(
    sample: dict[str, Any],
    *,
    graph_index: int,
) -> None:
    missing = [
        name
        for name in (
            *_REQUIRED_TENSOR_FIELDS,
            *_REQUIRED_METADATA_FIELDS,
        )
        if name not in sample
    ]

    if missing:
        raise ValueError(
            f"Graph sample {graph_index} is missing required "
            f"fields: {missing}."
        )

    non_tensor = [
        name
        for name in _REQUIRED_TENSOR_FIELDS
        if not torch.is_tensor(sample[name])
    ]

    if non_tensor:
        raise TypeError(
            f"Graph sample {graph_index} contains non-tensor "
            f"fields where tensors are required: {non_tensor}."
        )


def _validate_sample(
    sample: dict[str, Any],
    *,
    graph_index: int,
    expected_bus_feature_width: int | None,
    expected_branch_feature_width: int | None,
) -> tuple[int, int, int, int, int]:
    """
    Validate one graph before it is packed.

    Returns
    -------
    tuple
        num_nodes, num_edges, num_actions,
        bus_feature_width, branch_feature_width.
    """

    _require_sample_fields(
        sample,
        graph_index=graph_index,
    )

    bus_features = sample["bus_features"]
    branch_features = sample["branch_features"]
    edge_index = sample["edge_index"]
    edge_active_mask = sample["edge_active_mask"]
    action_mask = sample["action_mask"]
    target_policy = sample["target_policy"]
    target_value = sample["target_value"]

    if bus_features.ndim != 2:
        raise ValueError(
            f"Graph sample {graph_index}: bus_features must "
            f"be 2D, got {tuple(bus_features.shape)}."
        )

    if branch_features.ndim != 2:
        raise ValueError(
            f"Graph sample {graph_index}: branch_features "
            f"must be 2D, got "
            f"{tuple(branch_features.shape)}."
        )

    num_nodes = int(bus_features.shape[0])
    num_edges = int(branch_features.shape[0])
    bus_feature_width = int(bus_features.shape[1])
    branch_feature_width = int(
        branch_features.shape[1]
    )

    if num_nodes <= 0:
        raise ValueError(
            f"Graph sample {graph_index} must contain "
            "at least one node."
        )

    if num_edges <= 0:
        raise ValueError(
            f"Graph sample {graph_index} must contain "
            "at least one edge."
        )

    if bus_feature_width <= 0:
        raise ValueError(
            f"Graph sample {graph_index} must contain "
            "at least one bus feature."
        )

    if branch_feature_width <= 0:
        raise ValueError(
            f"Graph sample {graph_index} must contain "
            "at least one branch feature."
        )

    if (
        expected_bus_feature_width is not None
        and bus_feature_width
        != expected_bus_feature_width
    ):
        raise ValueError(
            "All graphs in one batch must use the same "
            "bus feature width. "
            f"Expected {expected_bus_feature_width}, "
            f"got {bus_feature_width} for graph "
            f"{graph_index}."
        )

    if (
        expected_branch_feature_width is not None
        and branch_feature_width
        != expected_branch_feature_width
    ):
        raise ValueError(
            "All graphs in one batch must use the same "
            "branch feature width. "
            f"Expected {expected_branch_feature_width}, "
            f"got {branch_feature_width} for graph "
            f"{graph_index}."
        )

    if edge_index.shape != (2, num_edges):
        raise ValueError(
            f"Graph sample {graph_index}: edge_index must "
            f"have shape {(2, num_edges)}, got "
            f"{tuple(edge_index.shape)}."
        )

    if edge_active_mask.shape != (num_edges,):
        raise ValueError(
            f"Graph sample {graph_index}: "
            "edge_active_mask must match the number "
            f"of edges. Expected {(num_edges,)}, got "
            f"{tuple(edge_active_mask.shape)}."
        )

    if action_mask.ndim != 1:
        raise ValueError(
            f"Graph sample {graph_index}: action_mask "
            f"must be 1D, got "
            f"{tuple(action_mask.shape)}."
        )

    num_actions = int(action_mask.numel())
    expected_num_actions = num_edges + 1

    if num_actions != expected_num_actions:
        raise ValueError(
            f"Graph sample {graph_index}: policy must "
            "contain one stop action plus one action "
            f"per edge. Expected {expected_num_actions}, "
            f"got {num_actions}."
        )

    if not bool(action_mask.any()):
        raise ValueError(
            f"Graph sample {graph_index} must contain "
            "at least one legal action."
        )

    if target_policy.shape != (num_actions,):
        raise ValueError(
            f"Graph sample {graph_index}: target_policy "
            "must match action_mask. "
            f"Expected {(num_actions,)}, got "
            f"{tuple(target_policy.shape)}."
        )

    if target_value.numel() != 1:
        raise ValueError(
            f"Graph sample {graph_index}: target_value "
            "must contain exactly one value, "
            f"got shape {tuple(target_value.shape)}."
        )

    if not torch.isfinite(
        bus_features
    ).all():
        raise ValueError(
            f"Graph sample {graph_index}: bus_features "
            "must contain only finite values."
        )

    if not torch.isfinite(
        branch_features
    ).all():
        raise ValueError(
            f"Graph sample {graph_index}: "
            "branch_features must contain only "
            "finite values."
        )

    if not torch.isfinite(
        target_policy
    ).all():
        raise ValueError(
            f"Graph sample {graph_index}: target_policy "
            "must contain only finite values."
        )

    if not torch.isfinite(
        target_value
    ).all():
        raise ValueError(
            f"Graph sample {graph_index}: target_value "
            "must be finite."
        )

    if bool(
        (
            target_policy[
                ~action_mask.bool()
            ]
            != 0.0
        ).any()
    ):
        raise ValueError(
            f"Graph sample {graph_index}: target_policy "
            "assigns probability to masked actions."
        )

    _normalize_edge_index(
        edge_index,
        num_nodes=num_nodes,
    )

    return (
        num_nodes,
        num_edges,
        num_actions,
        bus_feature_width,
        branch_feature_width,
    )


def collate_graph_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Pack variable-size samples into one disconnected graph batch.

    Nodes and edges are concatenated without padding. Each edge_index is shifted
    by the accumulated node count. node_batch and edge_batch identify the graph
    to which every node and edge belongs.

    Policy tensors remain dense because the policy loss expects a
    [batch_size, num_actions] tensor. They are padded only to the largest action
    count in the current batch. Padded actions are always masked.
    """

    if not samples:
        raise ValueError(
            "Cannot collate an empty graph batch."
        )

    sample_dimensions: list[
        tuple[int, int, int]
    ] = []

    expected_bus_feature_width: int | None = None
    expected_branch_feature_width: int | None = None

    for graph_index, sample in enumerate(samples):
        (
            num_nodes,
            num_edges,
            num_actions,
            bus_feature_width,
            branch_feature_width,
        ) = _validate_sample(
            sample,
            graph_index=graph_index,
            expected_bus_feature_width=(
                expected_bus_feature_width
            ),
            expected_branch_feature_width=(
                expected_branch_feature_width
            ),
        )

        if expected_bus_feature_width is None:
            expected_bus_feature_width = (
                bus_feature_width
            )

        if expected_branch_feature_width is None:
            expected_branch_feature_width = (
                branch_feature_width
            )

        sample_dimensions.append(
            (
                num_nodes,
                num_edges,
                num_actions,
            )
        )

    batch_size = len(samples)
    max_num_actions = max(
        num_actions
        for _, _, num_actions in sample_dimensions
    )

    bus_parts: list[torch.Tensor] = []
    branch_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_active_parts: list[torch.Tensor] = []

    node_batch_parts: list[torch.Tensor] = []
    edge_batch_parts: list[torch.Tensor] = []

    target_values: list[torch.Tensor] = []
    scenario_ids: list[int] = []
    steps: list[int] = []
    state_ids: list[str] = []

    node_ptr = [0]
    edge_ptr = [0]

    action_mask = torch.zeros(
        batch_size,
        max_num_actions,
        dtype=torch.bool,
    )

    target_policy = torch.zeros(
        batch_size,
        max_num_actions,
        dtype=torch.float32,
    )

    node_offset = 0

    for graph_index, sample in enumerate(samples):
        (
            num_nodes,
            num_edges,
            num_actions,
        ) = sample_dimensions[graph_index]

        bus_features = sample[
            "bus_features"
        ].float()

        branch_features = sample[
            "branch_features"
        ].float()

        edge_index = _normalize_edge_index(
            sample["edge_index"],
            num_nodes=num_nodes,
        )

        edge_active_mask = sample[
            "edge_active_mask"
        ].bool()

        sample_action_mask = sample[
            "action_mask"
        ].bool()

        sample_target_policy = sample[
            "target_policy"
        ].float()

        shifted_edge_index = (
            edge_index + node_offset
        )

        bus_parts.append(
            bus_features
        )
        branch_parts.append(
            branch_features
        )
        edge_index_parts.append(
            shifted_edge_index
        )
        edge_active_parts.append(
            edge_active_mask
        )

        node_batch_parts.append(
            torch.full(
                (num_nodes,),
                graph_index,
                dtype=torch.long,
            )
        )

        edge_batch_parts.append(
            torch.full(
                (num_edges,),
                graph_index,
                dtype=torch.long,
            )
        )

        action_mask[
            graph_index,
            :num_actions,
        ] = sample_action_mask

        target_policy[
            graph_index,
            :num_actions,
        ] = sample_target_policy

        target_values.append(
            sample["target_value"]
            .float()
            .reshape(())
        )

        scenario_ids.append(
            int(sample["scenario_id"])
        )

        steps.append(
            int(sample["step"])
        )

        state_ids.append(
            str(sample["state_id"])
        )

        node_offset += num_nodes

        node_ptr.append(
            node_offset
        )

        edge_ptr.append(
            edge_ptr[-1] + num_edges
        )

    return {
        "bus_features": torch.cat(
            bus_parts,
            dim=0,
        ),
        "branch_features": torch.cat(
            branch_parts,
            dim=0,
        ),
        "edge_index": torch.cat(
            edge_index_parts,
            dim=1,
        ),
        "edge_active_mask": torch.cat(
            edge_active_parts,
            dim=0,
        ),
        "node_batch": torch.cat(
            node_batch_parts,
            dim=0,
        ),
        "edge_batch": torch.cat(
            edge_batch_parts,
            dim=0,
        ),
        "node_ptr": torch.tensor(
            node_ptr,
            dtype=torch.long,
        ),
        "edge_ptr": torch.tensor(
            edge_ptr,
            dtype=torch.long,
        ),
        "action_mask": action_mask,
        "target_policy": target_policy,
        "target_value": torch.stack(
            target_values,
        ),
        "scenario_id": torch.tensor(
            scenario_ids,
            dtype=torch.long,
        ),
        "step": torch.tensor(
            steps,
            dtype=torch.long,
        ),
        "state_id": state_ids,
    }
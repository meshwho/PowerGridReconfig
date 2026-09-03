from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.self_play.example_validation import (
    load_and_validate_examples_csv,
    policy_vector_from_json,
    validate_example_contract_versions,
    validate_example_topology_action_contracts,
)
from grid_topology_ai.actions import (
    action_layout_fingerprint,
    require_branch_status_policy_layout,
)

_STATE_ARRAYS = (
    "bus_features",
    "branch_features",
    "edge_index",
    "branch_status",
    "action_mask",
)


class _RunningMoments:
    def __init__(self, feature_count: int):
        if feature_count <= 0:
            raise ValueError("feature_count must be positive.")

        self.feature_count = int(feature_count)
        self.count = 0
        self.mean = np.zeros(self.feature_count, dtype=np.float64)
        self.m2 = np.zeros(self.feature_count, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)

        if batch.ndim != 2:
            raise ValueError(
                f"Feature batch must be 2D, got {batch.shape}."
            )
        if batch.shape[1] != self.feature_count:
            raise ValueError(
                "Feature batch width mismatch. "
                f"Expected {self.feature_count}, got {batch.shape[1]}."
            )
        if batch.shape[0] == 0:
            raise ValueError("Feature batch must not be empty.")
        if not np.isfinite(batch).all():
            raise ValueError("Feature batch must contain only finite values.")

        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean.copy()
            self.m2 = batch_m2.copy()
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean += delta * (batch_count / total_count)
        self.m2 += (
            batch_m2
            + delta * delta * self.count * batch_count / total_count
        )
        self.count = total_count

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("No feature batches were added.")

        variance = np.maximum(self.m2 / self.count, 0.0)
        mean = self.mean.astype(np.float32)
        std = np.sqrt(variance).astype(np.float32)
        std[std < 1e-6] = 1.0
        return mean, std


class GraphSelfPlayDataset(Dataset):
    """
    Graph dataset for AlphaZero-like policy-value training.

    This dataset is parallel to SelfPlayDataset.

    It reads the same:
        examples.csv
        states/*.npz

    But instead of a flat state_vector it returns graph tensors:
        bus_features
        branch_features
        edge_index
        edge_active_mask
        action_mask
        target_policy
        target_value

    This is used by GraphPolicyValueNet / GNN / GAT-like models.
    """

    def __init__(
        self,
        examples_csv: str | Path,
        normalize_features: bool = True,
        normalization_stats: dict[str, np.ndarray] | None = None,
        physics_config: PhysicsConfig | None = None,
    ):
        self.examples_csv = Path(examples_csv)
        self.normalize_features = bool(normalize_features)
        self.examples = load_and_validate_examples_csv(
            self.examples_csv
        )

        self.physics_config = validate_example_contract_versions(
            self.examples,
            source_path=self.examples_csv,
            expected_physics_config=physics_config,
        )
        (
            self.topology_action_config,
            self.action_layout,
        ) = validate_example_topology_action_contracts(
            self.examples,
            source_path=self.examples_csv,
        )

        self.action_layout_fingerprint = action_layout_fingerprint(
            self.action_layout
        )
        self.policy_layout = require_branch_status_policy_layout(
            self.action_layout
        )

        self.action_layout_count = int(
            self.examples[
                "action_layout_fingerprint"
            ].nunique(dropna=False)
        )

        first_data = self._load_npz_by_index(0)

        self.num_bus_features = int(
            first_data["bus_features"].shape[1]
        )
        self.num_branch_features = int(
            first_data["branch_features"].shape[1]
        )
        first_num_actions = int(first_data["action_mask"].shape[0])
        if first_num_actions != len(self.action_layout):
            raise ValueError(
                "The first state action mask does not "
                "match its action layout. "
                f"Expected {len(self.action_layout)}, "
                f"got {first_num_actions}."
            )

        if normalization_stats is not None:
            self.bus_feature_mean = np.array(
                normalization_stats["bus_feature_mean"],
                dtype=np.float32,
                copy=True,
            )
            self.bus_feature_std = np.array(
                normalization_stats["bus_feature_std"],
                dtype=np.float32,
                copy=True,
            )
            self.branch_feature_mean = np.array(
                normalization_stats["branch_feature_mean"],
                dtype=np.float32,
                copy=True,
            )
            self.branch_feature_std = np.array(
                normalization_stats["branch_feature_std"],
                dtype=np.float32,
                copy=True,
            )
        elif self.normalize_features:
            (
                self.bus_feature_mean,
                self.bus_feature_std,
                self.branch_feature_mean,
                self.branch_feature_std,
            ) = self._compute_feature_statistics()
        else:
            self.bus_feature_mean = np.zeros(
                self.num_bus_features,
                dtype=np.float32,
            )
            self.bus_feature_std = np.ones(
                self.num_bus_features,
                dtype=np.float32,
            )
            self.branch_feature_mean = np.zeros(
                self.num_branch_features,
                dtype=np.float32,
            )
            self.branch_feature_std = np.ones(
                self.num_branch_features,
                dtype=np.float32,
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples.iloc[idx]
        state_path = self._state_path(row["state_path"])
        data = self._load_npz_by_index(idx)

        bus_features = data["bus_features"].astype(np.float32)
        branch_features = data["branch_features"].astype(np.float32)
        edge_index = data["edge_index"].astype(np.int64)
        branch_status = np.asarray(
            data["branch_status"],
            dtype=np.float32,
        )
        edge_active_mask = branch_status > 0.5
        action_mask = data["action_mask"].astype(bool)

        self._validate_graph_shapes(
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            branch_status=branch_status,
            edge_active_mask=edge_active_mask,
            action_mask=action_mask,
            state_path=state_path,
        )

        bus_features = self._normalize_bus_features(bus_features)
        branch_features = self._normalize_branch_features(
            branch_features
        )

        target_policy = policy_vector_from_json(
            row["mcts_policy_json"],
            action_mask=action_mask,
            index=idx,
            source_path=self.examples_csv,
        )

        # Strict v3 logic:
        # The value head is trained only on outcome_value_target.
        target_value = float(row["outcome_value_target"])

        return {
            "bus_features": torch.tensor(
                bus_features,
                dtype=torch.float32,
            ),
            "branch_features": torch.tensor(
                branch_features,
                dtype=torch.float32,
            ),
            "edge_index": torch.tensor(
                edge_index,
                dtype=torch.long,
            ),
            "edge_active_mask": torch.tensor(
                edge_active_mask,
                dtype=torch.bool,
            ),
            "action_mask": torch.tensor(
                action_mask,
                dtype=torch.bool,
            ),
            "target_policy": torch.tensor(
                target_policy,
                dtype=torch.float32,
            ),
            "target_value": torch.tensor(
                target_value,
                dtype=torch.float32,
            ),
            "scenario_id": int(row["scenario_id"]),
            "step": int(row["step"]),
            "state_id": str(row["state_id"]),
        }

    def _state_path(self, state_path: object) -> Path:
        return Path(str(state_path).strip())

    def _load_npz_by_index(
        self,
        idx: int,
    ) -> dict[str, np.ndarray]:
        row = self.examples.iloc[idx]
        state_path = self._state_path(row["state_path"])

        with np.load(state_path, allow_pickle=False) as data:
            missing = [
                name
                for name in _STATE_ARRAYS
                if name not in data.files
            ]
            if missing:
                raise ValueError(
                    f"State NPZ is missing required arrays {missing}: "
                    f"{state_path}"
                )
            return {
                name: np.asarray(data[name]).copy()
                for name in _STATE_ARRAYS
            }

    def _compute_feature_statistics(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-feature statistics without materializing all graphs."""

        bus_moments = _RunningMoments(self.num_bus_features)
        branch_moments = _RunningMoments(self.num_branch_features)

        for idx in range(len(self.examples)):
            data = self._load_npz_by_index(idx)
            bus_moments.update(data["bus_features"])
            branch_moments.update(data["branch_features"])

        bus_mean, bus_std = bus_moments.finish()
        branch_mean, branch_std = branch_moments.finish()
        return bus_mean, bus_std, branch_mean, branch_std

    def _normalize_bus_features(
        self,
        bus_features: np.ndarray,
    ) -> np.ndarray:
        normalized = (
            bus_features.astype(np.float32)
            - self.bus_feature_mean
        ) / self.bus_feature_std

        return normalized.astype(np.float32)

    def _normalize_branch_features(
        self,
        branch_features: np.ndarray,
    ) -> np.ndarray:
        normalized = (
            branch_features.astype(np.float32)
            - self.branch_feature_mean
        ) / self.branch_feature_std

        return normalized.astype(np.float32)

    def _validate_graph_shapes(
            self,
            bus_features: np.ndarray,
            branch_features: np.ndarray,
            edge_index: np.ndarray,
            branch_status: np.ndarray,
            edge_active_mask: np.ndarray,
            action_mask: np.ndarray,
            state_path: Path,
    ) -> None:
        """
        Validate one graph independently of other dataset samples.

        Graph cardinality may vary between samples. Feature widths and the
        stop-plus-one-action-per-branch semantics remain fixed.
        """

        if (
                bus_features.ndim != 2
                or bus_features.shape[0] <= 0
        ):
            raise ValueError(
                f"{state_path}: bus_features must be "
                f"non-empty 2D, got "
                f"{bus_features.shape}"
            )

        if (
                branch_features.ndim != 2
                or branch_features.shape[0] <= 0
        ):
            raise ValueError(
                f"{state_path}: branch_features must be "
                f"non-empty 2D, got "
                f"{branch_features.shape}"
            )

        num_buses = int(bus_features.shape[0])
        num_branches = int(
            branch_features.shape[0]
        )

        if edge_index.shape != (
                2,
                num_branches,
        ):
            raise ValueError(
                f"{state_path}: edge_index must have "
                f"shape (2, num_branches), got "
                f"{edge_index.shape}"
            )

        if branch_status.shape != (
                num_branches,
        ):
            raise ValueError(
                f"{state_path}: branch_status must "
                f"have shape ({num_branches},), got "
                f"{branch_status.shape}"
            )

        if not np.isfinite(branch_status).all():
            raise ValueError(
                f"{state_path}: branch_status must "
                "contain only finite values"
            )

        if not np.isin(
                branch_status,
                (0.0, 1.0),
        ).all():
            raise ValueError(
                f"{state_path}: branch_status must "
                "contain only 0 or 1"
            )

        if edge_active_mask.shape != (
                num_branches,
        ):
            raise ValueError(
                f"{state_path}: edge_active_mask must "
                f"have shape ({num_branches},), got "
                f"{edge_active_mask.shape}"
            )

        expected_edge_active_mask = (
                branch_status > 0.5
        )

        if not np.array_equal(
                edge_active_mask,
                expected_edge_active_mask,
        ):
            raise ValueError(
                f"{state_path}: edge_active_mask must "
                "be derived from branch_status"
            )

        if action_mask.shape != (
                num_branches + 1,
        ):
            raise ValueError(
                f"{state_path}: action_mask must "
                "contain one stop action plus one "
                "action per branch. Expected "
                f"{num_branches + 1}, got "
                f"{action_mask.shape}"
            )

        if not bool(action_mask.any()):
            raise ValueError(
                f"{state_path}: action_mask must "
                "contain at least one legal action"
            )

        if (
                bus_features.shape[1]
                != self.num_bus_features
        ):
            raise ValueError(
                f"{state_path}: bus feature dim "
                f"mismatch. Expected "
                f"{self.num_bus_features}, got "
                f"{bus_features.shape[1]}"
            )

        if (
                branch_features.shape[1]
                != self.num_branch_features
        ):
            raise ValueError(
                f"{state_path}: branch feature dim "
                f"mismatch. Expected "
                f"{self.num_branch_features}, got "
                f"{branch_features.shape[1]}"
            )

        if not np.isfinite(edge_index).all():
            raise ValueError(
                f"{state_path}: edge_index must "
                "contain only finite values"
            )

        if not np.equal(
                edge_index,
                np.rint(edge_index),
        ).all():
            raise ValueError(
                f"{state_path}: edge_index must be "
                "integer-valued"
            )

        if (
                int(edge_index.min()) < 0
                or int(edge_index.max()) >= num_buses
        ):
            raise ValueError(
                f"{state_path}: edge_index values "
                "are out of bounds"
            )

    def normalization_state_dict(self) -> dict[str, np.ndarray]:
        """
        Return normalization arrays for saving into checkpoint.
        """

        return {
            "bus_feature_mean": self.bus_feature_mean.astype(
                np.float32,
                copy=True,
            ),
            "bus_feature_std": self.bus_feature_std.astype(
                np.float32,
                copy=True,
            ),
            "branch_feature_mean": self.branch_feature_mean.astype(
                np.float32,
                copy=True,
            ),
            "branch_feature_std": self.branch_feature_std.astype(
                np.float32,
                copy=True,
            ),
        }


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
    """Validate zero-based node positions used by the current graph contract."""

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

    if minimum < 0 or maximum >= num_nodes:
        raise ValueError(
            "edge_index contains node indices outside the graph: "
            f"min={minimum}, max={maximum}, num_nodes={num_nodes}."
        )

    return edge_index


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

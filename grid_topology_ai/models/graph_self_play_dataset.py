from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from grid_topology_ai.self_play.example_validation import REQUIRED_EXAMPLE_COLUMNS


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
    ):
        self.examples_csv = Path(examples_csv)
        self.normalize_features = bool(normalize_features)

        if not self.examples_csv.exists():
            raise FileNotFoundError(f"Examples file not found: {self.examples_csv}")

        self.examples = pd.read_csv(self.examples_csv)

        if self.examples.empty:
            raise ValueError("Graph self-play examples CSV is empty.")

        required_columns = set(REQUIRED_EXAMPLE_COLUMNS)

        missing = required_columns - set(self.examples.columns)

        if missing:
            raise ValueError(
                f"Examples CSV is missing required columns: {sorted(missing)}"
            )

        self._validate_outcome_value_targets()
        self._validate_state_files()

        first_data = self._load_npz_by_index(0)

        self.num_bus_features = int(first_data["bus_features"].shape[1])
        self.num_branch_features = int(first_data["branch_features"].shape[1])
        self.num_buses = int(first_data["bus_features"].shape[0])
        self.num_branches = int(first_data["branch_features"].shape[0])
        self.num_actions = int(first_data["action_mask"].shape[0])

        if self.num_actions != self.num_branches + 1:
            raise ValueError(
                f"Expected num_actions = num_branches + 1, got "
                f"num_actions={self.num_actions}, num_branches={self.num_branches}"
            )

        if normalization_stats is not None:
            self.bus_feature_mean = np.asarray(
                normalization_stats["bus_feature_mean"],
                dtype=np.float32,
            )
            self.bus_feature_std = np.asarray(
                normalization_stats["bus_feature_std"],
                dtype=np.float32,
            )
            self.branch_feature_mean = np.asarray(
                normalization_stats["branch_feature_mean"],
                dtype=np.float32,
            )
            self.branch_feature_std = np.asarray(
                normalization_stats["branch_feature_std"],
                dtype=np.float32,
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

        data = self._load_npz_by_index(idx)

        bus_features = data["bus_features"].astype(np.float32)
        branch_features = data["branch_features"].astype(np.float32)
        edge_index = data["edge_index"].astype(np.int64)
        action_mask = data["action_mask"].astype(bool)

        self._validate_graph_shapes(
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            action_mask=action_mask,
            state_path=Path(str(row["state_path"])),
        )

        bus_features = self._normalize_bus_features(bus_features)
        branch_features = self._normalize_branch_features(branch_features)

        target_policy = self._policy_json_to_vector(
            policy_json=str(row["mcts_policy_json"]),
            num_actions=int(action_mask.shape[0]),
        )

        # Safety: target policy must only assign probability to valid actions.
        target_policy = target_policy * action_mask.astype(np.float32)

        target_sum = float(target_policy.sum())

        if target_sum > 0.0:
            target_policy = target_policy / target_sum

        # Strict v3 logic:
        # The value head is trained only on outcome_value_target.
        # No value_target fallback and no discounted_return_from_step / value_scale fallback.
        target_value = float(row["outcome_value_target"])

        return {
            "bus_features": torch.tensor(bus_features, dtype=torch.float32),
            "branch_features": torch.tensor(branch_features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "action_mask": torch.tensor(action_mask, dtype=torch.bool),
            "target_policy": torch.tensor(target_policy, dtype=torch.float32),
            "target_value": torch.tensor(target_value, dtype=torch.float32),
            "scenario_id": int(row["scenario_id"]),
            "step": int(row["step"]),
            "state_id": str(row["state_id"]),
        }

    def _validate_outcome_value_targets(self) -> None:
        """
        Validate strict AlphaZero-like value targets.

        New datasets must contain outcome_value_target for every row.
        Legacy fallback to discounted_return_from_step / value_scale is intentionally removed.
        """

        values = pd.to_numeric(
            self.examples["outcome_value_target"],
            errors="coerce",
        )

        invalid_mask = values.isna() | ~np.isfinite(values.to_numpy(dtype=np.float64))

        if bool(invalid_mask.any()):
            bad_count = int(invalid_mask.sum())
            raise ValueError(
                f"{bad_count} rows in {self.examples_csv} have invalid "
                f"'outcome_value_target'. Regenerate the dataset with the new "
                f"teacher generator."
            )

        outside_mask = values.abs() > 1.0 + 1e-6

        if bool(outside_mask.any()):
            bad_count = int(outside_mask.sum())
            min_value = float(values.min())
            max_value = float(values.max())

            raise ValueError(
                f"{bad_count} rows in {self.examples_csv} have "
                f"'outcome_value_target' outside [-1, 1]. "
                f"Observed range: [{min_value:.6f}, {max_value:.6f}]."
            )

    def _validate_state_files(self) -> None:
        """
        Check that all referenced .npz state files exist.
        """

        for _, row in self.examples.iterrows():
            state_path = Path(str(row["state_path"]))

            if not state_path.exists():
                raise FileNotFoundError(f"State file not found: {state_path}")

    def _load_npz_by_index(self, idx: int):
        row = self.examples.iloc[idx]
        state_path = Path(str(row["state_path"]))

        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")

        return np.load(state_path, allow_pickle=False)

    def _compute_feature_statistics(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-feature normalization statistics.

        For graph model:
        - bus_feature_mean/std are computed over all buses in all examples;
        - branch_feature_mean/std are computed over all branches in all examples.

        This is better than using one global flat vector because each node/edge
        feature keeps its own physical meaning.
        """

        bus_chunks = []
        branch_chunks = []

        for idx in range(len(self.examples)):
            data = self._load_npz_by_index(idx)

            bus_features = data["bus_features"].astype(np.float32)
            branch_features = data["branch_features"].astype(np.float32)

            bus_chunks.append(bus_features)
            branch_chunks.append(branch_features)

        all_bus = np.concatenate(bus_chunks, axis=0)
        all_branch = np.concatenate(branch_chunks, axis=0)

        bus_mean = all_bus.mean(axis=0).astype(np.float32)
        bus_std = all_bus.std(axis=0).astype(np.float32)

        branch_mean = all_branch.mean(axis=0).astype(np.float32)
        branch_std = all_branch.std(axis=0).astype(np.float32)

        bus_std[bus_std < 1e-6] = 1.0
        branch_std[branch_std < 1e-6] = 1.0

        return bus_mean, bus_std, branch_mean, branch_std

    def _normalize_bus_features(
        self,
        bus_features: np.ndarray,
    ) -> np.ndarray:
        normalized = (
            bus_features.astype(np.float32) - self.bus_feature_mean
        ) / self.bus_feature_std

        return normalized.astype(np.float32)

    def _normalize_branch_features(
        self,
        branch_features: np.ndarray,
    ) -> np.ndarray:
        normalized = (
            branch_features.astype(np.float32) - self.branch_feature_mean
        ) / self.branch_feature_std

        return normalized.astype(np.float32)

    def _validate_graph_shapes(
        self,
        bus_features: np.ndarray,
        branch_features: np.ndarray,
        edge_index: np.ndarray,
        action_mask: np.ndarray,
        state_path: Path,
    ) -> None:
        """
        Ensure every sample has the same fixed graph size.

        This is true for our current case118 dataset and allows the standard
        PyTorch DataLoader to stack samples into batches automatically.
        """

        if bus_features.ndim != 2:
            raise ValueError(
                f"{state_path}: bus_features must be 2D, got {bus_features.shape}"
            )

        if branch_features.ndim != 2:
            raise ValueError(
                f"{state_path}: branch_features must be 2D, got {branch_features.shape}"
            )

        if edge_index.shape != (2, branch_features.shape[0]):
            raise ValueError(
                f"{state_path}: edge_index must have shape "
                f"(2, num_branches), got {edge_index.shape}"
            )

        if action_mask.ndim != 1:
            raise ValueError(
                f"{state_path}: action_mask must be 1D, got {action_mask.shape}"
            )

        if bus_features.shape[1] != self.num_bus_features:
            raise ValueError(
                f"{state_path}: bus feature dim mismatch. "
                f"Expected {self.num_bus_features}, got {bus_features.shape[1]}"
            )

        if branch_features.shape[1] != self.num_branch_features:
            raise ValueError(
                f"{state_path}: branch feature dim mismatch. "
                f"Expected {self.num_branch_features}, got {branch_features.shape[1]}"
            )

        if bus_features.shape[0] != self.num_buses:
            raise ValueError(
                f"{state_path}: num_buses mismatch. "
                f"Expected {self.num_buses}, got {bus_features.shape[0]}"
            )

        if branch_features.shape[0] != self.num_branches:
            raise ValueError(
                f"{state_path}: num_branches mismatch. "
                f"Expected {self.num_branches}, got {branch_features.shape[0]}"
            )

        expected_num_actions = self.num_branches + 1

        if action_mask.shape[0] != expected_num_actions:
            raise ValueError(
                f"{state_path}: action_mask mismatch. "
                f"Expected {expected_num_actions}, got {action_mask.shape[0]}"
            )

    @staticmethod
    def _policy_json_to_vector(
        policy_json: str,
        num_actions: int,
    ) -> np.ndarray:
        """
        Convert MCTS/teacher policy JSON into dense vector.
        """

        policy_dict = json.loads(policy_json)

        policy = np.zeros(num_actions, dtype=np.float32)

        for action_id_str, probability in policy_dict.items():
            action_id = int(action_id_str)

            if 0 <= action_id < num_actions:
                policy[action_id] = float(probability)

        total = float(policy.sum())

        if total > 0.0:
            policy = policy / total

        return policy.astype(np.float32)

    def normalization_state_dict(self) -> dict[str, np.ndarray]:
        """
        Return normalization arrays for saving into checkpoint.
        """

        return {
            "bus_feature_mean": self.bus_feature_mean.astype(np.float32),
            "bus_feature_std": self.bus_feature_std.astype(np.float32),
            "branch_feature_mean": self.branch_feature_mean.astype(np.float32),
            "branch_feature_std": self.branch_feature_std.astype(np.float32),
        }
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SelfPlayDataset(Dataset):
    """
    Dataset for AlphaZero-like training.

    One sample contains:
        state_vector
        action_mask
        target_policy
        target_value

    target_policy:
        MCTS visit-count policy.

    target_value:
        discounted return from this step.
    """

    def __init__(
        self,
        examples_csv: str | Path,
        value_scale: float = 1000.0,
    ):
        self.examples_csv = Path(examples_csv)
        self.value_scale = float(value_scale)

        if not self.examples_csv.exists():
            raise FileNotFoundError(f"Examples file not found: {self.examples_csv}")

        self.examples = pd.read_csv(self.examples_csv)

        if self.examples.empty:
            raise ValueError("Self-play examples CSV is empty.")

        # Precompute normalization statistics for the flat state vectors.
        # This is important because raw features have very different scales:
        # MW, MVar, p.u., percent loading, resistance/reactance, etc.
        raw_vectors = []

        for _, row in self.examples.iterrows():
            state_path = Path(str(row["state_path"]))

            if not state_path.exists():
                raise FileNotFoundError(f"State file not found: {state_path}")

            data = np.load(state_path, allow_pickle=False)

            bus_features = data["bus_features"].astype(np.float32)
            branch_features = data["branch_features"].astype(np.float32)

            raw_vectors.append(
                self._make_flat_state_vector(
                    bus_features=bus_features,
                    branch_features=branch_features,
                )
            )

        matrix = np.stack(raw_vectors, axis=0)

        self.state_mean = matrix.mean(axis=0).astype(np.float32)
        self.state_std = matrix.std(axis=0).astype(np.float32)

        # Avoid division by zero for constant features.
        self.state_std[self.state_std < 1e-6] = 1.0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        row = self.examples.iloc[idx]

        state_path = Path(str(row["state_path"]))

        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")

        data = np.load(state_path, allow_pickle=False)

        bus_features = data["bus_features"].astype(np.float32)
        branch_features = data["branch_features"].astype(np.float32)
        action_mask = data["action_mask"].astype(bool)

        num_actions = int(action_mask.shape[0])

        target_policy = self._policy_json_to_vector(
            policy_json=str(row["mcts_policy_json"]),
            num_actions=num_actions,
        )

        # Safety: target policy should only assign probability to valid actions.
        # If anything becomes inconsistent after future code changes,
        # this prevents masked invalid actions from creating huge policy loss.
        target_policy = target_policy * action_mask.astype(np.float32)

        target_sum = float(target_policy.sum())

        if target_sum > 0:
            target_policy = target_policy / target_sum

        raw_value = float(row["discounted_return_from_step"])

        target_value = raw_value / self.value_scale
        target_value = float(np.clip(target_value, -1.0, 1.0))

        state_vector = self._make_flat_state_vector(
            bus_features=bus_features,
            branch_features=branch_features,
        )

        state_vector = (state_vector - self.state_mean) / self.state_std
        state_vector = state_vector.astype(np.float32)

        return {
            "state_vector": torch.tensor(state_vector, dtype=torch.float32),
            "action_mask": torch.tensor(action_mask, dtype=torch.bool),
            "target_policy": torch.tensor(target_policy, dtype=torch.float32),
            "target_value": torch.tensor(target_value, dtype=torch.float32),
            "scenario_id": int(row["scenario_id"]),
            "step": int(row["step"]),
            "state_id": str(row["state_id"]),
        }

    @staticmethod
    def _policy_json_to_vector(
        policy_json: str,
        num_actions: int,
    ) -> np.ndarray:
        """
        Convert MCTS policy JSON into dense policy vector.
        """

        policy_dict = json.loads(policy_json)

        policy = np.zeros(num_actions, dtype=np.float32)

        for action_id_str, probability in policy_dict.items():
            action_id = int(action_id_str)

            if 0 <= action_id < num_actions:
                policy[action_id] = float(probability)

        total = float(policy.sum())

        if total > 0:
            policy = policy / total

        return policy.astype(np.float32)

    @staticmethod
    def _make_flat_state_vector(
        bus_features: np.ndarray,
        branch_features: np.ndarray,
    ) -> np.ndarray:
        """
        Temporary non-GNN state representation.

        Later this will be replaced by a graph neural network.

        Current representation:
            mean, std, min, max of bus features
            mean, std, min, max of branch features
        """

        bus_mean = bus_features.mean(axis=0)
        bus_std = bus_features.std(axis=0)
        bus_min = bus_features.min(axis=0)
        bus_max = bus_features.max(axis=0)

        branch_mean = branch_features.mean(axis=0)
        branch_std = branch_features.std(axis=0)
        branch_min = branch_features.min(axis=0)
        branch_max = branch_features.max(axis=0)

        state_vector = np.concatenate(
            [
                bus_mean,
                bus_std,
                bus_min,
                bus_max,
                branch_mean,
                branch_std,
                branch_min,
                branch_max,
            ],
            axis=0,
        )

        return state_vector.astype(np.float32)
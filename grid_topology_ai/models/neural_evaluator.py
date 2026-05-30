from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.models.self_play_dataset import SelfPlayDataset
from grid_topology_ai.models.simple_policy_value_net import SimplePolicyValueNet


class NeuralPolicyValueEvaluator:
    """
    Wrapper for using a trained policy-value network inside MCTS.

    Input:
        GridFMState + action_mask

    Output:
        policy probabilities over actions
        value estimate in normalized scale [-1, 1]
    """

    def __init__(
            self,
            checkpoint_path: str | Path,
            device: str = "cpu",
            enable_cache: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.enable_cache = bool(enable_cache)

        self._cache: dict[tuple, tuple[np.ndarray, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.input_dim = int(checkpoint["input_dim"])
        self.num_actions = int(checkpoint["num_actions"])
        self.hidden_dim = int(checkpoint["hidden_dim"])
        self.value_scale = float(checkpoint.get("value_scale", 1000.0))

        self.state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)

        self.state_std[self.state_std < 1e-6] = 1.0

        self.model = SimplePolicyValueNet(
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            hidden_dim=self.hidden_dim,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "enabled": self.enable_cache,
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    def _make_cache_key(
            self,
            state: GridFMState,
            action_mask: np.ndarray,
    ) -> tuple:
        return (
            int(state.scenario_id),
            tuple(int(x) for x in sorted(state.outaged_branch_ids)),
            action_mask.astype(bool).tobytes(),
        )

    def evaluate(
        self,
        state: GridFMState,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Evaluate one state.

        Returns
        -------
        policy:
            Dense probability vector of shape [num_actions].

        value:
            Normalized scalar value in [-1, 1].
        """

        cache_key = self._make_cache_key(
            state=state,
            action_mask=action_mask,
        )

        if self.enable_cache and cache_key in self._cache:
            self.cache_hits += 1

            cached_policy, cached_value = self._cache[cache_key]

            return cached_policy.copy(), float(cached_value)

        if self.enable_cache:
            self.cache_misses += 1

        state_vector = SelfPlayDataset._make_flat_state_vector(
            bus_features=state.bus_features.astype(np.float32),
            branch_features=state.branch_features.astype(np.float32),
        )

        state_vector = (state_vector - self.state_mean) / self.state_std
        state_vector = state_vector.astype(np.float32)

        if action_mask.shape[0] != self.num_actions:
            raise ValueError(
                f"Action mask size mismatch: expected {self.num_actions}, "
                f"got {action_mask.shape[0]}"
            )

        state_tensor = torch.tensor(
            state_vector,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            action_mask.astype(bool),
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(
                state_vector=state_tensor,
                action_mask=mask_tensor,
            )

            policy = torch.softmax(logits, dim=1)[0].cpu().numpy()
            value_float = float(value.item())

        # Numerical safety.
        policy = policy.astype(np.float32)
        policy = policy * action_mask.astype(np.float32)

        total = float(policy.sum())

        if total > 0:
            policy = policy / total
        else:
            valid = action_mask.astype(bool)
            policy = np.zeros_like(policy, dtype=np.float32)
            policy[valid] = 1.0 / max(int(valid.sum()), 1)

        if self.enable_cache:
            self._cache[cache_key] = (policy.copy(), float(value_float))

        return policy, value_float
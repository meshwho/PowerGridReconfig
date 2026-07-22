from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import require_checkpoint_contracts
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.models.graph_policy_value_net import GraphPolicyValueNet
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.self_play_dataset import SelfPlayDataset
from grid_topology_ai.models.simple_policy_value_net import SimplePolicyValueNet


class NeuralPolicyValueEvaluator:
    """
    Wrapper for using a trained policy-value network inside MCTS.

    Supported checkpoint types:
        1. simple_policy_value_net / MLP checkpoint
        2. graph_policy_value_net / graph checkpoint

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
        physics_config: PhysicsConfig | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.enable_cache = bool(enable_cache)

        self._cache: dict[tuple, tuple[np.ndarray, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(self.checkpoint, dict):
            raise ValueError(
                f"Checkpoint payload must be a mapping: {self.checkpoint_path}"
            )
        self.physics_config = require_checkpoint_contracts(
            self.checkpoint,
            source=str(self.checkpoint_path),
            expected_physics_config=physics_config,
        )

        self.model_type = str(
            self.checkpoint.get("model_type", "simple_policy_value_net")
        )

        if self.model_type in {
            "graph_policy_value_net",
            "graph_policy_value_net_v2",
        }:
            self._init_graph_model()
        elif self.model_type in {
            "simple_policy_value_net",
            "mlp",
            "simple_mlp",
        }:
            self._init_mlp_model()
        else:
            raise ValueError(
                f"Unsupported checkpoint model_type={self.model_type!r}. "
                "Expected 'simple_policy_value_net', 'graph_policy_value_net' or 'graph_policy_value_net_v2'."
            )

        self.model.to(self.device)
        self.model.eval()

    def _init_mlp_model(self) -> None:
        """
        Initialize old flat-vector MLP evaluator.
        """

        checkpoint = self.checkpoint

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

    def _init_graph_model(self) -> None:
        """
        Initialize graph/GNN evaluator.
        """

        checkpoint = self.checkpoint

        self.num_bus_features = int(checkpoint["num_bus_features"])
        self.num_branch_features = int(checkpoint["num_branch_features"])
        self.num_buses = int(checkpoint["num_buses"])
        self.num_branches = int(checkpoint["num_branches"])
        self.num_actions = int(checkpoint["num_actions"])
        self.hidden_dim = int(checkpoint["hidden_dim"])
        self.num_layers = int(checkpoint["num_layers"])
        self.dropout = float(checkpoint.get("dropout", 0.0))
        self.value_scale = float(checkpoint.get("value_scale", 10000.0))

        self.bus_feature_mean = np.asarray(
            checkpoint["bus_feature_mean"],
            dtype=np.float32,
        )
        self.bus_feature_std = np.asarray(
            checkpoint["bus_feature_std"],
            dtype=np.float32,
        )
        self.branch_feature_mean = np.asarray(
            checkpoint["branch_feature_mean"],
            dtype=np.float32,
        )
        self.branch_feature_std = np.asarray(
            checkpoint["branch_feature_std"],
            dtype=np.float32,
        )

        self.bus_feature_std[self.bus_feature_std < 1e-6] = 1.0
        self.branch_feature_std[self.branch_feature_std < 1e-6] = 1.0

        if self.model_type == "graph_policy_value_net_v2":
            model_cls = GraphPolicyValueNetV2
        else:
            model_cls = GraphPolicyValueNet

        self.model = model_cls(
            num_bus_features=self.num_bus_features,
            num_branch_features=self.num_branch_features,
            num_actions=self.num_actions,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "enabled": self.enable_cache,
            "model_type": self.model_type,
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
        """
        Cache key for neural evaluations.

        branch_status is included because the same scenario can appear in
        different topology states after several switching actions.
        """

        return (
            self.model_type,
            int(state.scenario_id),
            tuple(int(x) for x in sorted(state.outaged_branch_ids)),
            state.branch_status.astype(np.int8).tobytes(),
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

        if action_mask.shape[0] != self.num_actions:
            raise ValueError(
                f"Action mask size mismatch: expected {self.num_actions}, "
                f"got {action_mask.shape[0]}"
            )

        if self.model_type in {
            "graph_policy_value_net",
            "graph_policy_value_net_v2",
        }:
            policy, value_float = self._evaluate_graph(
                state=state,
                action_mask=action_mask,
            )
        else:
            policy, value_float = self._evaluate_mlp(
                state=state,
                action_mask=action_mask,
            )

        policy = self._sanitize_policy(
            policy=policy,
            action_mask=action_mask,
        )

        if self.enable_cache:
            self._cache[cache_key] = (policy.copy(), float(value_float))

        return policy, float(value_float)

    def _evaluate_mlp(
        self,
        state: GridFMState,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Evaluate state with old flat-vector MLP.
        """

        state_vector = SelfPlayDataset._make_flat_state_vector(
            bus_features=state.bus_features.astype(np.float32),
            branch_features=state.branch_features.astype(np.float32),
        )

        state_vector = (state_vector - self.state_mean) / self.state_std
        state_vector = state_vector.astype(np.float32)

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

            policy = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            value_float = float(value.detach().cpu().item())

        return policy.astype(np.float32), value_float

    def _evaluate_graph(
        self,
        state: GridFMState,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Evaluate state with graph/GNN model.
        """

        bus_features = state.bus_features.astype(np.float32)
        branch_features = state.branch_features.astype(np.float32)
        edge_index = state.edge_index.astype(np.int64)
        branch_status = np.asarray(
            state.branch_status,
            dtype=np.float32,
        )
        edge_active_mask = branch_status > 0.5

        if bus_features.shape != (self.num_buses, self.num_bus_features):
            raise ValueError(
                f"bus_features shape mismatch: expected "
                f"({self.num_buses}, {self.num_bus_features}), "
                f"got {bus_features.shape}"
            )

        if branch_features.shape != (
            self.num_branches,
            self.num_branch_features,
        ):
            raise ValueError(
                f"branch_features shape mismatch: expected "
                f"({self.num_branches}, {self.num_branch_features}), "
                f"got {branch_features.shape}"
            )

        if edge_index.shape != (2, self.num_branches):
            raise ValueError(
                f"edge_index shape mismatch: expected "
                f"(2, {self.num_branches}), got {edge_index.shape}"
            )

        if branch_status.shape != (self.num_branches,):
            raise ValueError(
                "branch_status shape mismatch: expected "
                f"({self.num_branches},), got {branch_status.shape}"
            )

        if not np.isfinite(branch_status).all():
            raise ValueError(
                "branch_status must contain only finite values"
            )

        if not np.isin(branch_status, (0.0, 1.0)).all():
            raise ValueError(
                "branch_status must contain only 0 or 1"
            )

        bus_features = (
            bus_features - self.bus_feature_mean
        ) / self.bus_feature_std

        branch_features = (
            branch_features - self.branch_feature_mean
        ) / self.branch_feature_std

        bus_tensor = torch.tensor(
            bus_features.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        branch_tensor = torch.tensor(
            branch_features.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        edge_index_tensor = torch.tensor(
            edge_index,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        edge_active_mask_tensor = torch.tensor(
            edge_active_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            action_mask.astype(bool),
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            if self.model_type == "graph_policy_value_net_v2":
                logits, value = self.model(
                    bus_features=bus_tensor,
                    branch_features=branch_tensor,
                    edge_index=edge_index_tensor,
                    edge_active_mask=edge_active_mask_tensor,
                    action_mask=mask_tensor,
                )
            else:
                logits, value = self.model(
                    bus_features=bus_tensor,
                    branch_features=branch_tensor,
                    edge_index=edge_index_tensor,
                    action_mask=mask_tensor,
                )

            policy = torch.softmax(
                logits,
                dim=1,
            )[0].detach().cpu().numpy()

            value_float = float(
                value.detach().cpu().item()
            )

        return policy.astype(np.float32), value_float

    @staticmethod
    def _sanitize_policy(
        policy: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Numerical safety:
        - cast to float32;
        - remove invalid actions;
        - renormalize;
        - fallback to uniform over valid actions if needed.
        """

        policy = policy.astype(np.float32)
        policy = policy * action_mask.astype(np.float32)

        total = float(policy.sum())

        if total > 0:
            policy = policy / total
        else:
            valid = action_mask.astype(bool)
            policy = np.zeros_like(policy, dtype=np.float32)
            policy[valid] = 1.0 / max(int(valid.sum()), 1)

        return policy.astype(np.float32)

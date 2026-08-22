from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from grid_topology_ai.config import (
    PhysicsConfig,
    require_physics_config_payload,
)
from grid_topology_ai.model import GraphPolicyValueNetV2
from grid_topology_ai.state import GridFMState
from grid_topology_ai.actions import (
    STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    action_layout_fingerprint,
    build_branch_action_slots,
    require_topology_action_payload,
)


def require_graph_batching_checkpoint_contract(
    payload: Mapping[str, object],
    *,
    source: str,
) -> None:
    model_type = str(payload.get("model_type", "")).strip()
    if model_type != "graph_policy_value_net_v2":
        return

    if payload.get("topology_cardinality_independent") is not True:
        raise ValueError(
            "Graph V2 checkpoint must declare "
            f"topology_cardinality_independent=True for {source}."
        )


def _require_graph_checkpoint_feature_dimensions(
    payload: Mapping[str, object],
    *,
    source: str,
) -> None:
    import numpy as np

    from grid_topology_ai.state import (
        BRANCH_FEATURE_COLUMNS,
        BUS_FEATURE_COLUMNS,
    )

    model_type = str(payload.get("model_type", ""))
    if model_type not in {
        "graph_policy_value_net",
        "graph_policy_value_net_v2",
    }:
        return

    expected_dimensions = {
        "num_bus_features": len(BUS_FEATURE_COLUMNS),
        "num_branch_features": len(BRANCH_FEATURE_COLUMNS),
    }
    for key, expected in expected_dimensions.items():
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"Graph checkpoint {source} is missing exact integer {key}."
            )
        if int(value) != expected:
            raise ValueError(
                f"Graph checkpoint {key} mismatch for {source}: "
                f"expected {expected}, observed {int(value)}."
            )

    for key, expected in (
        ("bus_feature_mean", len(BUS_FEATURE_COLUMNS)),
        ("bus_feature_std", len(BUS_FEATURE_COLUMNS)),
        ("branch_feature_mean", len(BRANCH_FEATURE_COLUMNS)),
        ("branch_feature_std", len(BRANCH_FEATURE_COLUMNS)),
    ):
        value = payload.get(key)
        try:
            size = len(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                f"Graph checkpoint is missing normalization vector "
                f"{key} for {source}."
            ) from exc
        if size != expected:
            raise ValueError(
                f"Graph checkpoint normalization vector {key} mismatch "
                f"for {source}: expected {expected}, observed {size}."
            )
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(
                f"Graph checkpoint normalization vector {key} is not finite for {source}."
            )
        if key.endswith("_std") and not (array > 0.0).all():
            raise ValueError(
                f"Graph checkpoint normalization vector {key} must be positive for {source}."
            )


def require_checkpoint_contracts(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_physics_config: "PhysicsConfig | None" = None,
) -> "PhysicsConfig":
    """Validate the semantic facts needed to reconstruct a current Graph V2."""

    if payload.get("model_type") != "graph_policy_value_net_v2":
        raise ValueError(f"Unsupported graph checkpoint model_type for {source}.")
    if payload.get("topology_cardinality_independent") is not True:
        raise ValueError(
            f"Graph checkpoint must be topology-cardinality independent for {source}."
        )
    if payload.get("policy_layout") != "stop_plus_branch_status_v1":
        raise ValueError(f"Unsupported graph checkpoint policy_layout for {source}.")
    if "model_state_dict" not in payload:
        raise ValueError(f"Graph checkpoint is missing model_state_dict for {source}.")
    for key in ("hidden_dim", "num_layers"):
        if int(payload.get(key, 0)) <= 0:
            raise ValueError(f"Graph checkpoint has invalid {key} for {source}.")
    dropout = float(payload.get("dropout", -1.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"Graph checkpoint has invalid dropout for {source}.")
    require_topology_action_payload(
        payload,
        source=source,
    )
    physics_config = require_physics_config_payload(
        payload,
        source=source,
        expected_physics_config=expected_physics_config,
    )

    _require_graph_checkpoint_feature_dimensions(
        payload,
        source=source,
    )
    return physics_config


_FINGERPRINT_VERSION = b"physical-state-v1"


def _update_fingerprint_array(
    digest,
    *,
    name: str,
    value: object,
    dtype: str,
) -> None:
    array = np.asarray(value, dtype=np.dtype(dtype))
    array = np.ascontiguousarray(array)

    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))


def physical_state_fingerprint(state: GridFMState) -> str:
    """Return the stable physical-state cache key used by this evaluator."""

    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_VERSION)
    digest.update(b"\0")

    for name, value, dtype in (
        ("scenario_id", [state.scenario_id], "<i8"),
        ("load_scenario_idx", [state.load_scenario_idx], "<f8"),
        ("bus_features", state.bus_features, "<f4"),
        ("branch_features", state.branch_features, "<f4"),
        ("edge_index", state.edge_index, "<i8"),
    ):
        _update_fingerprint_array(
            digest, name=name, value=value, dtype=dtype
        )

    if state.bus_ids is None:
        digest.update(b"bus_ids\0none\0")
    else:
        _update_fingerprint_array(
            digest, name="bus_ids", value=state.bus_ids, dtype="<i8"
        )

    for name, value, dtype in (
        ("branch_ids", state.branch_ids, "<i8"),
        ("branch_status", state.branch_status, "<f4"),
        (
            "outaged_branch_ids",
            sorted(int(value) for value in state.outaged_branch_ids),
            "<i8",
        ),
    ):
        _update_fingerprint_array(
            digest, name=name, value=value, dtype=dtype
        )

    return digest.hexdigest()

_MODEL_TYPE = "graph_policy_value_net_v2"


class NeuralPolicyValueEvaluator:
    """Use a trained Graph V2 policy-value network inside MCTS."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        enable_cache: bool = True,
        physics_config: PhysicsConfig | None = None,
    ) -> None:
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
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Checkpoint payload must be a mapping: {self.checkpoint_path}"
            )
        self.checkpoint = checkpoint

        self.physics_config = require_checkpoint_contracts(
            checkpoint,
            source=str(self.checkpoint_path),
            expected_physics_config=physics_config,
        )
        self.topology_action_config, _ = require_topology_action_payload(
            checkpoint,
            source=str(self.checkpoint_path),
        )

        self.policy_layout = str(checkpoint.get("policy_layout", ""))
        if self.policy_layout != STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT:
            raise ValueError(
                "Unsupported checkpoint policy layout: "
                f"{self.policy_layout!r}."
            )

        self.model_type = str(checkpoint.get("model_type", ""))
        if self.model_type != _MODEL_TYPE:
            raise ValueError(
                f"Unsupported checkpoint model_type={self.model_type!r}. "
                f"Expected {_MODEL_TYPE!r}."
            )

        self.num_bus_features = int(checkpoint["num_bus_features"])
        self.num_branch_features = int(checkpoint["num_branch_features"])
        hidden_dim = int(checkpoint["hidden_dim"])
        num_layers = int(checkpoint["num_layers"])
        dropout = float(checkpoint.get("dropout", 0.0))

        self.bus_feature_mean = np.asarray(
            checkpoint["bus_feature_mean"], dtype=np.float32
        )
        self.bus_feature_std = np.asarray(
            checkpoint["bus_feature_std"], dtype=np.float32
        )
        self.branch_feature_mean = np.asarray(
            checkpoint["branch_feature_mean"], dtype=np.float32
        )
        self.branch_feature_std = np.asarray(
            checkpoint["branch_feature_std"], dtype=np.float32
        )
        self.bus_feature_std[self.bus_feature_std < 1e-6] = 1.0
        self.branch_feature_std[self.branch_feature_std < 1e-6] = 1.0

        self.model = GraphPolicyValueNetV2(
            num_bus_features=self.num_bus_features,
            num_branch_features=self.num_branch_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self) -> dict[str, object]:
        total = self.cache_hits + self.cache_misses
        return {
            "enabled": self.enable_cache,
            "model_type": self.model_type,
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }

    def _make_cache_key(
        self,
        state: GridFMState,
        action_mask: np.ndarray,
    ) -> tuple:
        mask = np.asarray(action_mask, dtype=np.bool_)
        return (
            self.physics_config.fingerprint(),
            action_layout_fingerprint(build_branch_action_slots(state.branch_ids)),
            physical_state_fingerprint(state),
            mask.shape,
            mask.tobytes(order="C"),
        )

    def evaluate(
        self,
        state: GridFMState,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if action_mask.ndim != 1:
            raise ValueError(f"Action mask must be 1D, got {action_mask.shape}.")

        expected_num_actions = len(state.branch_ids) + 1
        if action_mask.shape[0] != expected_num_actions:
            raise ValueError(
                "Action mask size mismatch: "
                f"expected {expected_num_actions}, got {action_mask.shape[0]}."
            )

        cache_key = self._make_cache_key(state, action_mask)
        if self.enable_cache and cache_key in self._cache:
            self.cache_hits += 1
            policy, value = self._cache[cache_key]
            return policy.copy(), float(value)

        if self.enable_cache:
            self.cache_misses += 1

        policy, value = self._evaluate_graph(state, action_mask)
        policy = self._sanitize_policy(policy, action_mask)

        if self.enable_cache:
            self._cache[cache_key] = (policy.copy(), float(value))

        return policy, float(value)

    def _evaluate_graph(
        self,
        state: GridFMState,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        bus_features = np.asarray(state.bus_features, dtype=np.float32)
        branch_features = np.asarray(state.branch_features, dtype=np.float32)
        edge_index = np.asarray(state.edge_index, dtype=np.int64)
        branch_status = np.asarray(state.branch_status, dtype=np.float32)

        if bus_features.ndim != 2 or bus_features.shape[0] <= 0:
            raise ValueError(
                "bus_features must be a non-empty 2D array, "
                f"got {bus_features.shape}."
            )
        if branch_features.ndim != 2 or branch_features.shape[0] <= 0:
            raise ValueError(
                "branch_features must be a non-empty 2D array, "
                f"got {branch_features.shape}."
            )

        num_branches = int(branch_features.shape[0])
        if bus_features.shape[1] != self.num_bus_features:
            raise ValueError(
                "bus feature width mismatch: "
                f"expected {self.num_bus_features}, got {bus_features.shape[1]}."
            )
        if branch_features.shape[1] != self.num_branch_features:
            raise ValueError(
                "branch feature width mismatch: "
                f"expected {self.num_branch_features}, got {branch_features.shape[1]}."
            )
        if edge_index.shape != (2, num_branches):
            raise ValueError(
                "edge_index shape mismatch: "
                f"expected {(2, num_branches)}, got {edge_index.shape}."
            )
        if branch_status.shape != (num_branches,):
            raise ValueError(
                "branch_status shape mismatch: "
                f"expected {(num_branches,)}, got {branch_status.shape}."
            )
        if not np.isfinite(branch_status).all() or not np.isin(
            branch_status, (0.0, 1.0)
        ).all():
            raise ValueError("branch_status must contain only finite 0/1 values.")

        bus_features = (
            bus_features - self.bus_feature_mean
        ) / self.bus_feature_std
        branch_features = (
            branch_features - self.branch_feature_mean
        ) / self.branch_feature_std

        bus_tensor = torch.as_tensor(
            bus_features, dtype=torch.float32, device=self.device
        )
        branch_tensor = torch.as_tensor(
            branch_features, dtype=torch.float32, device=self.device
        )
        edge_index_tensor = torch.as_tensor(
            edge_index, dtype=torch.long, device=self.device
        )
        edge_active_mask = torch.as_tensor(
            branch_status > 0.5, dtype=torch.bool, device=self.device
        )
        mask_tensor = torch.as_tensor(
            action_mask.astype(bool), dtype=torch.bool, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(
                bus_features=bus_tensor,
                branch_features=branch_tensor,
                edge_index=edge_index_tensor,
                edge_active_mask=edge_active_mask,
                action_mask=mask_tensor,
            )
            policy = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            value_float = float(value.detach().cpu().item())

        return policy.astype(np.float32), value_float

    @staticmethod
    def _sanitize_policy(
        policy: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        policy = policy.astype(np.float32)
        policy *= action_mask.astype(np.float32)
        total = float(policy.sum())
        if total > 0.0:
            policy /= total
            return policy

        valid = action_mask.astype(bool)
        policy = np.zeros_like(policy, dtype=np.float32)
        policy[valid] = 1.0 / max(int(valid.sum()), 1)
        return policy

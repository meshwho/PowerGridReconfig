from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import require_topology_action_provenance
from grid_topology_ai.models.graph_policy_value_net import GraphPolicyValueNet
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.self_play.artifacts import sha256_file
from grid_topology_ai.self_play.replay_error_sampling import (
    PREDICTION_ERROR_SCHEMA_VERSION,
)
from grid_topology_ai.training.checkpoints import (
    extract_normalization_stats,
    load_checkpoint_payload,
)

GraphModel = GraphPolicyValueNet | GraphPolicyValueNetV2


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> GraphModel:
    model_type = str(checkpoint.get("model_type", "")).strip()
    common_kwargs = {
        "num_bus_features": int(checkpoint["num_bus_features"]),
        "num_branch_features": int(checkpoint["num_branch_features"]),
        "num_actions": int(checkpoint["num_actions"]),
        "hidden_dim": int(checkpoint.get("hidden_dim", 128)),
        "num_layers": int(checkpoint.get("num_layers", 3)),
        "dropout": float(checkpoint.get("dropout", 0.0)),
    }

    if model_type in {"graph_v2", "graph_policy_value_net_v2"}:
        model: GraphModel = GraphPolicyValueNetV2(**common_kwargs)
    elif model_type in {"graph_v1", "graph_policy_value_net"}:
        model = GraphPolicyValueNet(**common_kwargs)
    else:
        raise ValueError(
            "Replay priority scoring requires a graph policy-value checkpoint, "
            f"got model_type={model_type!r}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _forward(
    model: GraphModel,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {
        "bus_features": batch["bus_features"],
        "branch_features": batch["branch_features"],
        "edge_index": batch["edge_index"],
        "action_mask": batch["action_mask"],
    }
    if isinstance(model, GraphPolicyValueNetV2):
        kwargs["edge_active_mask"] = batch["edge_active_mask"]
    return model(**kwargs)


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for name, value in batch.items()
    }


def score_replay_prediction_errors(
    *,
    examples_csv: str | Path,
    checkpoint_path: str | Path,
    physics_config: PhysicsConfig,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Score replay examples against one checkpoint without mutating them."""

    examples_csv = Path(examples_csv)
    checkpoint_path = Path(checkpoint_path)
    device = _resolve_device()
    checkpoint = dict(
        load_checkpoint_payload(
            checkpoint_path,
            map_location=device,
            expected_physics_config=physics_config,
        )
    )
    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
        normalization_stats=extract_normalization_stats(
            checkpoint,
            source=checkpoint_path,
        ),
        physics_config=physics_config,
    )

    expected_dimensions = {
        "num_bus_features": dataset.num_bus_features,
        "num_branch_features": dataset.num_branch_features,
        "num_actions": dataset.num_actions,
    }
    mismatches = [
        name
        for name, expected in expected_dimensions.items()
        if int(checkpoint.get(name, -1)) != int(expected)
    ]
    if mismatches:
        raise ValueError(
            "Replay examples are incompatible with the scoring checkpoint: "
            + ", ".join(mismatches)
            + "."
        )

    require_topology_action_provenance(
        checkpoint,
        source=str(checkpoint_path),
        expected_action_space_config=dataset.topology_action_config,
        expected_action_layout=dataset.action_layout,
    )
    model = _load_model(checkpoint, device=device)
    loader = DataLoader(
        dataset,
        batch_size=min(max(1, int(batch_size)), len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    entries: dict[str, dict[str, float]] = {}
    value_errors: list[float] = []
    policy_errors: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            policy_logits, predicted_value = _forward(model, batch)
            target_policy = batch["target_policy"].float()
            target_value = batch["target_value"].float().reshape(-1)
            predicted_value = predicted_value.float().reshape(-1)

            log_probs = torch.log_softmax(policy_logits.float(), dim=1)
            positive = target_policy > 0.0
            target_log = torch.zeros_like(target_policy)
            target_log[positive] = torch.log(target_policy[positive])
            policy_kl = torch.where(
                positive,
                target_policy * (target_log - log_probs),
                torch.zeros_like(target_policy),
            ).sum(dim=1).clamp_min(0.0)
            value_error = torch.abs(predicted_value - target_value)

            state_ids = [str(value) for value in batch["state_id"]]
            batch_value_errors = value_error.detach().cpu().numpy()
            batch_policy_errors = policy_kl.detach().cpu().numpy()

            for state_id, value_item, policy_item in zip(
                state_ids,
                batch_value_errors,
                batch_policy_errors,
            ):
                if state_id in entries:
                    raise ValueError(
                        f"Duplicate state_id while scoring replay: {state_id!r}."
                    )
                value_number = float(value_item)
                policy_number = float(policy_item)
                if not np.isfinite(value_number) or not np.isfinite(policy_number):
                    raise ValueError(
                        f"Non-finite replay prediction error for state {state_id!r}."
                    )
                entries[state_id] = {
                    "value_error": value_number,
                    "policy_kl_error": policy_number,
                }
                value_errors.append(value_number)
                policy_errors.append(policy_number)

    if len(entries) != len(dataset):
        raise RuntimeError(
            "Replay prediction scoring did not cover every example: "
            f"expected {len(dataset)}, observed {len(entries)}."
        )

    return {
        "schema_version": PREDICTION_ERROR_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_contract_version": int(
            checkpoint["checkpoint_contract_version"]
        ),
        "model_type": str(checkpoint["model_type"]),
        "examples_csv": str(examples_csv),
        "examples_csv_sha256": sha256_file(examples_csv),
        "example_count": len(entries),
        "mean_value_error": float(np.mean(value_errors)),
        "mean_policy_kl_error": float(np.mean(policy_errors)),
        "entries": entries,
    }

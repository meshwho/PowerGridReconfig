from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch

from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.evaluator import require_checkpoint_contracts
from grid_topology_ai.dataset import (
    GraphSelfPlayDataset,
)
from grid_topology_ai.training.metrics import build_value_target_diagnostics
from grid_topology_ai.actions import (
    action_layout_to_list,
    require_topology_action_payload,
)

_SELECTOR_METRIC_NAMES = {
    "val_loss": "validation_loss",
    "val_top1": "validation_top1",
    "val_top5": "validation_top5",
    "val_switch": "validation_switch_accuracy",
    "policy_selection_score": "policy_selection_score",
    "last_epoch": "last_epoch",
}

NORMALIZATION_STAT_KEYS = (
    "bus_feature_mean",
    "bus_feature_std",
    "branch_feature_mean",
    "branch_feature_std",
)


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_physics_config: PhysicsConfig | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint payload must be a mapping. Checkpoint: {checkpoint_path}"
        )
    require_checkpoint_contracts(
        payload,
        source=str(checkpoint_path),
        expected_physics_config=expected_physics_config,
    )
    return payload


def extract_normalization_stats(
    checkpoint_payload: Mapping[str, object],
    *,
    source: str | Path,
) -> dict[str, np.ndarray]:
    source_text = str(source)
    stats: dict[str, np.ndarray] = {}
    for key in NORMALIZATION_STAT_KEYS:
        if key not in checkpoint_payload:
            raise ValueError(
                "Initial checkpoint is missing required normalization statistics: "
                f"{key}. Checkpoint: {source_text}"
            )
        try:
            array = np.array(checkpoint_payload[key], dtype=np.float32, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid normalization statistic {key!r} in checkpoint "
                f"{source_text}: cannot convert to float32 ({exc})"
            ) from exc
        if array.ndim != 1:
            raise ValueError(
                f"Invalid normalization statistic {key!r} in checkpoint "
                f"{source_text}: expected 1D array, got shape {array.shape}"
            )
        if array.size == 0:
            raise ValueError(
                f"Invalid normalization statistic {key!r} in checkpoint "
                f"{source_text}: array must not be empty"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"Invalid normalization statistic {key!r} in checkpoint "
                f"{source_text}: all values must be finite"
            )
        if key.endswith("_std") and not (array > 0.0).all():
            raise ValueError(
                f"Invalid normalization statistic {key!r} in checkpoint "
                f"{source_text}: std values must be strictly positive"
            )
        stats[key] = array

    for mean_key, std_key in (
        ("bus_feature_mean", "bus_feature_std"),
        ("branch_feature_mean", "branch_feature_std"),
    ):
        if stats[mean_key].shape != stats[std_key].shape:
            raise ValueError(
                f"Invalid normalization statistics in checkpoint {source_text}: "
                f"{mean_key} shape {stats[mean_key].shape} does not match "
                f"{std_key} shape {stats[std_key].shape}"
            )

    return {key: value.copy() for key, value in stats.items()}


if TYPE_CHECKING:
    from grid_topology_ai.training.graph_policy_value import TrainingRequest


def get_git_commit(repo_root: Path) -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None

    return commit or None


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return str(value)


def build_training_config_payload(request: "TrainingRequest") -> dict[str, Any]:
    return {
        "examples_csv": make_json_safe(request.examples_csv),
        "seed": int(request.seed),
        "epochs": int(request.config.epochs),
        "lr": float(request.config.learning_rate),
        "hidden_dim": int(request.config.hidden_dim),
        "num_layers": int(request.config.num_layers),
        "dropout": float(request.config.dropout),
        "batch_size": int(request.config.batch_size),
        "value_loss_weight": float(request.config.value_loss_weight),
        "value_huber_delta": float(request.config.value_huber_delta),
        "device": str(request.config.device),
        "amp": bool(request.use_amp),
        "num_workers": int(request.config.num_workers),
        "no_normalize_features": bool(not request.normalize_features),
        "output": make_json_safe(request.output_path),
        "init_checkpoint": make_json_safe(request.init_checkpoint),
        "resume_checkpoint": make_json_safe(request.resume_checkpoint),
        "val_examples_csv": make_json_safe(request.validation_examples_csv),
        "save_best": bool(request.save_best),
        "tensorboard_log_dir": make_json_safe(request.tensorboard_log_dir),
        "run_name": make_json_safe(request.run_name),
        "no_tensorboard": bool(request.config.no_tensorboard),
        "metrics_csv": make_json_safe(request.metrics_csv),
    }


def make_checkpoint(
    *,
    model: torch.nn.Module,
    dataset: GraphSelfPlayDataset,
    request: "TrainingRequest",
    device: torch.device,
    use_amp: bool,
    normalization_metadata: Mapping[str, object] | None = None,
    validation_dataset: GraphSelfPlayDataset | None = None,
) -> dict[str, Any]:
    model_state_dict_cpu = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    normalization = dataset.normalization_state_dict()
    repo_root = request.project_root.resolve()
    validation_examples_count = (
        0 if validation_dataset is None else int(len(validation_dataset))
    )
    validation_scenario_count = (
        0
        if validation_dataset is None
        else int(validation_dataset.examples["scenario_id"].nunique())
        if "scenario_id" in validation_dataset.examples.columns
        else 0
    )
    training_scenario_count = (
        int(dataset.examples["scenario_id"].nunique())
        if "scenario_id" in dataset.examples.columns
        else 0
    )
    value_target_diagnostics = build_value_target_diagnostics(dataset=dataset)

    checkpoint = {
        "physics_config": dataset.physics_config.to_dict(),
        "topology_action_config": dataset.topology_action_config.to_contract_dict(),
        "action_layout": action_layout_to_list(dataset.action_layout),
        "policy_layout": str(dataset.policy_layout),
        "model_type": "graph_policy_value_net_v2",
        "topology_cardinality_independent": True,
        "model_state_dict": model_state_dict_cpu,
        "num_bus_features": int(dataset.num_bus_features),
        "num_branch_features": int(dataset.num_branch_features),
        "hidden_dim": int(request.config.hidden_dim),
        "num_layers": int(request.config.num_layers),
        "dropout": float(request.config.dropout),
        "examples_csv": str(request.examples_csv),
        "training_seed": int(request.seed),
        "checkpoint_selection_metric": (
            "validation_loss" if validation_dataset is not None else "training_loss"
        ),
        "validation_examples_csv": (
            None
            if request.validation_examples_csv is None
            else str(request.validation_examples_csv)
        ),
        "validation_examples_count": validation_examples_count,
        "validation_scenario_count": validation_scenario_count,
        "training_scenario_count": training_scenario_count,
        "scenario_split_verified": bool(validation_dataset is not None),
        "value_scale": 1.0,
        "value_target_mode": "outcome_value_target",
        "normalize_features": bool(request.normalize_features),
        "device_used_for_training": str(device),
        "amp_used": bool(use_amp),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(repo_root),
        "repo_root": str(repo_root),
        "training_config": build_training_config_payload(request),
        "value_target_diagnostics": value_target_diagnostics,
        "bus_feature_mean": normalization["bus_feature_mean"],
        "bus_feature_std": normalization["bus_feature_std"],
        "branch_feature_mean": normalization["branch_feature_mean"],
        "branch_feature_std": normalization["branch_feature_std"],
        "normalization_source": "training_dataset",
        "normalization_frozen_from_init_checkpoint": False,
        "normalization_source_checkpoint": None,
    }

    if normalization_metadata is not None:
        checkpoint.update(make_json_safe(dict(normalization_metadata)))

    return checkpoint


def atomic_save_checkpoint(payload: Mapping[str, object], path: Path) -> None:
    """Write a checkpoint without exposing a partially-written final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_initial_checkpoint_into_model(
    *,
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    dataset: GraphSelfPlayDataset,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
    checkpoint_payload: Mapping[str, object] | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    checkpoint = (
        dict(checkpoint_payload)
        if checkpoint_payload is not None
        else load_checkpoint_payload(checkpoint_path, map_location=device)
    )
    require_checkpoint_contracts(
        checkpoint,
        source=str(checkpoint_path),
        expected_physics_config=dataset.physics_config,
    )
    require_topology_action_payload(
        checkpoint,
        source=str(checkpoint_path),
        expected_action_space_config=dataset.topology_action_config,
    )

    if checkpoint.get("policy_layout") != dataset.policy_layout:
        raise ValueError(
            "Initial checkpoint policy layout does not match the dataset. "
            f"Checkpoint: {checkpoint_path}"
        )

    expected_model_type = "graph_policy_value_net_v2"
    actual_model_type = str(checkpoint.get("model_type", ""))
    if actual_model_type != expected_model_type:
        raise ValueError(
            "Initial checkpoint model_type mismatch. "
            f"Expected {expected_model_type!r}, got {actual_model_type!r}. "
            f"Checkpoint: {checkpoint_path}"
        )

    checks = {
        "num_bus_features": int(dataset.num_bus_features),
        "num_branch_features": int(dataset.num_branch_features),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
    }

    for key, expected_value in checks.items():
        if key not in checkpoint:
            raise KeyError(
                f"Initial checkpoint is missing required key {key!r}: {checkpoint_path}"
            )
        actual_value = int(checkpoint[key])
        if actual_value != expected_value:
            raise ValueError(
                f"Initial checkpoint {key} mismatch. "
                f"Expected {expected_value}, got {actual_value}. "
                f"Checkpoint: {checkpoint_path}"
            )

    expected_dropout = float(dropout)
    if float(checkpoint.get("dropout", -1.0)) != expected_dropout:
        raise ValueError(
            "Initial checkpoint dropout mismatch. "
            f"Expected {expected_dropout}, got {checkpoint.get('dropout')}. "
            f"Checkpoint: {checkpoint_path}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Initial checkpoint has no model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])

    print("")
    print("=" * 100)
    print("INITIAL CHECKPOINT LOADED")
    print("=" * 100)
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Model type:     {actual_model_type}")
    print(f"Hidden dim:     {checkpoint['hidden_dim']}")
    print(f"Num layers:     {checkpoint['num_layers']}")
    print("Policy actions: dynamic per graph")

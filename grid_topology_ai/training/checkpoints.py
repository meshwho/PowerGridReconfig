from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.self_play.artifacts import sha256_file
from grid_topology_ai.training.metrics import build_value_target_diagnostics

if TYPE_CHECKING:
    from grid_topology_ai.training.graph_policy_value import TrainingRequest


def sha256_text(text: str) -> str:
    """
    Compute SHA256 for a UTF-8 string.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_git_commit(repo_root: Path) -> str | None:
    """
    Return current git commit hash if the project is inside a git repo.
    """

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
    """
    Convert config values to JSON-safe primitives.
    """

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
    """
    Store command-line-equivalent training arguments in the checkpoint.
    """

    return {
        "examples_csv": make_json_safe(request.examples_csv),
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
        "val_examples_csv": make_json_safe(request.validation_examples_csv),
        "save_best": bool(request.save_best),
        "tensorboard_log_dir": make_json_safe(request.tensorboard_log_dir),
        "run_name": make_json_safe(request.run_name),
        "no_tensorboard": bool(request.config.no_tensorboard),
        "metrics_csv": make_json_safe(request.metrics_csv),
        "model_type": str(request.config.model_type),
        "save_multiple_best": bool(request.config.save_multiple_best),
    }


def build_dataset_metadata(
    dataset: GraphSelfPlayDataset,
    repo_root: Path,
) -> dict[str, Any]:
    """
    Build reproducibility metadata for the training dataset.
    """

    examples_csv = Path(dataset.examples_csv)
    examples_csv_abs = examples_csv.resolve()

    state_paths = [
        str(p).replace("\\", "/")
        for p in dataset.examples["state_path"].astype(str).tolist()
    ]

    unique_state_paths = sorted(set(state_paths))

    existing_state_count = 0
    missing_state_count = 0
    state_total_bytes = 0

    for state_path_str in unique_state_paths:
        state_path = Path(state_path_str)

        if not state_path.is_absolute():
            state_path = repo_root / state_path

        if state_path.exists():
            existing_state_count += 1
            state_total_bytes += int(state_path.stat().st_size)
        else:
            missing_state_count += 1

    return {
        "examples_csv": str(examples_csv),
        "examples_csv_abs": str(examples_csv_abs),
        "examples_csv_sha256": sha256_file(examples_csv_abs),
        "examples_count": int(len(dataset.examples)),
        "scenario_count": int(dataset.examples["scenario_id"].nunique())
        if "scenario_id" in dataset.examples.columns
        else None,
        "state_reference_count": int(len(state_paths)),
        "unique_state_count": int(len(unique_state_paths)),
        "existing_state_count": int(existing_state_count),
        "missing_state_count": int(missing_state_count),
        "state_total_bytes": int(state_total_bytes),
        "state_paths_sha256": sha256_text("\n".join(unique_state_paths)),
    }


def make_checkpoint(
    *,
    model: torch.nn.Module,
    dataset: GraphSelfPlayDataset,
    request: "TrainingRequest",
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    """
    Build checkpoint dictionary.

    We save model weights on CPU so the checkpoint can be loaded on any machine.
    """

    model_state_dict_cpu = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    normalization = dataset.normalization_state_dict()
    repo_root = request.project_root.resolve()
    dataset_metadata = build_dataset_metadata(
        dataset=dataset,
        repo_root=repo_root,
    )
    value_target_diagnostics = build_value_target_diagnostics(dataset=dataset)

    checkpoint = {
        "model_type": str(getattr(model, "model_type", "graph_policy_value_net")),
        "model_state_dict": model_state_dict_cpu,
        "num_bus_features": int(dataset.num_bus_features),
        "num_branch_features": int(dataset.num_branch_features),
        "num_buses": int(dataset.num_buses),
        "num_branches": int(dataset.num_branches),
        "num_actions": int(dataset.num_actions),
        "hidden_dim": int(request.config.hidden_dim),
        "num_layers": int(request.config.num_layers),
        "dropout": float(request.config.dropout),
        "examples_csv": str(request.examples_csv),
        "value_scale": 1.0,
        "value_target_mode": "outcome_value_target",
        "normalize_features": bool(request.normalize_features),
        "device_used_for_training": str(device),
        "amp_used": bool(use_amp),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(repo_root),
        "repo_root": str(repo_root),
        "training_config": build_training_config_payload(request),
        "dataset_metadata": dataset_metadata,
        "value_target_diagnostics": value_target_diagnostics,
        "bus_feature_mean": normalization["bus_feature_mean"],
        "bus_feature_std": normalization["bus_feature_std"],
        "branch_feature_mean": normalization["branch_feature_mean"],
        "branch_feature_std": normalization["branch_feature_std"],
    }

    return checkpoint


def load_initial_checkpoint_into_model(
    *,
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    dataset: GraphSelfPlayDataset,
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
) -> None:
    """
    Load model weights from an existing graph policy-value checkpoint.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    expected_model_type = (
        "graph_policy_value_net_v2"
        if model_type == "graph_v2"
        else "graph_policy_value_net"
    )
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
        "num_actions": int(dataset.num_actions),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
    }

    for key, expected_value in checks.items():
        if key not in checkpoint:
            raise KeyError(
                f"Initial checkpoint is missing required key {key!r}: "
                f"{checkpoint_path}"
            )

        actual_value = int(checkpoint[key])

        if actual_value != expected_value:
            raise ValueError(
                f"Initial checkpoint {key} mismatch. "
                f"Expected {expected_value}, got {actual_value}. "
                f"Checkpoint: {checkpoint_path}"
            )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Initial checkpoint has no model_state_dict: {checkpoint_path}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("")
    print("=" * 100)
    print("INITIAL CHECKPOINT LOADED")
    print("=" * 100)
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Model type:     {actual_model_type}")
    print(f"Hidden dim:     {checkpoint['hidden_dim']}")
    print(f"Num layers:     {checkpoint['num_layers']}")
    print(f"Num actions:    {checkpoint['num_actions']}")


def checkpoint_variant_path(
    output_path: Path,
    variant_name: str,
) -> Path:
    """
    Build path for additional checkpoint variants.
    """

    return output_path.with_name(
        f"{output_path.stem}_{variant_name}{output_path.suffix}"
    )


def save_checkpoint_now(
    *,
    path: Path,
    model: torch.nn.Module,
    dataset: GraphSelfPlayDataset,
    request: "TrainingRequest",
    device: torch.device,
    use_amp: bool,
    epoch: int,
    selector_name: str,
    selector_value: float,
    val_metrics: dict[str, float] | None,
) -> None:
    """
    Save checkpoint immediately when a selector improves.
    """

    checkpoint = make_checkpoint(
        model=model,
        dataset=dataset,
        request=request,
        device=device,
        use_amp=use_amp,
    )

    checkpoint["saved_epoch"] = int(epoch)
    checkpoint["selector_name"] = str(selector_name)
    checkpoint["selector_value"] = float(selector_value)

    if val_metrics is not None:
        checkpoint["val_metrics"] = {
            key: float(value)
            for key, value in val_metrics.items()
            if isinstance(value, (int, float))
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

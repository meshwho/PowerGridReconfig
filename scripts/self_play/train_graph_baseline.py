from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from grid_topology_ai.models.graph_policy_value_net import GraphPolicyValueNet
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2

def resolve_device(device_arg: str) -> torch.device:
    """
    Resolve requested training device.

    device_arg:
        auto -> cuda if available, otherwise cpu
        cuda -> force CUDA
        cpu  -> force CPU
    """

    device_arg = str(device_arg).lower().strip()

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --device cpu."
            )

        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {device_arg}. "
        "Use one of: auto, cuda, cpu."
    )

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA256 for a file.

    Used to bind checkpoints to the exact examples.csv that was used
    during training.
    """

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    """
    Compute SHA256 for a UTF-8 string.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_git_commit(repo_root: Path) -> str | None:
    """
    Return current git commit hash if the project is inside a git repo.

    If git is not available or the directory is not a git repository,
    return None instead of failing training.
    """

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        return commit or None

    except Exception:
        return None


def make_json_safe(value: Any) -> Any:
    """
    Convert argparse/config values to JSON-safe primitives.
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


def build_training_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Store all command-line training arguments in the checkpoint.
    """

    return {
        key: make_json_safe(value)
        for key, value in vars(args).items()
    }


def build_dataset_metadata(
    dataset: GraphSelfPlayDataset,
    repo_root: Path,
) -> dict[str, Any]:
    """
    Build reproducibility metadata for the training dataset.

    We intentionally hash examples.csv and the list of referenced state paths.
    Hashing every .npz file can be expensive, so we store state count and
    total byte size instead.
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

def build_value_target_diagnostics(
    dataset: GraphSelfPlayDataset,
    value_scale: float,
) -> dict[str, Any]:
    """
    Analyze value targets.

    Preferred modern mode:
        examples.csv contains explicit value_target in [-1, 1].

    Legacy fallback:
        target_value = discounted_return_from_step / value_scale
        target_value = clip(target_value, -1, 1)
    """

    def q(x, p):
        return float(np.quantile(x, p)) if len(x) > 0 else 0.0

    if "value_target" in dataset.examples.columns:
        target = dataset.examples["value_target"].astype(float).to_numpy()

        abs_target = np.abs(target)
        outside_mask = abs_target > 1.0

        return {
            "available": True,
            "mode": "explicit_value_target",
            "count": int(len(target)),

            "target_min": float(target.min()) if len(target) else 0.0,
            "target_max": float(target.max()) if len(target) else 0.0,
            "target_mean": float(target.mean()) if len(target) else 0.0,
            "target_std": float(target.std()) if len(target) else 0.0,

            "abs_target_p50": q(abs_target, 0.50),
            "abs_target_p90": q(abs_target, 0.90),
            "abs_target_p95": q(abs_target, 0.95),
            "abs_target_p99": q(abs_target, 0.99),
            "abs_target_max": float(abs_target.max()) if len(abs_target) else 0.0,

            "outside_minus1_plus1_count": int(outside_mask.sum()),
            "outside_minus1_plus1_percent": (
                float(outside_mask.mean() * 100.0) if len(outside_mask) else 0.0
            ),

            "positive_count": int((target > 0).sum()),
            "zero_count": int((target == 0).sum()),
            "negative_count": int((target < 0).sum()),
        }

    if "discounted_return_from_step" not in dataset.examples.columns:
        return {
            "available": False,
            "reason": "Neither value_target nor discounted_return_from_step column is available.",
        }

    raw = dataset.examples["discounted_return_from_step"].astype(float).to_numpy()

    scale = float(value_scale)

    if scale <= 0:
        return {
            "available": False,
            "reason": f"value_scale must be positive, got {scale}",
        }

    normalized = raw / scale
    abs_normalized = np.abs(normalized)

    clipped_mask = abs_normalized >= 1.0

    return {
        "available": True,
        "mode": "legacy_scaled_discounted_return",
        "value_scale": scale,
        "count": int(len(raw)),

        "raw_min": float(raw.min()) if len(raw) else 0.0,
        "raw_max": float(raw.max()) if len(raw) else 0.0,
        "raw_mean": float(raw.mean()) if len(raw) else 0.0,
        "raw_std": float(raw.std()) if len(raw) else 0.0,

        "normalized_min": float(normalized.min()) if len(normalized) else 0.0,
        "normalized_max": float(normalized.max()) if len(normalized) else 0.0,
        "normalized_mean": float(normalized.mean()) if len(normalized) else 0.0,
        "normalized_std": float(normalized.std()) if len(normalized) else 0.0,

        "abs_normalized_p50": q(abs_normalized, 0.50),
        "abs_normalized_p90": q(abs_normalized, 0.90),
        "abs_normalized_p95": q(abs_normalized, 0.95),
        "abs_normalized_p99": q(abs_normalized, 0.99),
        "abs_normalized_max": float(abs_normalized.max()) if len(abs_normalized) else 0.0,

        "clipped_count": int(clipped_mask.sum()),
        "clipped_percent": (
            float(clipped_mask.mean() * 100.0) if len(clipped_mask) else 0.0
        ),

        "positive_count": int((raw > 0).sum()),
        "zero_count": int((raw == 0).sum()),
        "negative_count": int((raw < 0).sum()),
    }
    """
    Analyze value targets before clipping.

    GraphSelfPlayDataset uses:
        target_value = discounted_return_from_step / value_scale
        target_value = clip(target_value, -1, 1)

    This diagnostic tells us how often value targets saturate.
    """

    if "value_target" in dataset.examples.columns:
        target = dataset.examples["value_target"].astype(float).to_numpy()

    abs_target = np.abs(target)
    outside_mask = abs_target > 1.0

    def q(x, p):
        return float(np.quantile(x, p)) if len(x) > 0 else 0.0

    return {
        "available": True,
        "mode": "explicit_value_target",
        "count": int(len(target)),
        "target_min": float(target.min()) if len(target) else 0.0,
        "target_max": float(target.max()) if len(target) else 0.0,
        "target_mean": float(target.mean()) if len(target) else 0.0,
        "target_std": float(target.std()) if len(target) else 0.0,
        "abs_target_p50": q(abs_target, 0.50),
        "abs_target_p90": q(abs_target, 0.90),
        "abs_target_p95": q(abs_target, 0.95),
        "abs_target_p99": q(abs_target, 0.99),
        "abs_target_max": float(abs_target.max()) if len(abs_target) else 0.0,
        "outside_minus1_plus1_count": int(outside_mask.sum()),
        "outside_minus1_plus1_percent": float(outside_mask.mean() * 100.0)
        if len(outside_mask)
        else 0.0,
        "positive_count": int((target > 0).sum()),
        "zero_count": int((target == 0).sum()),
        "negative_count": int((target < 0).sum()),
    }


    if "discounted_return_from_step" not in dataset.examples.columns:
        return {
            "available": False,
            "reason": "discounted_return_from_step column is missing",
        }

    raw = dataset.examples["discounted_return_from_step"].astype(float).to_numpy()

    scale = float(value_scale)

    if scale <= 0:
        return {
            "available": False,
            "reason": f"value_scale must be positive, got {scale}",
        }

    normalized = raw / scale
    abs_normalized = abs(normalized)

    clipped_mask = abs_normalized >= 1.0

    def q(x, p):
        return float(np.quantile(x, p)) if len(x) > 0 else 0.0

    return {
        "available": True,
        "value_scale": scale,
        "count": int(len(raw)),

        "raw_min": float(raw.min()) if len(raw) else 0.0,
        "raw_max": float(raw.max()) if len(raw) else 0.0,
        "raw_mean": float(raw.mean()) if len(raw) else 0.0,
        "raw_std": float(raw.std()) if len(raw) else 0.0,

        "normalized_min": float(normalized.min()) if len(normalized) else 0.0,
        "normalized_max": float(normalized.max()) if len(normalized) else 0.0,
        "normalized_mean": float(normalized.mean()) if len(normalized) else 0.0,
        "normalized_std": float(normalized.std()) if len(normalized) else 0.0,

        "abs_normalized_p50": q(abs_normalized, 0.50),
        "abs_normalized_p90": q(abs_normalized, 0.90),
        "abs_normalized_p95": q(abs_normalized, 0.95),
        "abs_normalized_p99": q(abs_normalized, 0.99),
        "abs_normalized_max": float(abs_normalized.max()) if len(abs_normalized) else 0.0,

        "clipped_count": int(clipped_mask.sum()),
        "clipped_percent": float(clipped_mask.mean() * 100.0) if len(clipped_mask) else 0.0,

        "positive_count": int((raw > 0).sum()),
        "zero_count": int((raw == 0).sum()),
        "negative_count": int((raw < 0).sum()),
    }


def print_value_target_diagnostics(
    diagnostics: dict[str, Any],
) -> None:
    """
    Print value target diagnostics in a compact readable form.
    """

    print("")
    print("=" * 100)
    print("VALUE TARGET DIAGNOSTICS")
    print("=" * 100)

    if not diagnostics.get("available", False):
        print(f"Unavailable: {diagnostics.get('reason')}")
        return

    print(f"value_scale:        {diagnostics['value_scale']}")
    print(f"count:              {diagnostics['count']}")
    print("")
    print(f"raw min:            {diagnostics['raw_min']:.6f}")
    print(f"raw max:            {diagnostics['raw_max']:.6f}")
    print(f"raw mean:           {diagnostics['raw_mean']:.6f}")
    print(f"raw std:            {diagnostics['raw_std']:.6f}")
    print("")
    print(f"normalized min:     {diagnostics['normalized_min']:.6f}")
    print(f"normalized max:     {diagnostics['normalized_max']:.6f}")
    print(f"normalized mean:    {diagnostics['normalized_mean']:.6f}")
    print(f"normalized std:     {diagnostics['normalized_std']:.6f}")
    print("")
    print(f"abs norm p50:       {diagnostics['abs_normalized_p50']:.6f}")
    print(f"abs norm p90:       {diagnostics['abs_normalized_p90']:.6f}")
    print(f"abs norm p95:       {diagnostics['abs_normalized_p95']:.6f}")
    print(f"abs norm p99:       {diagnostics['abs_normalized_p99']:.6f}")
    print(f"abs norm max:       {diagnostics['abs_normalized_max']:.6f}")
    print("")
    print(f"clipped count:      {diagnostics['clipped_count']}")
    print(f"clipped percent:    {diagnostics['clipped_percent']:.2f}%")
    print("")
    print(f"positive count:     {diagnostics['positive_count']}")
    print(f"zero count:         {diagnostics['zero_count']}")
    print(f"negative count:     {diagnostics['negative_count']}")

    if diagnostics["clipped_percent"] > 10.0:
        print("")
        print("WARNING: More than 10% of value targets are clipped.")
        print("The value head may receive saturated targets.")
    if diagnostics.get("mode") == "explicit_value_target":
    print(f"mode:               {diagnostics['mode']}")
    print(f"count:              {diagnostics['count']}")
    print("")
    print(f"target min:         {diagnostics['target_min']:.6f}")
    print(f"target max:         {diagnostics['target_max']:.6f}")
    print(f"target mean:        {diagnostics['target_mean']:.6f}")
    print(f"target std:         {diagnostics['target_std']:.6f}")
    print("")
    print(f"abs target p50:     {diagnostics['abs_target_p50']:.6f}")
    print(f"abs target p90:     {diagnostics['abs_target_p90']:.6f}")
    print(f"abs target p95:     {diagnostics['abs_target_p95']:.6f}")
    print(f"abs target p99:     {diagnostics['abs_target_p99']:.6f}")
    print(f"abs target max:     {diagnostics['abs_target_max']:.6f}")
    print("")
    print(f"outside [-1,1]:     {diagnostics['outside_minus1_plus1_count']}")
    print(
        f"outside percent:    "
        f"{diagnostics['outside_minus1_plus1_percent']:.2f}%"
    )
    return


def soft_policy_loss(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy with soft policy target.

    loss = -sum_a pi(a) * log p(a)
    """

    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(target_policy * log_probs).sum(dim=1).mean()

    return loss


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    Move tensor batch fields to selected device.
    """

    moved: dict[str, Any] = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def make_checkpoint(
    model: GraphPolicyValueNet,
    dataset: GraphSelfPlayDataset,
    args: argparse.Namespace,
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

    repo_root = Path.cwd().resolve()
    dataset_metadata = build_dataset_metadata(
        dataset=dataset,
        repo_root=repo_root,
    )

    value_target_diagnostics = build_value_target_diagnostics(
        dataset=dataset,
        value_scale=float(args.value_scale),
    )

    checkpoint = {
        "model_type": str(getattr(model, "model_type", "graph_policy_value_net")),
        "model_state_dict": model_state_dict_cpu,

        "num_bus_features": int(dataset.num_bus_features),
        "num_branch_features": int(dataset.num_branch_features),
        "num_buses": int(dataset.num_buses),
        "num_branches": int(dataset.num_branches),
        "num_actions": int(dataset.num_actions),

        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),

        "examples_csv": str(args.examples_csv),
        "value_scale": float(args.value_scale),
        "normalize_features": bool(not args.no_normalize_features),

        "device_used_for_training": str(device),
        "amp_used": bool(use_amp),

        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(repo_root),
        "repo_root": str(repo_root),
        "training_config": build_training_config(args),
        "dataset_metadata": dataset_metadata,
        "value_target_diagnostics": value_target_diagnostics,

        "bus_feature_mean": normalization["bus_feature_mean"],
        "bus_feature_std": normalization["bus_feature_std"],
        "branch_feature_mean": normalization["branch_feature_mean"],
        "branch_feature_std": normalization["branch_feature_std"],
    }

    return checkpoint

def checkpoint_variant_path(
    output_path: Path,
    variant_name: str,
) -> Path:
    """
    Build path for additional checkpoint variants.

    Example:
        graph_policy_value_net_v2.pt
        graph_policy_value_net_v2_best_switch.pt
    """

    return output_path.with_name(
        f"{output_path.stem}_{variant_name}{output_path.suffix}"
    )


def compute_policy_selection_score(
    val_metrics: dict[str, float],
) -> float:
    """
    Composite score for selecting policy-useful checkpoints.

    Higher is better.

    This score is not a scientific metric.
    It is a practical checkpoint-selection metric for MCTS.

    We care about:
    - top1: direct imitation quality;
    - top5: whether MCTS sees the correct action among candidates;
    - switch_acc: branch-selection quality;
    - stop_acc: handoff quality;
    - value_loss: small penalty, because bad value can hurt search.
    """

    top1 = float(val_metrics["top1"])
    top5 = float(val_metrics["top5"])
    stop = float(val_metrics["stop_acc"])
    switch = float(val_metrics["switch_acc"])
    value_loss = float(val_metrics["value_loss"])

    balance = min(stop, switch)

    score = (
        1.00 * top1
        + 1.00 * top5
        + 1.50 * switch
        + 0.50 * stop
        + 0.50 * balance
        - 0.25 * value_loss
    )

    return float(score)


def save_checkpoint_now(
    *,
    path: Path,
    model: GraphPolicyValueNet,
    dataset: GraphSelfPlayDataset,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    selector_name: str,
    selector_value: float,
    val_metrics: dict[str, float] | None,
) -> None:
    """
    Save checkpoint immediately when a selector improves.

    This protects us from losing a useful checkpoint if training later overfits
    or if the run is interrupted.
    """

    checkpoint = make_checkpoint(
        model=model,
        dataset=dataset,
        args=args,
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

def train_one_epoch(
    model: GraphPolicyValueNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    value_loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    value_loss_weight: float,
) -> tuple[float, float, float]:
    """
    Train one epoch.
    """

    model.train()

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        bus_features = batch["bus_features"]
        branch_features = batch["branch_features"]
        edge_index = batch["edge_index"]
        action_mask = batch["action_mask"]
        target_policy = batch["target_policy"]
        target_value = batch["target_value"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            policy_logits, predicted_value = model(
                bus_features=bus_features,
                branch_features=branch_features,
                edge_index=edge_index,
                action_mask=action_mask,
            )

            policy_loss = soft_policy_loss(
                logits=policy_logits,
                target_policy=target_policy,
            )

            value_loss = value_loss_fn(
                predicted_value,
                target_value,
            )

            total_loss = policy_loss + float(value_loss_weight) * value_loss

        scaler.scale(total_loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += float(total_loss.detach().item())
        policy_loss_sum += float(policy_loss.detach().item())
        value_loss_sum += float(value_loss.detach().item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Training loader produced zero batches.")

    return (
        total_loss_sum / batches,
        policy_loss_sum / batches,
        value_loss_sum / batches,
    )


def evaluate_one_epoch(
    model: GraphPolicyValueNet,
    loader: DataLoader,
    value_loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    value_loss_weight: float,
) -> dict[str, float]:
    """
    Evaluate graph model on validation data.

    Computes:
    - validation total loss;
    - policy loss;
    - value loss;
    - top-1 / top-3 / top-5 policy accuracy;
    - stop accuracy;
    - switch accuracy.
    """

    model.eval()

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    total_examples = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    stop_total = 0
    stop_correct = 0

    switch_total = 0
    switch_correct = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)

            bus_features = batch["bus_features"]
            branch_features = batch["branch_features"]
            edge_index = batch["edge_index"]
            action_mask = batch["action_mask"]
            target_policy = batch["target_policy"]
            target_value = batch["target_value"]

            batch_size = int(bus_features.shape[0])

            with torch.amp.autocast("cuda", enabled=use_amp):
                policy_logits, predicted_value = model(
                    bus_features=bus_features,
                    branch_features=branch_features,
                    edge_index=edge_index,
                    action_mask=action_mask,
                )

                policy_loss = soft_policy_loss(
                    logits=policy_logits,
                    target_policy=target_policy,
                )

                value_loss = value_loss_fn(
                    predicted_value,
                    target_value,
                )

                total_loss = policy_loss + float(value_loss_weight) * value_loss

            total_loss_sum += float(total_loss.detach().item()) * batch_size
            policy_loss_sum += float(policy_loss.detach().item()) * batch_size
            value_loss_sum += float(value_loss.detach().item()) * batch_size

            target_action = torch.argmax(target_policy, dim=1)
            predicted_top = torch.argmax(policy_logits, dim=1)

            topk = torch.topk(
                policy_logits,
                k=min(5, policy_logits.shape[1]),
                dim=1,
            ).indices

            top1_correct += int((predicted_top == target_action).sum().item())

            top3 = topk[:, : min(3, topk.shape[1])]
            top5 = topk[:, : min(5, topk.shape[1])]

            top3_correct += int(
                (top3 == target_action.unsqueeze(1)).any(dim=1).sum().item()
            )

            top5_correct += int(
                (top5 == target_action.unsqueeze(1)).any(dim=1).sum().item()
            )

            stop_mask = target_action == 0
            switch_mask = target_action != 0

            stop_total += int(stop_mask.sum().item())
            switch_total += int(switch_mask.sum().item())

            if stop_mask.any():
                stop_correct += int(
                    (predicted_top[stop_mask] == 0).sum().item()
                )

            if switch_mask.any():
                switch_correct += int(
                    (
                        predicted_top[switch_mask]
                        == target_action[switch_mask]
                    ).sum().item()
                )

            total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("Validation loader produced zero examples.")

    return {
        "loss": total_loss_sum / total_examples,
        "policy_loss": policy_loss_sum / total_examples,
        "value_loss": value_loss_sum / total_examples,
        "top1": top1_correct / total_examples,
        "top3": top3_correct / total_examples,
        "top5": top5_correct / total_examples,
        "stop_acc": stop_correct / stop_total if stop_total > 0 else 0.0,
        "switch_acc": switch_correct / switch_total if switch_total > 0 else 0.0,
        "examples": float(total_examples),
    }


def evaluate_training_samples(
    model: GraphPolicyValueNet,
    dataset: GraphSelfPlayDataset,
    device: torch.device,
    max_samples: int = 20,
) -> None:
    """
    Print final predictions on a small subset of training data.
    """

    model.eval()

    n = min(len(dataset), int(max_samples))

    print("\nFinal predictions:")

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]

            bus_features = sample["bus_features"].unsqueeze(0).to(device)
            branch_features = sample["branch_features"].unsqueeze(0).to(device)
            edge_index = sample["edge_index"].unsqueeze(0).to(device)
            action_mask = sample["action_mask"].unsqueeze(0).to(device)
            target_policy = sample["target_policy"].unsqueeze(0).to(device)

            target_value = float(sample["target_value"].item())

            logits, value = model(
                bus_features=bus_features,
                branch_features=branch_features,
                edge_index=edge_index,
                action_mask=action_mask,
            )

            probabilities = torch.softmax(logits, dim=1)

            predicted_action = int(torch.argmax(probabilities, dim=1).item())
            target_top_action = int(torch.argmax(target_policy, dim=1).item())

            predicted_value = float(value.detach().cpu().item())
            predicted_prob = float(
                probabilities[0, predicted_action].detach().cpu().item()
            )
            target_prob = float(
                target_policy[0, target_top_action].detach().cpu().item()
            )

            print(
                f"Scenario {sample['scenario_id']:>5} | "
                f"step={sample['step']:>2} | "
                f"{sample['state_id']} | "
                f"target_top={target_top_action:>3} "
                f"(pi={target_prob:.3f}) | "
                f"pred_top={predicted_action:>3} "
                f"(p={predicted_prob:.3f}) | "
                f"value target={target_value:+.3f} | "
                f"value pred={predicted_value:+.3f}"
            )


def setup_live_logging(
    args: argparse.Namespace,
    output_path: Path,
) -> tuple[Any, Path]:
    """
    Initialize TensorBoard writer and metrics CSV before the training loop.

    This must happen before log_epoch_metrics() is called.
    """

    run_name = args.run_name

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"graph_train_{timestamp}"

    if args.tensorboard_log_dir is None:
        tensorboard_root = output_path.parent / "tensorboard"
    else:
        tensorboard_root = Path(args.tensorboard_log_dir)

    tensorboard_dir = tensorboard_root / run_name

    writer = None

    if not args.no_tensorboard:
        if SummaryWriter is None:
            print(
                "WARNING: TensorBoard is not installed. "
                "Run: python -m pip install tensorboard"
            )
        else:
            writer = SummaryWriter(log_dir=str(tensorboard_dir))
            print(f"TensorBoard log dir: {tensorboard_dir}")

    if args.metrics_csv is None:
        metrics_csv_path = output_path.parent / f"{run_name}_metrics.csv"
    else:
        metrics_csv_path = Path(args.metrics_csv)

    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_fieldnames = [
        "epoch",
        "train_loss",
        "train_policy",
        "train_value",
        "val_loss",
        "val_policy",
        "val_value",
        "val_top1",
        "val_top3",
        "val_top5",
        "val_stop",
        "val_switch",
        "best_epoch",
        "best_metric",
        "learning_rate",
    ]

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=metrics_fieldnames)
        writer_csv.writeheader()

    print(f"Metrics CSV: {metrics_csv_path}")

    return writer, metrics_csv_path


def log_epoch_metrics(
    *,
    tensorboard_writer,
    metrics_csv_path: Path,
    epoch: int,
    train_loss: float,
    train_policy: float,
    train_value: float,
    val_metrics: dict[str, float] | None,
    best_epoch: int,
    best_metric: float,
    learning_rate: float,
) -> None:
    """
    Save epoch metrics to TensorBoard and CSV.

    TensorBoard gives live charts in browser.
    CSV is useful for later analysis and plotting.
    """

    row = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "train_policy": float(train_policy),
        "train_value": float(train_value),
        "val_loss": "",
        "val_policy": "",
        "val_value": "",
        "val_top1": "",
        "val_top3": "",
        "val_top5": "",
        "val_stop": "",
        "val_switch": "",
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "learning_rate": float(learning_rate),
    }

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("loss/train_total", train_loss, epoch)
        tensorboard_writer.add_scalar("loss/train_policy", train_policy, epoch)
        tensorboard_writer.add_scalar("loss/train_value", train_value, epoch)
        tensorboard_writer.add_scalar("train/learning_rate", learning_rate, epoch)
        tensorboard_writer.add_scalar("best/best_epoch", best_epoch, epoch)
        tensorboard_writer.add_scalar("best/best_metric", best_metric, epoch)

    if val_metrics is not None:
        row.update(
            {
                "val_loss": float(val_metrics["loss"]),
                "val_policy": float(val_metrics["policy_loss"]),
                "val_value": float(val_metrics["value_loss"]),
                "val_top1": float(val_metrics["top1"]),
                "val_top3": float(val_metrics["top3"]),
                "val_top5": float(val_metrics["top5"]),
                "val_stop": float(val_metrics["stop_acc"]),
                "val_switch": float(val_metrics["switch_acc"]),
            }
        )

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                "loss/val_total",
                val_metrics["loss"],
                epoch,
            )
            tensorboard_writer.add_scalar(
                "loss/val_policy",
                val_metrics["policy_loss"],
                epoch,
            )
            tensorboard_writer.add_scalar(
                "loss/val_value",
                val_metrics["value_loss"],
                epoch,
            )

            tensorboard_writer.add_scalar(
                "accuracy/val_top1",
                val_metrics["top1"],
                epoch,
            )
            tensorboard_writer.add_scalar(
                "accuracy/val_top3",
                val_metrics["top3"],
                epoch,
            )
            tensorboard_writer.add_scalar(
                "accuracy/val_top5",
                val_metrics["top5"],
                epoch,
            )
            tensorboard_writer.add_scalar(
                "accuracy/val_stop",
                val_metrics["stop_acc"],
                epoch,
            )
            tensorboard_writer.add_scalar(
                "accuracy/val_switch",
                val_metrics["switch_acc"],
                epoch,
            )

    with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer_csv.writerow(row)

    if tensorboard_writer is not None:
        tensorboard_writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train graph/GNN policy-value baseline."
    )

    parser.add_argument(
        "examples_csv",
        type=str,
        help="Path to self-play/teacher examples.csv.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Graph hidden dimension.",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of graph message-passing layers.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size.",
    )

    parser.add_argument(
        "--value-scale",
        type=float,
        default=10000.0,
        help="Scale for value targets.",
    )

    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Weight of value loss.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device: auto, cuda, or cpu.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision on CUDA.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. On Windows, 0 is safest. "
            "Try 2-4 only if loading becomes a bottleneck."
        ),
    )

    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Disable graph feature normalization.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/self_play/graph_v1/graph_policy_value_net.pt",
        help="Output checkpoint path.",
    )

    parser.add_argument(
        "--val-examples-csv",
        type=str,
        default=None,
        help="Optional validation examples.csv. If provided, validation runs every epoch.",
    )

    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save best checkpoint by validation loss instead of only the last epoch.",
    )

    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs. If omitted, logs are saved near output checkpoint.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for TensorBoard.",
    )

    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )

    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Optional CSV file for epoch metrics.",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="graph_v1",
        choices=["graph_v1", "graph_v2"],
        help="Graph model architecture: graph_v1 or graph_v2.",
    )

    parser.add_argument(
        "--save-multiple-best",
        action="store_true",
        help=(
            "Save several best checkpoint variants: "
            "best_loss, best_top1, best_top5, best_switch, best_policy, and last."
        ),
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Training graph/GNN policy-value baseline")
    print("=" * 100)

    print(f"Examples CSV:  {args.examples_csv}")
    print(f"Device:        {device}")
    print(f"CUDA available:{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device:   {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:  {torch.version.cuda}")

    print(f"AMP enabled:   {use_amp}")

    dataset = GraphSelfPlayDataset(
        examples_csv=args.examples_csv,
        value_scale=args.value_scale,
        normalize_features=not args.no_normalize_features,
    )

    val_dataset = None

    if args.val_examples_csv is not None:
        val_dataset = GraphSelfPlayDataset(
            examples_csv=args.val_examples_csv,
            value_scale=args.value_scale,
            normalize_features=not args.no_normalize_features,
            normalization_stats={
                "bus_feature_mean": dataset.bus_feature_mean,
                "bus_feature_std": dataset.bus_feature_std,
                "branch_feature_mean": dataset.branch_feature_mean,
                "branch_feature_std": dataset.branch_feature_std,
            },
        )

    print(f"Examples:      {len(dataset)}")
    print(f"Num buses:     {dataset.num_buses}")
    print(f"Num branches:  {dataset.num_branches}")
    print(f"Num actions:   {dataset.num_actions}")
    print(f"Bus features:  {dataset.num_bus_features}")
    print(f"Branch feats:  {dataset.num_branch_features}")
    print(f"Value scale:   {args.value_scale}")

    train_value_diagnostics = build_value_target_diagnostics(
        dataset=dataset,
        value_scale=float(args.value_scale),
    )

    print_value_target_diagnostics(train_value_diagnostics)

    print(f"Batch size:    {args.batch_size}")
    print(f"Num workers:   {args.num_workers}")
    print(f"Hidden dim:    {args.hidden_dim}")
    print(f"Num layers:    {args.num_layers}")
    print(f"Dropout:       {args.dropout}")
    print(f"Model type:    {args.model_type}")

    if val_dataset is not None:
        print(f"Val examples:   {len(val_dataset)}")
        print(f"Val CSV:        {args.val_examples_csv}")

    # This must be before the training loop.
    writer, metrics_csv_path = setup_live_logging(
        args=args,
        output_path=output_path,
    )

    pin_memory = device.type == "cuda"

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    val_loader = None

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=pin_memory,
        )

    if args.model_type == "graph_v2":
        model = GraphPolicyValueNetV2(
            num_bus_features=dataset.num_bus_features,
            num_branch_features=dataset.num_branch_features,
            num_actions=dataset.num_actions,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        model = GraphPolicyValueNet(
            num_bus_features=dataset.num_bus_features,
            num_branch_features=dataset.num_branch_features,
            num_actions=dataset.num_actions,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    value_loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metric = float("inf")
    best_epoch = 0
    best_checkpoint = None

    best_top1 = -float("inf")
    best_top1_epoch = 0

    best_top5 = -float("inf")
    best_top5_epoch = 0

    best_switch = -float("inf")
    best_switch_epoch = 0

    best_policy_score = -float("inf")
    best_policy_score_epoch = 0

    for epoch in range(1, args.epochs + 1):
        total_loss, policy_loss, value_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            value_loss_fn=value_loss_fn,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            value_loss_weight=args.value_loss_weight,
        )

        val_metrics = None

        if val_loader is not None:
            val_metrics = evaluate_one_epoch(
                model=model,
                loader=val_loader,
                value_loss_fn=value_loss_fn,
                device=device,
                use_amp=use_amp,
                value_loss_weight=args.value_loss_weight,
            )

            current_metric = float(val_metrics["loss"])

            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch

                best_checkpoint = make_checkpoint(
                    model=model,
                    dataset=dataset,
                    args=args,
                    device=device,
                    use_amp=use_amp,
                )

            if args.save_multiple_best:
                current_top1 = float(val_metrics["top1"])
                current_top5 = float(val_metrics["top5"])
                current_switch = float(val_metrics["switch_acc"])
                current_policy_score = compute_policy_selection_score(val_metrics)

                # Best by validation loss.
                if current_metric <= best_metric:
                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_loss"),
                        model=model,
                        dataset=dataset,
                        args=args,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="val_loss",
                        selector_value=current_metric,
                        val_metrics=val_metrics,
                    )

                # Best by top-1 accuracy.
                if current_top1 > best_top1:
                    best_top1 = current_top1
                    best_top1_epoch = epoch

                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_top1"),
                        model=model,
                        dataset=dataset,
                        args=args,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="val_top1",
                        selector_value=current_top1,
                        val_metrics=val_metrics,
                    )

                # Best by top-5 accuracy.
                if current_top5 > best_top5:
                    best_top5 = current_top5
                    best_top5_epoch = epoch

                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_top5"),
                        model=model,
                        dataset=dataset,
                        args=args,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="val_top5",
                        selector_value=current_top5,
                        val_metrics=val_metrics,
                    )

                # Best by switch accuracy.
                if current_switch > best_switch:
                    best_switch = current_switch
                    best_switch_epoch = epoch

                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_switch"),
                        model=model,
                        dataset=dataset,
                        args=args,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="val_switch",
                        selector_value=current_switch,
                        val_metrics=val_metrics,
                    )

                # Best by composite policy score.
                if current_policy_score > best_policy_score:
                    best_policy_score = current_policy_score
                    best_policy_score_epoch = epoch

                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_policy"),
                        model=model,
                        dataset=dataset,
                        args=args,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="policy_selection_score",
                        selector_value=current_policy_score,
                        val_metrics=val_metrics,
                    )

            print(
                f"Epoch {epoch:4d} | "
                f"train_loss={total_loss:.6f} | "
                f"train_policy={policy_loss:.6f} | "
                f"train_value={value_loss:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_policy={val_metrics['policy_loss']:.6f} | "
                f"val_value={val_metrics['value_loss']:.6f} | "
                f"val_top1={val_metrics['top1']:.4f} | "
                f"val_top5={val_metrics['top5']:.4f} | "
                f"val_stop={val_metrics['stop_acc']:.4f} | "
                f"val_switch={val_metrics['switch_acc']:.4f} | "
                f"best_epoch={best_epoch}"
            )

        else:
            current_metric = total_loss

            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch

                best_checkpoint = make_checkpoint(
                    model=model,
                    dataset=dataset,
                    args=args,
                    device=device,
                    use_amp=use_amp,
                )

            if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:4d} | "
                    f"loss={total_loss:.6f} | "
                    f"policy_loss={policy_loss:.6f} | "
                    f"value_loss={value_loss:.6f} | "
                    f"best={best_metric:.6f} | "
                    f"best_epoch={best_epoch}"
                )

        learning_rate = float(optimizer.param_groups[0]["lr"])

        log_epoch_metrics(
            tensorboard_writer=writer,
            metrics_csv_path=metrics_csv_path,
            epoch=epoch,
            train_loss=total_loss,
            train_policy=policy_loss,
            train_value=value_loss,
            val_metrics=val_metrics,
            best_epoch=best_epoch,
            best_metric=best_metric,
            learning_rate=learning_rate,
        )

    if args.save_best and best_checkpoint is not None:
        checkpoint = best_checkpoint
        checkpoint["best_epoch"] = int(best_epoch)
        checkpoint["best_metric"] = float(best_metric)
    else:
        checkpoint = make_checkpoint(
            model=model,
            dataset=dataset,
            args=args,
            device=device,
            use_amp=use_amp,
        )
        checkpoint["best_epoch"] = int(best_epoch)
        checkpoint["best_metric"] = float(best_metric)

    torch.save(checkpoint, output_path)

    if args.save_multiple_best:
        last_checkpoint_path = checkpoint_variant_path(output_path, "last")

        last_checkpoint = make_checkpoint(
            model=model,
            dataset=dataset,
            args=args,
            device=device,
            use_amp=use_amp,
        )

        last_checkpoint["saved_epoch"] = int(args.epochs)
        last_checkpoint["selector_name"] = "last_epoch"
        last_checkpoint["selector_value"] = float(args.epochs)

        torch.save(last_checkpoint, last_checkpoint_path)

        print("\nSaved additional checkpoint variants:")
        print(checkpoint_variant_path(output_path, "best_loss"))
        print(checkpoint_variant_path(output_path, "best_top1"))
        print(checkpoint_variant_path(output_path, "best_top5"))
        print(checkpoint_variant_path(output_path, "best_switch"))
        print(checkpoint_variant_path(output_path, "best_policy"))
        print(last_checkpoint_path)

        print("\nBest selector epochs:")
        print(f"  best_loss epoch:   {best_epoch}")
        print(f"  best_top1 epoch:   {best_top1_epoch}")
        print(f"  best_top5 epoch:   {best_top5_epoch}")
        print(f"  best_switch epoch: {best_switch_epoch}")
        print(f"  best_policy epoch: {best_policy_score_epoch}")

    if writer is not None:
        writer.close()

    print("\nSaved graph model:")
    print(output_path)
    print(f"Best epoch:  {best_epoch}")
    print(f"Best metric: {best_metric:.6f}")

    evaluate_training_samples(
        model=model,
        dataset=dataset,
        device=device,
        max_samples=20,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
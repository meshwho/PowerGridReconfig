from __future__ import annotations

import csv
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


DIFFICULTY_CLASSES = ("simple", "medium", "hard", "unknown")

_BASE_DIAGNOSTIC_KEYS = (
    "target_policy_entropy",
    "predicted_policy_entropy",
    "policy_entropy_gap",
    "policy_kl",
    "top1_target_mass",
    "top3_target_mass",
    "top5_target_mass",
    "value_brier",
    "value_calibration_error",
    "value_bias",
    "value_mae",
    "mean_legal_action_count",
    "mean_target_support_size",
    "mean_target_support_fraction",
    "stop_target_fraction",
    "switch_target_fraction",
    "predicted_stop_fraction",
    "predicted_switch_fraction",
)

_DIFFICULTY_METRIC_KEYS = (
    "loss",
    "policy_loss",
    "value_loss",
    "top1",
    "top3",
    "top5",
    "stop_acc",
    "switch_acc",
    *_BASE_DIAGNOSTIC_KEYS,
)

VALIDATION_DIAGNOSTIC_KEYS = (
    *_BASE_DIAGNOSTIC_KEYS,
    *(
        f"difficulty_{difficulty}_{metric}"
        for difficulty in DIFFICULTY_CLASSES
        for metric in ("examples", *_DIFFICULTY_METRIC_KEYS)
    ),
)

_CALIBRATION_BINS: ContextVar[int] = ContextVar(
    "checkpoint_validation_calibration_bins",
    default=10,
)
_LAST_VALIDATION_METRICS: ContextVar[dict[str, float] | None] = ContextVar(
    "checkpoint_last_validation_metrics",
    default=None,
)


def current_validation_metrics() -> dict[str, float] | None:
    metrics = _LAST_VALIDATION_METRICS.get()
    return None if metrics is None else dict(metrics)


def _difficulty_name(value: object) -> str:
    name = str(value).strip().lower()
    return name if name in DIFFICULTY_CLASSES[:-1] else "unknown"


def _scenario_difficulty(loader: DataLoader) -> dict[int, str]:
    dataset = getattr(loader, "dataset", None)
    frame = getattr(dataset, "examples", None)
    columns = getattr(frame, "columns", ())
    if "scenario_id" not in columns or "difficulty_class" not in columns:
        return {}

    mapping: dict[int, str] = {}
    for scenario_id, difficulty in zip(
        frame["scenario_id"].tolist(),
        frame["difficulty_class"].tolist(),
    ):
        key = int(float(scenario_id))
        value = _difficulty_name(difficulty)
        previous = mapping.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                "Validation scenario maps to multiple difficulty classes: "
                f"scenario_id={key}, values={previous!r}/{value!r}."
            )
        mapping[key] = value
    return mapping


def _forward_graph_model(
    model: nn.Module,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {
        "bus_features": batch["bus_features"],
        "branch_features": batch["branch_features"],
        "edge_index": batch["edge_index"],
        "action_mask": batch["action_mask"],
        "edge_active_mask": batch["edge_active_mask"],
        "node_batch": batch.get("node_batch"),
        "edge_batch": batch.get("edge_batch"),
    }
    return model(**kwargs)


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _masked_policy_terms(
    *,
    logits: torch.Tensor,
    target_policy: torch.Tensor,
    action_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    legal = action_mask.bool()
    if logits.ndim != 2 or target_policy.shape != logits.shape:
        raise ValueError("Policy logits and targets must have matching 2D shapes.")
    if legal.shape != logits.shape:
        raise ValueError("action_mask must match policy logits.")
    if not bool(legal.any(dim=1).all()):
        raise ValueError("Every validation example must have a legal action.")

    target = target_policy.float().masked_fill(~legal, 0.0)
    target_mass = target.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(target).all()) or not bool((target_mass > 0).all()):
        raise ValueError("Validation policy targets must have finite positive mass.")
    target = target / target_mass

    masked_logits = logits.float().masked_fill(~legal, -torch.inf)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    probabilities = torch.exp(log_probs).masked_fill(~legal, 0.0)

    positive_target = target > 0.0
    target_log = torch.zeros_like(target)
    target_log[positive_target] = torch.log(target[positive_target])

    target_entropy = -(target * target_log).sum(dim=1)
    predicted_entropy = -(
        probabilities * log_probs.masked_fill(~legal, 0.0)
    ).sum(dim=1)
    policy_loss = -(target * log_probs.masked_fill(~legal, 0.0)).sum(dim=1)
    policy_kl = (
        target
        * (target_log - log_probs.masked_fill(~legal, 0.0))
    ).sum(dim=1)

    target_action = torch.argmax(target, dim=1)
    predicted_action = torch.argmax(masked_logits, dim=1)
    max_k = min(5, int(logits.shape[1]))
    top_indices = torch.topk(masked_logits, k=max_k, dim=1).indices

    def target_mass_at(k: int) -> torch.Tensor:
        width = min(k, max_k)
        return torch.gather(target, 1, top_indices[:, :width]).sum(dim=1)

    support_size = (target > 0.0).sum(dim=1).float()
    legal_count = legal.sum(dim=1).float()

    return {
        "policy_loss": policy_loss,
        "target_policy_entropy": target_entropy,
        "predicted_policy_entropy": predicted_entropy,
        "policy_entropy_gap": predicted_entropy - target_entropy,
        "policy_kl": policy_kl,
        "top1_target_mass": target_mass_at(1),
        "top3_target_mass": target_mass_at(3),
        "top5_target_mass": target_mass_at(5),
        "legal_action_count": legal_count,
        "target_support_size": support_size,
        "target_support_fraction": support_size / legal_count,
        "target_action": target_action,
        "predicted_action": predicted_action,
        "top_indices": top_indices,
    }


def _expected_calibration_error(
    predicted_probability: np.ndarray,
    target_probability: np.ndarray,
    bins: int,
) -> float:
    if predicted_probability.size == 0:
        return 0.0
    indices = np.minimum(
        (predicted_probability * bins).astype(np.int64),
        bins - 1,
    )
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        weight = float(mask.mean())
        error += weight * abs(
            float(predicted_probability[mask].mean())
            - float(target_probability[mask].mean())
        )
    return float(error)


def _mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = values if mask is None else values[mask]
    return float(selected.mean()) if selected.size else 0.0


def _summarize(
    arrays: Mapping[str, np.ndarray],
    *,
    mask: np.ndarray,
    calibration_bins: int,
) -> dict[str, float]:
    target_action = arrays["target_action"]
    predicted_action = arrays["predicted_action"]
    stop = target_action == 0
    switch = ~stop
    masked_stop = mask & stop
    masked_switch = mask & switch

    top_indices = arrays["top_indices"]
    top1 = top_indices[:, :1]
    top3 = top_indices[:, : min(3, top_indices.shape[1])]
    top5 = top_indices[:, : min(5, top_indices.shape[1])]

    result = {
        "loss": _mean(arrays["loss"], mask),
        "policy_loss": _mean(arrays["policy_loss"], mask),
        "value_loss": _mean(arrays["value_loss"], mask),
        "top1": _mean((top1[:, 0] == target_action).astype(float), mask),
        "top3": _mean(
            (top3 == target_action[:, None]).any(axis=1).astype(float),
            mask,
        ),
        "top5": _mean(
            (top5 == target_action[:, None]).any(axis=1).astype(float),
            mask,
        ),
        "stop_acc": _mean(
            (predicted_action == target_action).astype(float),
            masked_stop,
        ),
        "switch_acc": _mean(
            (predicted_action == target_action).astype(float),
            masked_switch,
        ),
        "target_policy_entropy": _mean(arrays["target_policy_entropy"], mask),
        "predicted_policy_entropy": _mean(
            arrays["predicted_policy_entropy"], mask
        ),
        "policy_entropy_gap": _mean(arrays["policy_entropy_gap"], mask),
        "policy_kl": _mean(arrays["policy_kl"], mask),
        "top1_target_mass": _mean(arrays["top1_target_mass"], mask),
        "top3_target_mass": _mean(arrays["top3_target_mass"], mask),
        "top5_target_mass": _mean(arrays["top5_target_mass"], mask),
        "value_brier": _mean(arrays["value_brier"], mask),
        "value_calibration_error": _expected_calibration_error(
            arrays["predicted_probability"][mask],
            arrays["target_probability"][mask],
            calibration_bins,
        ),
        "value_bias": _mean(arrays["value_bias"], mask),
        "value_mae": _mean(arrays["value_mae"], mask),
        "mean_legal_action_count": _mean(arrays["legal_action_count"], mask),
        "mean_target_support_size": _mean(arrays["target_support_size"], mask),
        "mean_target_support_fraction": _mean(
            arrays["target_support_fraction"], mask
        ),
        "stop_target_fraction": _mean(stop.astype(float), mask),
        "switch_target_fraction": _mean(switch.astype(float), mask),
        "predicted_stop_fraction": _mean(
            (predicted_action == 0).astype(float), mask
        ),
        "predicted_switch_fraction": _mean(
            (predicted_action != 0).astype(float), mask
        ),
    }
    return result


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    value_loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    value_loss_weight: float,
) -> dict[str, float]:
    """Evaluate validation loss plus policy, calibration and coverage diagnostics."""

    huber_delta = float(getattr(value_loss_fn, "delta", 1.0))
    model.eval()
    difficulty_by_scenario = _scenario_difficulty(loader)
    collected: dict[str, list[np.ndarray]] = {}

    def append(name: str, tensor: torch.Tensor) -> None:
        collected.setdefault(name, []).append(
            tensor.detach().float().cpu().numpy()
        )

    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            target_policy = batch["target_policy"]
            target_value = batch["target_value"].float().reshape(-1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                policy_logits, predicted_value = _forward_graph_model(model, batch)

            predicted_value = predicted_value.float().reshape(-1)
            policy = _masked_policy_terms(
                logits=policy_logits,
                target_policy=target_policy,
                action_mask=batch["action_mask"],
            )
            if isinstance(value_loss_fn, nn.MSELoss):
                value_loss = (predicted_value - target_value).square()
            elif isinstance(value_loss_fn, nn.L1Loss):
                value_loss = (predicted_value - target_value).abs()
            else:
                value_loss = F.huber_loss(
                    predicted_value,
                    target_value,
                    reduction="none",
                    delta=huber_delta,
                )
            total_loss = policy["policy_loss"] + float(value_loss_weight) * value_loss

            target_probability = ((target_value + 1.0) * 0.5).clamp(0.0, 1.0)
            predicted_probability = (
                (predicted_value + 1.0) * 0.5
            ).clamp(0.0, 1.0)

            append("loss", total_loss)
            append("value_loss", value_loss)
            for name in (
                "policy_loss",
                "target_policy_entropy",
                "predicted_policy_entropy",
                "policy_entropy_gap",
                "policy_kl",
                "top1_target_mass",
                "top3_target_mass",
                "top5_target_mass",
                "legal_action_count",
                "target_support_size",
                "target_support_fraction",
                "target_action",
                "predicted_action",
                "top_indices",
            ):
                append(name, policy[name])
            append("target_probability", target_probability)
            append("predicted_probability", predicted_probability)
            append(
                "value_brier",
                (predicted_probability - target_probability).square(),
            )
            append("value_bias", predicted_value - target_value)
            append("value_mae", (predicted_value - target_value).abs())

            scenario_ids = batch.get("scenario_id")
            if scenario_ids is None:
                difficulties = np.full(
                    target_value.numel(), "unknown", dtype="U7"
                )
            else:
                difficulties = np.array(
                    [
                        difficulty_by_scenario.get(int(value), "unknown")
                        for value in scenario_ids.detach().cpu().tolist()
                    ],
                    dtype="U7",
                )
            collected.setdefault("difficulty", []).append(difficulties)

    if not collected:
        raise RuntimeError("Validation loader produced zero examples.")

    arrays = {
        name: np.concatenate(parts, axis=0)
        for name, parts in collected.items()
    }
    total_examples = int(arrays["loss"].shape[0])
    all_mask = np.ones(total_examples, dtype=bool)
    bins = _CALIBRATION_BINS.get()
    metrics = _summarize(arrays, mask=all_mask, calibration_bins=bins)
    metrics["examples"] = float(total_examples)

    difficulties = arrays["difficulty"]
    for difficulty in DIFFICULTY_CLASSES:
        mask = difficulties == difficulty
        prefix = f"difficulty_{difficulty}_"
        metrics[prefix + "examples"] = float(mask.sum())
        summary = _summarize(arrays, mask=mask, calibration_bins=bins)
        for name, value in summary.items():
            metrics[prefix + name] = value

    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Validation diagnostics contain non-finite values.")

    _LAST_VALIDATION_METRICS.set(dict(metrics))
    return metrics


def attach_validation_metrics(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    metrics = current_validation_metrics()
    if metrics is not None:
        checkpoint["val_metrics"] = metrics
    return checkpoint


def log_epoch_metrics(
    legacy_logger,
    **kwargs: Any,
) -> None:
    """Run legacy logging, then append diagnostics to CSV and TensorBoard."""

    legacy_logger(**kwargs)
    val_metrics = kwargs.get("val_metrics")
    if not isinstance(val_metrics, Mapping):
        return

    metrics_csv_path = Path(kwargs["metrics_csv_path"])
    with metrics_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not rows:
        return

    new_fields = [
        f"val_{name}"
        for name in VALIDATION_DIAGNOSTIC_KEYS
        if f"val_{name}" not in fieldnames
    ]
    fieldnames.extend(new_fields)
    for name in VALIDATION_DIAGNOSTIC_KEYS:
        value = val_metrics.get(name, "")
        rows[-1][f"val_{name}"] = (
            float(value) if isinstance(value, (int, float)) else ""
        )

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tensorboard_writer = kwargs.get("tensorboard_writer")
    if tensorboard_writer is None:
        return
    epoch = int(kwargs["epoch"])
    for name in VALIDATION_DIAGNOSTIC_KEYS:
        value = val_metrics.get(name)
        if not isinstance(value, (int, float)):
            continue
        if name.startswith("difficulty_"):
            _, difficulty, metric = name.split("_", 2)
            tag = f"validation/{difficulty}/{metric}"
        else:
            tag = f"validation/{name}"
        tensorboard_writer.add_scalar(tag, float(value), epoch)
    tensorboard_writer.flush()

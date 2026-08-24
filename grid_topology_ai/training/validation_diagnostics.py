"""Basic validation metrics used by Light training."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _forward(
    model: nn.Module, batch: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        bus_features=batch["bus_features"],
        branch_features=batch["branch_features"],
        edge_index=batch["edge_index"],
        edge_active_mask=batch["edge_active_mask"],
        action_mask=batch["action_mask"],
        node_batch=batch["node_batch"],
        edge_batch=batch["edge_batch"],
    )


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    value_loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    value_loss_weight: float,
) -> dict[str, float]:
    """Return the losses and compact policy feedback required by Light training."""
    model.eval()
    totals = {
        name: 0.0
        for name in (
            "loss",
            "policy_loss",
            "value_loss",
            "top1",
            "top3",
            "top5",
            "stop_acc",
            "switch_acc",
        )
    }
    examples = 0
    stop_examples = 0
    switch_examples = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            target_policy = batch["target_policy"]
            target_value = batch["target_value"].float().reshape(-1)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, predicted_value = _forward(model, batch)
                log_probs = torch.log_softmax(logits, dim=1)
                policy_losses = -(target_policy * log_probs).sum(dim=1)
                value_losses = torch.nn.functional.huber_loss(
                    predicted_value.float().reshape(-1),
                    target_value,
                    reduction="none",
                    delta=float(getattr(value_loss_fn, "delta", 1.0)),
                )
                losses = policy_losses + float(value_loss_weight) * value_losses

            count = int(target_value.numel())
            target_action = target_policy.argmax(dim=1)
            top = logits.topk(min(5, logits.shape[1]), dim=1).indices
            top3 = top[:, : min(3, top.shape[1])]
            predicted_action = top[:, 0]
            stop_mask = target_action == 0
            switch_mask = ~stop_mask
            totals["loss"] += float(losses.sum().item())
            totals["policy_loss"] += float(policy_losses.sum().item())
            totals["value_loss"] += float(value_losses.sum().item())
            totals["top1"] += float((predicted_action == target_action).sum().item())
            totals["top3"] += float(
                (top3 == target_action[:, None]).any(dim=1).sum().item()
            )
            totals["top5"] += float(
                (top == target_action[:, None]).any(dim=1).sum().item()
            )
            totals["stop_acc"] += float(
                ((predicted_action == target_action) & stop_mask).sum().item()
            )
            totals["switch_acc"] += float(
                ((predicted_action == target_action) & switch_mask).sum().item()
            )
            examples += count
            stop_examples += int(stop_mask.sum().item())
            switch_examples += int(switch_mask.sum().item())

    if examples == 0:
        raise RuntimeError("Validation loader produced zero examples.")
    return {
        "loss": totals["loss"] / examples,
        "policy_loss": totals["policy_loss"] / examples,
        "value_loss": totals["value_loss"] / examples,
        "top1": totals["top1"] / examples,
        "top3": totals["top3"] / examples,
        "top5": totals["top5"] / examples,
        "stop_acc": totals["stop_acc"] / max(stop_examples, 1),
        "switch_acc": totals["switch_acc"] / max(switch_examples, 1),
        "examples": float(examples),
    }


def attach_validation_metrics(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Keep checkpoint construction independent of reporting-only metrics."""
    return checkpoint


def log_epoch_metrics(base_logger: Callable[..., None], **kwargs: Any) -> None:
    """Delegate compact validation logging to the standard epoch logger."""
    base_logger(**kwargs)

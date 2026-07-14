from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset

if TYPE_CHECKING:
    from grid_topology_ai.training.graph_policy_value import TrainingRequest


def build_value_target_diagnostics(
    dataset: GraphSelfPlayDataset,
) -> dict[str, Any]:
    """
    Analyze strict outcome_value_target.

    The project no longer supports legacy value fallback.
    Every training row must contain outcome_value_target in [-1, 1].
    """

    def q(x, p):
        return float(np.quantile(x, p)) if len(x) > 0 else 0.0

    if "outcome_value_target" not in dataset.examples.columns:
        return {
            "available": False,
            "reason": "outcome_value_target column is missing.",
        }

    target = dataset.examples["outcome_value_target"].astype(float).to_numpy()
    abs_target = np.abs(target)
    outside_mask = abs_target > 1.0

    return {
        "available": True,
        "mode": "outcome_value_target",
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


def print_value_target_diagnostics(
    diagnostics: dict[str, Any],
) -> None:
    """
    Print value target diagnostics in a compact readable form.

    Current training uses outcome_value_target as the strict value target.
    This target is already bounded to [-1, 1], so no legacy value_scale is needed.
    """

    print("")
    print("=" * 100)
    print("VALUE TARGET DIAGNOSTICS")
    print("=" * 100)

    if not diagnostics.get("available", False):
        print(f"Unavailable: {diagnostics.get('reason')}")
        return

    print(f"mode:               {diagnostics.get('mode', 'unknown')}")
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
    print(f"outside [-1, 1]:    {diagnostics['outside_minus1_plus1_count']}")
    print(
        f"outside percent:    "
        f"{diagnostics['outside_minus1_plus1_percent']:.2f}%"
    )
    print("")
    print(f"positive count:     {diagnostics['positive_count']}")
    print(f"zero count:         {diagnostics['zero_count']}")
    print(f"negative count:     {diagnostics['negative_count']}")

    if diagnostics["outside_minus1_plus1_count"] > 0:
        print("")
        print("WARNING: Some outcome_value_target values are outside [-1, 1].")


def compute_policy_selection_score(
    val_metrics: dict[str, float],
) -> float:
    """
    Composite score for selecting policy-useful checkpoints.
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


def setup_live_logging(
    request: "TrainingRequest",
    output_path: Path,
) -> tuple[Any, Path]:
    """
    Initialize TensorBoard writer and metrics CSV before the training loop.

    This must happen before log_epoch_metrics() is called.
    """

    run_name = request.run_name

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"graph_train_{timestamp}"

    if request.tensorboard_log_dir is None:
        tensorboard_root = output_path.parent / "tensorboard"
    else:
        tensorboard_root = request.tensorboard_log_dir

    tensorboard_dir = tensorboard_root / run_name

    writer = None

    if not request.config.no_tensorboard:
        if SummaryWriter is None:
            print(
                "WARNING: TensorBoard is not installed. "
                "Run: python -m pip install tensorboard"
            )
        else:
            writer = SummaryWriter(log_dir=str(tensorboard_dir))
            print(f"TensorBoard log dir: {tensorboard_dir}")

    if request.metrics_csv is None:
        metrics_csv_path = output_path.parent / f"{run_name}_metrics.csv"
    else:
        metrics_csv_path = request.metrics_csv

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
            tensorboard_writer.add_scalar("loss/val_total", val_metrics["loss"], epoch)
            tensorboard_writer.add_scalar(
                "loss/val_policy", val_metrics["policy_loss"], epoch
            )
            tensorboard_writer.add_scalar(
                "loss/val_value", val_metrics["value_loss"], epoch
            )
            tensorboard_writer.add_scalar("accuracy/val_top1", val_metrics["top1"], epoch)
            tensorboard_writer.add_scalar("accuracy/val_top3", val_metrics["top3"], epoch)
            tensorboard_writer.add_scalar("accuracy/val_top5", val_metrics["top5"], epoch)
            tensorboard_writer.add_scalar(
                "accuracy/val_stop", val_metrics["stop_acc"], epoch
            )
            tensorboard_writer.add_scalar(
                "accuracy/val_switch", val_metrics["switch_acc"], epoch
            )

    with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer_csv.writerow(row)

    if tensorboard_writer is not None:
        tensorboard_writer.flush()

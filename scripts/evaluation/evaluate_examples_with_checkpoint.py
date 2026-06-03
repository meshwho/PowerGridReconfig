from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from grid_topology_ai.models.graph_policy_value_net import GraphPolicyValueNet
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from scripts.self_play.train_graph_baseline import (
    evaluate_one_epoch,
    resolve_device,
)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model_type = str(checkpoint.get("model_type", "graph_policy_value_net"))

    common_kwargs = dict(
        num_bus_features=int(checkpoint["num_bus_features"]),
        num_branch_features=int(checkpoint["num_branch_features"]),
        num_actions=int(checkpoint["num_actions"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_layers=int(checkpoint.get("num_layers", 3)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )

    if model_type in {"graph_v2", "graph_policy_value_net_v2"}:
        model = GraphPolicyValueNetV2(**common_kwargs)
    else:
        model = GraphPolicyValueNet(**common_kwargs)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def apply_checkpoint_normalization(
    dataset: GraphSelfPlayDataset,
    checkpoint: dict[str, Any],
) -> None:
    dataset.bus_feature_mean = checkpoint["bus_feature_mean"]
    dataset.bus_feature_std = checkpoint["bus_feature_std"]
    dataset.branch_feature_mean = checkpoint["branch_feature_mean"]
    dataset.branch_feature_std = checkpoint["branch_feature_std"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate teacher examples against an existing graph checkpoint."
    )

    parser.add_argument("--examples-csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--value-scale", type=float, default=10000.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    checkpoint_path = Path(args.checkpoint)
    examples_csv = Path(args.examples_csv)

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    # normalize_features=False prevents recomputing stats from this test file.
    # We immediately replace normalization stats with those saved in the checkpoint.
    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        value_scale=float(args.value_scale),
        normalize_features=False,
    )

    apply_checkpoint_normalization(dataset, checkpoint)

    loader = DataLoader(
        dataset,
        batch_size=min(int(args.batch_size), len(dataset)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    value_loss_fn = nn.MSELoss()

    metrics = evaluate_one_epoch(
        model=model,
        loader=loader,
        value_loss_fn=value_loss_fn,
        device=device,
        use_amp=use_amp,
        value_loss_weight=float(args.value_loss_weight),
    )

    print("=" * 100)
    print("Checkpoint evaluation on examples")
    print("=" * 100)
    print(f"Examples CSV: {examples_csv}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Model type:   {checkpoint.get('model_type', 'unknown')}")
    print(f"Examples:     {int(metrics['examples'])}")
    print("")
    print(f"loss:         {metrics['loss']:.6f}")
    print(f"policy_loss:  {metrics['policy_loss']:.6f}")
    print(f"value_loss:   {metrics['value_loss']:.6f}")
    print(f"top1:         {metrics['top1']:.4f}")
    print(f"top3:         {metrics['top3']:.4f}")
    print(f"top5:         {metrics['top5']:.4f}")
    print(f"stop_acc:     {metrics['stop_acc']:.4f}")
    print(f"switch_acc:   {metrics['switch_acc']:.4f}")


if __name__ == "__main__":
    main()
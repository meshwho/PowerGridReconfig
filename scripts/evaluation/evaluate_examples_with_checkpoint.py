from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from grid_topology_ai.contracts import require_topology_action_provenance
from grid_topology_ai.models.graph_batch import collate_graph_samples
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.training.checkpoints import load_checkpoint_payload
from grid_topology_ai.training.graph_policy_value import (
    evaluate_one_epoch,
    resolve_device,
)


_MODEL_TYPE = "graph_policy_value_net_v2"


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[GraphPolicyValueNetV2, dict[str, Any]]:
    checkpoint = dict(
        load_checkpoint_payload(
            checkpoint_path,
            map_location=device,
        )
    )
    model_type = str(checkpoint.get("model_type", "")).strip()
    if model_type != _MODEL_TYPE:
        raise ValueError(
            "Checkpoint evaluation requires Graph V2, "
            f"got model_type={model_type!r}."
        )

    model = GraphPolicyValueNetV2(
        num_bus_features=int(checkpoint["num_bus_features"]),
        num_branch_features=int(checkpoint["num_branch_features"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_layers=int(checkpoint.get("num_layers", 3)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
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


def validate_checkpoint_dataset_compatibility(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    dataset: GraphSelfPlayDataset,
) -> None:
    mismatches = [
        name
        for name, expected in {
            "num_bus_features": dataset.num_bus_features,
            "num_branch_features": dataset.num_branch_features,
        }.items()
        if int(checkpoint.get(name, -1)) != int(expected)
    ]
    if mismatches:
        raise ValueError(
            "Examples are incompatible with the checkpoint: "
            + ", ".join(mismatches)
            + "."
        )

    require_topology_action_provenance(
        checkpoint,
        source=str(checkpoint_path),
        expected_action_space_config=dataset.topology_action_config,
    )

    if checkpoint.get("policy_layout") != dataset.policy_layout:
        raise ValueError(
            "Checkpoint policy layout does not match the examples dataset. "
            f"Checkpoint: {checkpoint_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate teacher examples against a Graph V2 checkpoint."
    )
    parser.add_argument("--examples-csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    checkpoint_path = Path(args.checkpoint)
    examples_csv = Path(args.examples_csv)

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )
    apply_checkpoint_normalization(dataset, checkpoint)
    validate_checkpoint_dataset_compatibility(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        dataset=dataset,
    )

    loader = DataLoader(
        dataset,
        batch_size=min(int(args.batch_size), len(dataset)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_graph_samples,
    )

    metrics = evaluate_one_epoch(
        model=model,
        loader=loader,
        value_loss_fn=nn.MSELoss(),
        device=device,
        use_amp=use_amp,
        value_loss_weight=float(args.value_loss_weight),
    )

    print("=" * 100)
    print("Checkpoint evaluation on examples")
    print("=" * 100)
    print(f"Examples CSV: {examples_csv}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Examples:     {int(metrics['examples'])}")
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

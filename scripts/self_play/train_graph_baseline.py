from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.training.graph_policy_value import (
    TrainingRequest,
    train_graph_policy_value_model,
)


def build_parser() -> argparse.ArgumentParser:
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
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Weight of value loss.",
    )
    parser.add_argument(
        "--value-huber-delta",
        type=float,
        default=0.5,
        help="Delta parameter for Huber loss used by the value head.",
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
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint used to initialize model weights before training. "
            "This enables fine-tuning in the iterative self-play loop."
        ),
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        value_loss_weight=args.value_loss_weight,
        value_huber_delta=args.value_huber_delta,
        num_workers=args.num_workers,
        device=args.device,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        save_multiple_best=args.save_multiple_best,
        no_tensorboard=args.no_tensorboard,
    )
    request = TrainingRequest(
        project_root=Path.cwd().resolve(),
        examples_csv=Path(args.examples_csv),
        output_path=Path(args.output),
        config=config,
        init_checkpoint=(
            None if args.init_checkpoint is None else Path(args.init_checkpoint)
        ),
        validation_examples_csv=(
            None
            if args.val_examples_csv is None
            else Path(args.val_examples_csv)
        ),
        use_amp=args.amp,
        normalize_features=not args.no_normalize_features,
        save_best=args.save_best,
        tensorboard_log_dir=(
            None
            if args.tensorboard_log_dir is None
            else Path(args.tensorboard_log_dir)
        ),
        run_name=args.run_name,
        metrics_csv=(None if args.metrics_csv is None else Path(args.metrics_csv)),
    )
    checkpoint = train_graph_policy_value_model(request)
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

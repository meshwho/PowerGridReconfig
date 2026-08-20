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
        description="Train the Graph V2 policy-value network."
    )
    parser.add_argument(
        "examples_csv",
        type=str,
        help="Path to self-play/teacher examples.csv.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-huber-delta", type=float, default=0.5)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="data/self_play/graph_v2/graph_policy_value_net.pt",
    )
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--val-examples-csv", type=str, default=None)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--tensorboard-log-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--save-multiple-best", action="store_true")
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
        seed=args.seed,
    )
    checkpoint = train_graph_policy_value_model(request)
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from grid_topology_ai.models.self_play_dataset import SelfPlayDataset
from grid_topology_ai.models.simple_policy_value_net import SimplePolicyValueNet


def soft_policy_loss(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy with soft MCTS policy target.

    loss = -sum_a pi(a) * log p(a)
    """

    log_probs = torch.log_softmax(logits, dim=1)

    loss = -(target_policy * log_probs).sum(dim=1).mean()

    return loss


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train simple AlphaZero-like policy-value baseline."
    )

    parser.add_argument(
        "examples_csv",
        type=str,
        help="Path to self-play examples.csv.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
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
        help="Hidden layer size.",
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
        default=1000.0,
        help="Scale for value targets.",
    )

    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Weight of value loss.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/self_play/mcts_v1/simple_policy_value_net.pt",
        help="Output checkpoint path.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("Training simple AlphaZero-like policy-value baseline")
    print("=" * 100)

    dataset = SelfPlayDataset(
        examples_csv=args.examples_csv,
        value_scale=args.value_scale,
    )

    first = dataset[0]

    input_dim = int(first["state_vector"].shape[0])
    num_actions = int(first["action_mask"].shape[0])

    print(f"Examples:      {len(dataset)}")
    print(f"Input dim:     {input_dim}")
    print(f"Num actions:   {num_actions}")
    print(f"Value scale:   {args.value_scale}")

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
    )

    model = SimplePolicyValueNet(
        input_dim=input_dim,
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    value_loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0

        batches = 0

        for batch in loader:
            state_vector = batch["state_vector"]
            action_mask = batch["action_mask"]
            target_policy = batch["target_policy"]
            target_value = batch["target_value"]

            policy_logits, predicted_value = model(
                state_vector=state_vector,
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

            total_loss = policy_loss + args.value_loss_weight * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_sum += float(total_loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            batches += 1

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"loss={total_loss_sum / batches:.6f} | "
                f"policy_loss={policy_loss_sum / batches:.6f} | "
                f"value_loss={value_loss_sum / batches:.6f}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_actions": num_actions,
            "hidden_dim": args.hidden_dim,
            "examples_csv": str(args.examples_csv),
            "value_scale": args.value_scale,

            # Required for using the model inside MCTS.
            # The model was trained on normalized flat state vectors.
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        },
        output_path,
    )

    print("\nSaved model:")
    print(output_path)

    print("\nFinal predictions:")
    evaluate_model(model, dataset)

    print("\nDone.")


def evaluate_model(
    model: SimplePolicyValueNet,
    dataset: SelfPlayDataset,
) -> None:
    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            state_vector = sample["state_vector"].unsqueeze(0)
            action_mask = sample["action_mask"].unsqueeze(0)
            target_policy = sample["target_policy"].unsqueeze(0)
            target_value = float(sample["target_value"].item())

            logits, value = model(
                state_vector=state_vector,
                action_mask=action_mask,
            )

            probabilities = torch.softmax(logits, dim=1)

            predicted_action = int(torch.argmax(probabilities, dim=1).item())
            target_top_action = int(torch.argmax(target_policy, dim=1).item())

            predicted_value = float(value.item())

            predicted_prob = float(probabilities[0, predicted_action].item())
            target_prob = float(target_policy[0, target_top_action].item())

            print(
                f"Scenario {sample['scenario_id']:>3} | "
                f"step={sample['step']:>2} | "
                f"{sample['state_id']} | "
                f"target_top={target_top_action:>3} "
                f"(pi={target_prob:.3f}) | "
                f"pred_top={predicted_action:>3} "
                f"(p={predicted_prob:.3f}) | "
                f"value target={target_value:+.3f} | "
                f"value pred={predicted_value:+.3f}"
            )


if __name__ == "__main__":
    main()
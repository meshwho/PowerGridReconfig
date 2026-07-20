from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.self_play.example_validation import (
    validate_example_contract_versions,
    validate_example_outcome_contracts,
)


@dataclass
class Metrics:
    total: int = 0
    top1: int = 0
    top3: int = 0
    top5: int = 0
    stop_total: int = 0
    stop_correct: int = 0
    switch_total: int = 0
    switch_correct: int = 0
    value_abs_error_sum: float = 0.0

    def update(
        self,
        target_action: int,
        predicted_top: int,
        top_actions: list[int],
        target_value: float,
        predicted_value: float,
    ) -> None:
        self.total += 1

        if predicted_top == target_action:
            self.top1 += 1

        if target_action in top_actions[:3]:
            self.top3 += 1

        if target_action in top_actions[:5]:
            self.top5 += 1

        if target_action == 0:
            self.stop_total += 1
            if predicted_top == 0:
                self.stop_correct += 1
        else:
            self.switch_total += 1
            if predicted_top == target_action:
                self.switch_correct += 1

        self.value_abs_error_sum += abs(float(predicted_value) - float(target_value))

    def as_dict(self) -> dict[str, float | int]:
        if self.total == 0:
            return {
                "total": 0,
                "top1_acc": 0.0,
                "top3_acc": 0.0,
                "top5_acc": 0.0,
                "stop_acc": 0.0,
                "switch_acc": 0.0,
                "mae_value": 0.0,
            }

        return {
            "total": self.total,
            "top1_acc": self.top1 / self.total,
            "top3_acc": self.top3 / self.total,
            "top5_acc": self.top5 / self.total,
            "stop_total": self.stop_total,
            "stop_acc": (
                self.stop_correct / self.stop_total
                if self.stop_total > 0
                else 0.0
            ),
            "switch_total": self.switch_total,
            "switch_acc": (
                self.switch_correct / self.switch_total
                if self.switch_total > 0
                else 0.0
            ),
            "mae_value": self.value_abs_error_sum / self.total,
        }


def load_state_from_npz(state_path: Path) -> tuple[GridFMState, np.ndarray]:
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    data = np.load(state_path, allow_pickle=False)

    bus_features = data["bus_features"].astype(np.float32)
    branch_features = data["branch_features"].astype(np.float32)
    edge_index = data["edge_index"].astype(np.int64)
    branch_ids = data["branch_ids"].astype(np.int64)
    branch_status = data["branch_status"].astype(np.float32)
    action_mask = data["action_mask"].astype(bool)

    metrics = json.loads(str(data["metrics_json"]))
    metadata = json.loads(str(data["metadata_json"]))

    state = GridFMState(
        scenario_id=int(metadata["scenario_id"]),
        load_scenario_idx=float(metadata.get("load_scenario_idx", 0.0)),
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        branch_ids=branch_ids,
        branch_status=branch_status,
        metrics=metrics,
        outaged_branch_ids=[
            int(x) for x in metadata.get("outaged_branch_ids", [])
        ],
    )

    return state, action_mask


def policy_json_to_vector(
    policy_json: str,
    num_actions: int,
) -> np.ndarray:
    policy_dict = json.loads(policy_json)
    policy = np.zeros(num_actions, dtype=np.float32)

    for action_id_str, probability in policy_dict.items():
        action_id = int(action_id_str)

        if 0 <= action_id < num_actions:
            policy[action_id] = float(probability)

    total = float(policy.sum())

    if total > 0.0:
        policy = policy / total

    return policy


def get_target_action(row: pd.Series, num_actions: int) -> int:
    if "selected_action_id" in row:
        return int(row["selected_action_id"])

    policy = policy_json_to_vector(
        policy_json=str(row["mcts_policy_json"]),
        num_actions=num_actions,
    )

    return int(np.argmax(policy))


def split_scenarios(
    scenario_ids: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    scenario_ids = sorted(int(x) for x in scenario_ids)

    rng = np.random.default_rng(int(seed))
    shuffled = np.array(scenario_ids, dtype=np.int64)
    rng.shuffle(shuffled)

    n_val = int(round(len(shuffled) * float(val_fraction)))
    n_val = max(1, n_val) if len(shuffled) > 1 and val_fraction > 0 else 0

    val = set(int(x) for x in shuffled[:n_val])
    train = set(int(x) for x in shuffled[n_val:])

    return train, val


def print_metrics(
    title: str,
    metrics: Metrics,
) -> None:
    d = metrics.as_dict()

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"Total examples:     {d['total']}")
    print(f"Top-1 accuracy:     {d['top1_acc']:.4f}")
    print(f"Top-3 accuracy:     {d['top3_acc']:.4f}")
    print(f"Top-5 accuracy:     {d['top5_acc']:.4f}")
    print(f"Stop examples:      {d.get('stop_total', 0)}")
    print(f"Stop accuracy:      {d['stop_acc']:.4f}")
    print(f"Switch examples:    {d.get('switch_total', 0)}")
    print(f"Switch accuracy:    {d['switch_acc']:.4f}")
    print(f"Value MAE:          {d['mae_value']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate supervised policy imitation accuracy."
    )

    parser.add_argument(
        "examples_csv",
        type=str,
        help="Path to teacher/self-play examples.csv.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to MLP or graph policy-value checkpoint.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda.",
    )

    parser.add_argument(
        "--value-scale",
        type=float,
        default=10000.0,
        help="Same value scale used during training.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Scenario-level validation fraction for reporting.",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for scenario-level split.",
    )

    parser.add_argument(
        "--show-errors",
        type=int,
        default=20,
        help="How many top-1 errors to print.",
    )

    args = parser.parse_args()

    examples_path = Path(args.examples_csv)
    checkpoint_path = Path(args.checkpoint)

    df = pd.read_csv(examples_path)

    if df.empty:
        raise ValueError(f"Examples CSV is empty: {examples_path}")
    validate_example_contract_versions(df, source_path=examples_path)
    validate_example_outcome_contracts(df, source_path=examples_path)

    evaluator = NeuralPolicyValueEvaluator(
        checkpoint_path=checkpoint_path,
        device=args.device,
        enable_cache=False,
    )

    scenario_ids = sorted(int(x) for x in df["scenario_id"].unique())
    train_scenarios, val_scenarios = split_scenarios(
        scenario_ids=scenario_ids,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )

    print("=" * 100)
    print("Supervised policy evaluation")
    print("=" * 100)
    print(f"Examples CSV:   {examples_path}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Device:         {args.device}")
    print(f"Model type:     {evaluator.model_type}")
    print(f"Examples:       {len(df)}")
    print(f"Scenarios:      {len(scenario_ids)}")
    print(f"Train scenarios:{len(train_scenarios)}")
    print(f"Val scenarios:  {len(val_scenarios)}")
    print()
    print(
        "Note: this split is diagnostic only unless the model was actually "
        "trained with the same train/val split."
    )

    all_metrics = Metrics()
    train_metrics = Metrics()
    val_metrics = Metrics()
    by_step: dict[int, Metrics] = {}

    errors: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        state_path = Path(str(row["state_path"]))
        state, action_mask = load_state_from_npz(state_path)

        policy, predicted_value = evaluator.evaluate(
            state=state,
            action_mask=action_mask,
        )

        num_actions = int(action_mask.shape[0])

        target_policy = policy_json_to_vector(
            policy_json=str(row["mcts_policy_json"]),
            num_actions=num_actions,
        )

        target_policy = target_policy * action_mask.astype(np.float32)

        target_sum = float(target_policy.sum())

        if target_sum > 0.0:
            target_policy = target_policy / target_sum

        target_action = get_target_action(row, num_actions=num_actions)
        predicted_top = int(np.argmax(policy))

        top_actions = [
            int(x)
            for x in np.argsort(policy)[::-1][:10]
        ]

        raw_value = float(row["discounted_return_from_step"])
        target_value = float(np.clip(raw_value / args.value_scale, -1.0, 1.0))

        step = int(row["step"])
        scenario_id = int(row["scenario_id"])

        all_metrics.update(
            target_action=target_action,
            predicted_top=predicted_top,
            top_actions=top_actions,
            target_value=target_value,
            predicted_value=predicted_value,
        )

        if step not in by_step:
            by_step[step] = Metrics()

        by_step[step].update(
            target_action=target_action,
            predicted_top=predicted_top,
            top_actions=top_actions,
            target_value=target_value,
            predicted_value=predicted_value,
        )

        if scenario_id in val_scenarios:
            val_metrics.update(
                target_action=target_action,
                predicted_top=predicted_top,
                top_actions=top_actions,
                target_value=target_value,
                predicted_value=predicted_value,
            )
        else:
            train_metrics.update(
                target_action=target_action,
                predicted_top=predicted_top,
                top_actions=top_actions,
                target_value=target_value,
                predicted_value=predicted_value,
            )

        if predicted_top != target_action and len(errors) < int(args.show_errors):
            errors.append(
                {
                    "scenario_id": scenario_id,
                    "step": step,
                    "state_id": str(row["state_id"]),
                    "target_action": int(target_action),
                    "predicted_top": int(predicted_top),
                    "target_prob": float(target_policy[target_action]),
                    "predicted_prob": float(policy[predicted_top]),
                    "target_in_top5": bool(target_action in top_actions[:5]),
                    "top5": top_actions[:5],
                }
            )

    print_metrics("All examples", all_metrics)
    print_metrics("Diagnostic train split", train_metrics)
    print_metrics("Diagnostic validation split", val_metrics)

    print("\n" + "=" * 100)
    print("By step")
    print("=" * 100)

    for step, metrics in sorted(by_step.items()):
        d = metrics.as_dict()
        print(
            f"step={step:>2} | "
            f"n={d['total']:>4} | "
            f"top1={d['top1_acc']:.4f} | "
            f"top5={d['top5_acc']:.4f} | "
            f"stop_acc={d['stop_acc']:.4f} | "
            f"switch_acc={d['switch_acc']:.4f} | "
            f"value_mae={d['mae_value']:.4f}"
        )

    if errors:
        print("\n" + "=" * 100)
        print("First top-1 errors")
        print("=" * 100)

        for err in errors:
            print(
                f"scenario={err['scenario_id']} | "
                f"step={err['step']} | "
                f"target={err['target_action']} | "
                f"pred={err['predicted_top']} | "
                f"target_in_top5={err['target_in_top5']} | "
                f"top5={err['top5']} | "
                f"state={err['state_id']}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

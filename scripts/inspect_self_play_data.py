from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grid_topology_ai.self_play.example_validation import (
    validate_example_contract_versions,
    validate_example_outcome_contracts,
)


def load_json_dict(value: str) -> dict:
    if pd.isna(value):
        return {}

    return json.loads(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect AlphaZero-like self-play replay buffer."
    )

    parser.add_argument(
        "examples_csv",
        type=str,
        help="Path to self-play examples.csv.",
    )

    parser.add_argument(
        "--top-actions",
        type=int,
        default=8,
        help="Number of top MCTS policy actions to print per example.",
    )

    args = parser.parse_args()

    examples_path = Path(args.examples_csv)

    if not examples_path.exists():
        raise FileNotFoundError(f"Examples file not found: {examples_path}")

    df = pd.read_csv(examples_path)

    if not df.empty:
        validate_example_contract_versions(df, source_path=examples_path)
        validate_example_outcome_contracts(df, source_path=examples_path)

    print("=" * 100)
    print("Inspecting self-play replay buffer")
    print("=" * 100)

    print(f"Examples file: {examples_path.resolve()}")
    print(f"Rows:          {len(df)}")

    if df.empty:
        print("Replay buffer is empty.")
        return

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nScenarios:")
    print(df["scenario_id"].value_counts().sort_index().to_string())

    print("\nTermination reasons:")
    print(df["termination_reason"].value_counts(dropna=False).to_string())

    print("\nSolved:")
    print(df["solved"].value_counts(dropna=False).to_string())

    print("\nReturn statistics:")
    print(df[["final_return", "discounted_return_from_step", "step_reward"]].describe())

    missing_states = []

    for path in df["state_path"]:
        if not Path(path).exists():
            missing_states.append(path)

    print("\nState files:")
    print(f"  Missing state files: {len(missing_states)}")

    if missing_states:
        for path in missing_states[:10]:
            print(f"    - {path}")

    print("\nSample examples:")
    print("-" * 100)

    for _, row in df.sort_values(["scenario_id", "step"]).iterrows():
        policy = load_json_dict(row["mcts_policy_json"])
        visit_counts = load_json_dict(row["visit_counts_json"])

        policy_items = sorted(
            policy.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        top_policy = policy_items[: args.top_actions]

        state_path = Path(str(row["state_path"]))

        print(
            f"Scenario {int(row['scenario_id']):>3} | "
            f"step={int(row['step']):>2} | "
            f"state={row['state_id']} | "
            f"selected_action={int(row['selected_action_id'])} | "
            f"selected_branch={row['selected_branch_id']} | "
            f"step_reward={float(row['step_reward']):>9.3f} | "
            f"return_from_step={float(row['discounted_return_from_step']):>9.3f} | "
            f"final_return={float(row['final_return']):>9.3f} | "
            f"solved={row['solved']} | "
            f"reason={row['termination_reason']}"
        )

        print(f"  state_file_exists={state_path.exists()}")

        print("  top MCTS policy actions:")

        for action_id, prob in top_policy:
            visits = visit_counts.get(str(action_id), 0)

            print(
                f"    action={int(action_id):>3} | "
                f"pi={float(prob):.4f} | "
                f"visits={int(visits):>4}"
            )

        print("-" * 100)

    print("\nDone.")


if __name__ == "__main__":
    main()

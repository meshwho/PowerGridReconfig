from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build supervised policy targets from transition dataset."
    )

    parser.add_argument(
        "transitions",
        type=str,
        help="Path to transitions CSV.",
    )

    parser.add_argument(
        "--states-summary",
        type=str,
        required=True,
        help="Path to states_summary.csv.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/training/policy_targets.csv",
        help="Output policy targets CSV.",
    )

    args = parser.parse_args()

    transitions_path = Path(args.transitions)
    states_summary_path = Path(args.states_summary)
    output_path = Path(args.output)

    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")

    if not states_summary_path.exists():
        raise FileNotFoundError(f"States summary file not found: {states_summary_path}")

    transitions = pd.read_csv(transitions_path)
    states_summary = pd.read_csv(states_summary_path)

    print("=" * 100)
    print("Building policy targets")
    print("=" * 100)

    print(f"Transitions:    {transitions_path.resolve()}")
    print(f"States summary: {states_summary_path.resolve()}")

    # Keep only transitions with successful power flow.
    successful = transitions[transitions["power_flow_success"] == True].copy()

    if successful.empty:
        raise RuntimeError("No successful transitions found.")

    # For each scenario choose the action with maximum reward.
    best_indices = successful.groupby("scenario_id")["reward"].idxmax()
    best_actions = successful.loc[best_indices].copy()

    best_actions = best_actions.sort_values("scenario_id").reset_index(drop=True)

    # Attach state_id and state_path.
    targets = best_actions.merge(
        states_summary[
            [
                "scenario_id",
                "state_id",
                "state_path",
                "num_actions",
                "num_valid_actions",
                "max_loading_percent",
            ]
        ],
        on="scenario_id",
        how="left",
    )

    missing_states = targets["state_id"].isna().sum()

    if missing_states > 0:
        raise RuntimeError(
            f"{missing_states} target rows do not have matching exported states."
        )

    targets = targets[
        [
            "scenario_id",
            "state_id",
            "state_path",
            "action_id",
            "action_type",
            "branch_id",
            "branch_pos",
            "reward",
            "done",
            "before_max_loading",
            "after_max_loading",
            "before_total_overload",
            "after_total_overload",
            "before_num_overloaded",
            "after_num_overloaded",
            "before_num_hard_overloaded",
            "after_num_hard_overloaded",
            "num_actions",
            "num_valid_actions",
            "max_loading_percent",
        ]
    ].copy()

    targets = targets.rename(
        columns={
            "action_id": "target_action_id",
            "action_type": "target_action_type",
            "branch_id": "target_branch_id",
            "branch_pos": "target_branch_pos",
            "reward": "target_reward",
            "done": "target_done",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_path, index=False)

    pretty = targets.copy()


    pretty["state_file"] = pretty["state_path"].apply(lambda p: Path(str(p)).name)


    for col in ["target_branch_id", "target_branch_pos"]:
        pretty[col] = pretty[col].apply(
            lambda x: "-" if pd.isna(x) else str(int(x))
        )


    float_cols = [
        "target_reward",
        "before_max_loading",
        "after_max_loading",
        "before_total_overload",
        "after_total_overload",
        "max_loading_percent",
    ]
    for col in float_cols:
        pretty[col] = pretty[col].map(lambda x: f"{float(x):.3f}")

    summary_cols = [
        "scenario_id",
        "state_id",
        "state_file",
        "target_action_id",
        "target_action_type",
        "target_branch_id",
        "target_branch_pos",
        "target_reward",
        "target_done",
        "before_max_loading",
        "after_max_loading",
        "before_total_overload",
        "after_total_overload",
        "num_actions",
        "num_valid_actions",
    ]

    print("\nPolicy targets summary:")
    print(pretty[summary_cols].to_string(index=False))

    print("\nQuick summary:")
    print(f"  Number of states:          {len(targets)}")
    print(f"  Done targets:              {int(targets['target_done'].sum())}")
    print(f"  Average target reward:     {targets['target_reward'].mean():.3f}")
    print(f"  Max target reward:         {targets['target_reward'].max():.3f}")
    print(f"  Min target reward:         {targets['target_reward'].min():.3f}")

    print("\nSaved:")
    print(output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
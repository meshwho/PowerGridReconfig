from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze generated GridFM transition dataset."
    )

    parser.add_argument(
        "transitions_csv",
        type=str,
        help="Path to transitions.csv file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save analysis CSV files.",
    )

    args = parser.parse_args()

    path = Path(args.transitions_csv)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print("=" * 100)
    print("GridFM transition dataset analysis")
    print("=" * 100)

    print(f"File: {path.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Scenarios: {df['scenario_id'].nunique()}")
    print(f"Actions per scenario:")
    print(df.groupby("scenario_id")["action_id"].count())

    print("\nReward statistics:")
    print(df["reward"].describe())

    print("\nPower flow success:")
    print(df["power_flow_success"].value_counts(dropna=False))

    print("\nDone:")
    print(df["done"].value_counts(dropna=False))

    print("\nReward sign counts:")
    print(f"Positive rewards: {(df['reward'] > 0).sum()}")
    print(f"Zero rewards:     {(df['reward'] == 0).sum()}")
    print(f"Negative rewards: {(df['reward'] < 0).sum()}")

    switch_df = df[df["action_type"] == "switch_off_branch"].copy()
    do_nothing_df = df[df["action_type"] == "do_nothing"].copy()

    print("\nSwitch action reward statistics:")
    if len(switch_df) > 0:
        print(switch_df["reward"].describe())
    else:
        print("No switch actions found.")

    print("\nDo-nothing reward statistics:")
    if len(do_nothing_df) > 0:
        print(do_nothing_df["reward"].describe())
    else:
        print("No do_nothing actions found.")

    # Best action per scenario.
    best_idx = df.groupby("scenario_id")["reward"].idxmax()
    best = df.loc[best_idx].copy().sort_values("scenario_id")

    print("\nBest action per scenario:")
    best_cols = [
        "scenario_id",
        "action_id",
        "action_type",
        "branch_id",
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
    ]

    print(best[best_cols].to_string(index=False))

    print("\nScenarios where best action is do_nothing:")
    best_do_nothing = best[best["action_type"] == "do_nothing"]
    print(f"Count: {len(best_do_nothing)}")
    if len(best_do_nothing) > 0:
        print(best_do_nothing[["scenario_id", "reward", "before_max_loading"]].to_string(index=False))

    print("\nScenarios where best action is topology switching:")
    best_switch = best[best["action_type"] == "switch_off_branch"]
    print(f"Count: {len(best_switch)}")
    if len(best_switch) > 0:
        print(best_switch[best_cols].to_string(index=False))

    print("\nTop 20 transitions by reward:")
    print(
        df.sort_values("reward", ascending=False)[best_cols]
        .head(20)
        .to_string(index=False)
    )

    print("\nWorst 20 transitions by reward:")
    print(
        df.sort_values("reward", ascending=True)[best_cols]
        .head(20)
        .to_string(index=False)
    )

    # Improvement diagnostics.
    df["max_loading_delta"] = df["after_max_loading"] - df["before_max_loading"]
    df["total_overload_delta"] = df["after_total_overload"] - df["before_total_overload"]
    df["overloaded_count_delta"] = df["after_num_overloaded"] - df["before_num_overloaded"]
    df["hard_overloaded_count_delta"] = (
        df["after_num_hard_overloaded"] - df["before_num_hard_overloaded"]
    )

    print("\nPhysical improvement counts:")
    print(f"Reduced max loading:        {(df['max_loading_delta'] < 0).sum()}")
    print(f"Reduced total overload:     {(df['total_overload_delta'] < 0).sum()}")
    print(f"Reduced overloaded count:   {(df['overloaded_count_delta'] < 0).sum()}")
    print(f"Reduced hard overload count:{(df['hard_overloaded_count_delta'] < 0).sum()}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best.to_csv(output_dir / "best_actions_per_scenario.csv", index=False)
        df.sort_values("reward", ascending=False).to_csv(
            output_dir / "transitions_sorted_by_reward.csv",
            index=False,
        )

        print(f"\nSaved analysis files to: {output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
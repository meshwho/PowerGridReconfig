from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.gridfm_action_space import GridFMActionSpace
from grid_topology_ai.gridfm_adapter import GridFMAdapter
from grid_topology_ai.gridfm_pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.gridfm_reward import GridFMReward
from grid_topology_ai.gridfm_transition_generator import (
    GridFMTransitionGenerator,
    save_transitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GridFM transition dataset for topology switching."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/gridfm_transitions/transitions.csv",
        help="Output CSV file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help=(
            "Evaluate only top-K valid switch-off actions by current loading. "
            "Use -1 to evaluate all valid switch actions."
        ),
    )

    parser.add_argument(
        "--no-do-nothing",
        action="store_true",
        help="Do not include do_nothing transitions.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    top_k = None if args.top_k == -1 else args.top_k

    print("=" * 100)
    print("Generating GridFM transition dataset")
    print("=" * 100)

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Output:        {args.output}")
    print(f"Top-K actions: {top_k if top_k is not None else 'all'}")
    print(f"Do nothing:    {not args.no_do_nothing}")

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)

    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        min_loading_for_switch_percent=0.0,
    )

    reward_fn = GridFMReward()

    generator = GridFMTransitionGenerator(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
    )

    transitions = generator.generate_for_useful_scenarios(
        max_switch_actions_per_scenario=top_k,
        include_do_nothing=not args.no_do_nothing,
    )

    save_transitions(transitions, args.output)

    print("\nGenerated transition dataset:")
    print(f"Rows: {len(transitions)}")

    if len(transitions) > 0:
        print("\nReward statistics:")
        print(transitions["reward"].describe())

        print("\nPower flow success rate:")
        print(transitions["power_flow_success"].value_counts(dropna=False))

        print("\nDone count:")
        print(transitions["done"].value_counts(dropna=False))

        print("\nTop 10 actions by reward:")
        cols = [
            "scenario_id",
            "action_id",
            "action_type",
            "branch_id",
            "reward",
            "before_max_loading",
            "after_max_loading",
            "before_num_overloaded",
            "after_num_overloaded",
            "before_num_hard_overloaded",
            "after_num_hard_overloaded",
            "done",
        ]

        print(
            transitions.sort_values("reward", ascending=False)[cols]
            .head(10)
            .to_string(index=False)
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
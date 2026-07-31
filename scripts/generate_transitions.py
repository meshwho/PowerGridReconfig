from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.transition_generator import GridFMTransitionGenerator, save_transitions
from scripts.topology_action_cli import (
    action_space_kwargs,
    add_topology_action_arguments,
    print_topology_action_config,
    topology_action_config_from_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GridFM transition dataset for topology switching."
    )
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("--output", default="data/gridfm_transitions/transitions.csv")
    parser.add_argument("--min-start-loading", type=float, default=100.0)
    parser.add_argument("--min-start-total-overload", type=float, default=0.0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help=(
            "Evaluate the top-K loading-ranked branch-opening actions plus every "
            "legal branch-closing action. Use -1 to evaluate every legal topology action."
        ),
    )
    parser.add_argument("--no-do-nothing", action="store_true")
    add_topology_action_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    top_k = None if args.top_k == -1 else args.top_k
    topology_config = topology_action_config_from_args(args)

    print("=" * 100)
    print("Generating GridFM transition dataset")
    print("=" * 100)
    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Output:        {args.output}")
    print(f"Top-K actions: {top_k if top_k is not None else 'all'}")
    print(f"Do nothing:    {not args.no_do_nothing}")
    print_topology_action_config(topology_config)

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)
    action_space = GridFMActionSpace(
        **action_space_kwargs(topology_config, enable_cache=True)
    )
    generator = GridFMTransitionGenerator(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=GridFMReward(),
    )
    transitions = generator.generate_for_useful_scenarios(
        max_switch_actions_per_scenario=top_k,
        include_do_nothing=not args.no_do_nothing,
        min_start_max_loading_percent=args.min_start_loading,
        min_start_total_overload=args.min_start_total_overload,
    )
    save_transitions(transitions, args.output)

    print(f"\nRows: {len(transitions)}")
    if len(transitions) > 0:
        print("\nAction types:")
        print(transitions["action_type"].value_counts(dropna=False))
        print("\nReward statistics:")
        print(transitions["reward"].describe())
        columns = [
            "scenario_id", "action_id", "action_type", "target_status",
            "branch_id", "reward", "before_max_loading", "after_max_loading", "done",
        ]
        print("\nTop 10 actions by reward:")
        print(
            transitions.sort_values("reward", ascending=False)[columns]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

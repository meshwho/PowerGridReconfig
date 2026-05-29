from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.beam_search import BeamSearchConfig, BeamSearchPlanner


def print_node(title: str, node) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(f"Sequence:              {node.short_sequence()}")
    print(f"Action IDs:            {node.action_ids}")
    print(f"Branch IDs:            {node.branch_ids}")
    print(f"Rewards:               {[round(x, 4) for x in node.rewards]}")
    print(f"Discounted return:     {node.discounted_return:.4f}")
    print(f"Undiscounted return:   {node.undiscounted_return:.4f}")
    print(f"Depth:                 {node.depth}")
    print(f"Done:                  {node.done}")
    print(f"Solved:                {node.solved}")
    print(f"Termination reason:    {node.termination_reason}")

    state = node.env.current_state

    if state is not None:
        print("\nFinal state metrics:")
        print(f"  max_loading_percent:          {state.metrics['max_loading_percent']:.4f}")
        print(f"  num_overloaded_branches:      {state.metrics['num_overloaded_branches']}")
        print(f"  num_hard_overloaded_branches: {state.metrics['num_hard_overloaded_branches']}")
        print(f"  num_outaged_branches:         {state.metrics['num_outaged_branches']}")
        print(f"  outaged_branch_ids:           {state.outaged_branch_ids}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run beam search for multi-step topology switching."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument(
        "--scenario",
        type=int,
        default=7,
        help="Scenario ID.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum search depth.",
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K switch actions by current loading at each state.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Running beam search")
    print("=" * 100)

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Scenario:      {args.scenario}")
    print(f"Depth:         {args.depth}")
    print(f"Beam width:    {args.beam_width}")
    print(f"Top-K actions: {args.top_k}")
    print(f"Gamma:         {args.gamma}")

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)
    action_space = GridFMActionSpace(require_connected_after_switch=True)
    reward_fn = GridFMReward()

    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=args.depth,
    )

    config = BeamSearchConfig(
        max_depth=args.depth,
        beam_width=args.beam_width,
        top_k_actions=args.top_k,
        gamma=args.gamma,
        include_stop_action=True,
    )

    planner = BeamSearchPlanner(config)

    result = planner.search(
        env=env,
        scenario_id=args.scenario,
    )

    print_node("Best sequence found", result.best_node)

    print("\n" + "=" * 100)
    print("Final beam")
    print("=" * 100)

    for i, node in enumerate(result.final_beam, start=1):
        print(
            f"{i:2d}. "
            f"seq={node.short_sequence():30s} | "
            f"R={node.discounted_return:10.4f} | "
            f"solved={str(node.solved):5s} | "
            f"done={str(node.done):5s} | "
            f"depth={node.depth}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.beam_search import BeamSearchConfig, BeamSearchPlanner
from scripts.topology_action_cli import (
    action_space_kwargs,
    add_topology_action_arguments,
    print_topology_action_config,
    topology_action_config_from_args,
)


def print_node(title: str, node) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"Sequence:            {node.short_sequence()}")
    print(f"Action IDs:          {node.action_ids}")
    print(f"Branch IDs:          {node.branch_ids}")
    print(f"Discounted return:   {node.discounted_return:.4f}")
    print(f"Undiscounted return: {node.undiscounted_return:.4f}")
    print(f"Depth:               {node.depth}")
    print(f"Done:                {node.done}")
    print(f"Solved:              {node.solved}")
    print(f"Termination reason:  {node.termination_reason}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run beam search for multi-step topology switching."
    )
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("--scenario", type=int, default=7)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.95)
    add_topology_action_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    topology_config = topology_action_config_from_args(args)

    print("=" * 100)
    print("Running beam search")
    print("=" * 100)
    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Scenario:      {args.scenario}")
    print(f"Depth:         {args.depth}")
    print(f"Beam width:    {args.beam_width}")
    print(f"Top-K actions: {args.top_k}")
    print(f"Gamma:         {args.gamma}")
    print_topology_action_config(topology_config)

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)
    action_space = GridFMActionSpace(
        **action_space_kwargs(topology_config, enable_cache=True)
    )
    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=GridFMReward(),
        max_steps=args.depth,
    )
    planner = BeamSearchPlanner(
        BeamSearchConfig(
            max_depth=args.depth,
            beam_width=args.beam_width,
            top_k_actions=args.top_k,
            gamma=args.gamma,
            include_stop_action=True,
        )
    )
    result = planner.search(env=env, scenario_id=args.scenario)
    print_node("Best sequence found", result.best_node)

    print("\nFinal beam")
    for index, node in enumerate(result.final_beam, start=1):
        print(
            f"{index:2d}. seq={node.short_sequence():30s} | "
            f"R={node.discounted_return:10.4f} | "
            f"solved={str(node.solved):5s} | "
            f"done={str(node.done):5s} | depth={node.depth}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner
from scripts.topology_action_cli import (
    action_space_kwargs,
    add_topology_action_arguments,
    print_topology_action_config,
    topology_action_config_from_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AlphaZero-style MCTS for topology switching."
    )
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("--scenario", type=int, default=7)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help=(
            "Initial number of switch actions exposed to PUCT at each node. "
            "Progressive widening may activate additional legal actions."
        ),
    )
    parser.add_argument("--widening-coefficient", type=float, default=2.0)
    parser.add_argument("--widening-exponent", type=float, default=0.5)
    parser.add_argument("--exploration-quota", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--prior-exponent", type=float, default=0.5)
    parser.add_argument(
        "--stop-policy",
        default="no_hard_overloads",
        choices=["never", "solved_only", "no_hard_overloads", "always"],
    )
    add_topology_action_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    topology_config = topology_action_config_from_args(args)

    print("=" * 100)
    print("Running AlphaZero-style MCTS")
    print("=" * 100)
    print(f"Raw directory:       {raw_dir.resolve()}")
    print(f"Scenario:            {args.scenario}")
    print(f"Simulations:         {args.simulations}")
    print(f"Depth:               {args.depth}")
    print(f"Initial width:       {args.top_k}")
    print(f"Widening coefficient:{args.widening_coefficient}")
    print(f"Widening exponent:   {args.widening_exponent}")
    print(f"Exploration quota:   {args.exploration_quota}")
    print(f"Random seed:         {args.seed}")
    print(f"Gamma:               {args.gamma}")
    print(f"C_PUCT:              {args.c_puct}")
    print_topology_action_config(topology_config)

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)
    action_space = GridFMActionSpace(
        **action_space_kwargs(topology_config, enable_cache=True)
    )
    reward_fn = GridFMReward(discount_factor=args.gamma)
    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=args.depth,
    )
    config = MCTSConfig(
        num_simulations=args.simulations,
        max_depth=args.depth,
        top_k_actions=args.top_k,
        widening_coefficient=args.widening_coefficient,
        widening_exponent=args.widening_exponent,
        exploration_quota=args.exploration_quota,
        random_seed=args.seed,
        gamma=args.gamma,
        c_puct=args.c_puct,
        include_stop_action=True,
        prior_exponent=args.prior_exponent,
        stop_policy=args.stop_policy,
    )
    result = MCTSPlanner(config).search(
        env=env,
        scenario_id=args.scenario,
    )

    print("\nRoot action coverage")
    print(f"Legal actions:       {result.root_legal_action_count}")
    print(f"Considered actions:  {result.root_considered_action_count}")
    print(f"Visited actions:     {result.root_visited_action_count}")
    print(f"Considered coverage: {result.root_action_coverage:.1%}")
    print(f"Visited coverage:    {result.root_visited_action_coverage:.1%}")

    rows = []
    for action_id, prior in result.root.action_priors.items():
        child = result.root.children.get(action_id)
        action = result.root.actions_by_id[action_id]
        rows.append(
            {
                "action_id": action_id,
                "action_type": action.action_type,
                "branch_id": action.branch_id,
                "prior": prior,
                "visits": 0 if child is None else child.visit_count,
                "policy": result.policy.get(action_id, 0.0),
                "q_value": 0.0 if child is None else child.mean_value,
            }
        )

    print("\nRoot policy from MCTS visit counts")
    for row in sorted(
        rows,
        key=lambda item: (item["visits"], item["q_value"]),
        reverse=True,
    )[:20]:
        print(row)

    print("\nBest action by visit count")
    print(f"Best action ID: {result.best_action_id}")
    print(f"Best branch ID: {result.best_branch_id}")
    print(f"Principal action IDs: {result.principal_action_ids}")
    print(f"Principal branch IDs: {result.principal_branch_ids}")
    print(f"Discounted return: {result.principal_return:.4f}")


if __name__ == "__main__":
    main()

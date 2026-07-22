from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlphaZero-style MCTS for topology switching."
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
        "--simulations",
        type=int,
        default=100,
        help="Number of MCTS simulations.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum search depth.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K switch actions considered per node.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor.",
    )

    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration constant.",
    )

    parser.add_argument(
        "--prior-exponent",
        type=float,
        default=0.5,
        help="Exponent for smoothing heuristic action priors.",
    )

    parser.add_argument(
        "--stop-policy",
        type=str,
        default="no_hard_overloads",
        choices=["never", "solved_only", "no_hard_overloads", "always"],
        help="When MCTS is allowed to use the stop/handoff action.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Running AlphaZero-style MCTS")
    print("=" * 100)

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Scenario:      {args.scenario}")
    print(f"Simulations:   {args.simulations}")
    print(f"Depth:         {args.depth}")
    print(f"Top-K actions: {args.top_k}")
    print(f"Gamma:         {args.gamma}")
    print(f"C_PUCT:        {args.c_puct}")

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

    config = MCTSConfig(
        num_simulations=args.simulations,
        max_depth=args.depth,
        top_k_actions=args.top_k,
        gamma=args.gamma,
        c_puct=args.c_puct,
        include_stop_action=True,
        prior_exponent=args.prior_exponent,
        stop_policy=args.stop_policy,
    )

    planner = MCTSPlanner(config)

    result = planner.search(
        env=env,
        scenario_id=args.scenario,
    )

    print("\n" + "=" * 100)
    print("Root policy from MCTS visit counts")
    print("=" * 100)

    root = result.root

    rows = []

    for action_id, prior in root.action_priors.items():
        child = root.children.get(action_id)
        action = root.actions_by_id[action_id]

        visits = 0 if child is None else child.visit_count
        q_value = 0.0 if child is None else child.mean_value
        reward = 0.0 if child is None else child.reward_from_parent
        policy_prob = result.policy.get(action_id, 0.0)

        rows.append(
            {
                "action_id": action_id,
                "action_type": action.action_type,
                "branch_id": action.branch_id,
                "prior": prior,
                "visits": visits,
                "policy": policy_prob,
                "q_value": q_value,
                "first_reward": reward,
            }
        )

    rows = sorted(
        rows,
        key=lambda row: (row["visits"], row["q_value"]),
        reverse=True,
    )

    print(
        " action_id | type              | branch | prior   | visits | policy  | q_value   | first_reward"
    )
    print("-" * 100)

    for row in rows[:20]:
        branch = "-" if row["branch_id"] is None else str(row["branch_id"])

        print(
            f"{row['action_id']:>9} | "
            f"{row['action_type']:<17} | "
            f"{branch:>6} | "
            f"{row['prior']:.4f} | "
            f"{row['visits']:>6} | "
            f"{row['policy']:.4f} | "
            f"{row['q_value']:>9.4f} | "
            f"{row['first_reward']:>12.4f}"
        )

    print("\n" + "=" * 100)
    print("Best action by visit count")
    print("=" * 100)

    print(f"Best action ID:       {result.best_action_id}")
    print(f"Best branch ID:       {result.best_branch_id}")

    print("\n" + "=" * 100)
    print("Principal variation")
    print("=" * 100)

    print(f"Action IDs:           {result.principal_action_ids}")
    print(f"Branch IDs:           {result.principal_branch_ids}")
    print(f"Rewards:              {[round(x, 4) for x in result.principal_rewards]}")
    print(f"Discounted return:    {result.principal_return:.4f}")
    print(f"Prior exponent:{args.prior_exponent}")

    if result.principal_final_metrics:
        print("\nFinal metrics:")
        print(
            f"  max_loading_percent:          "
            f"{result.principal_final_metrics['max_loading_percent']:.4f}"
        )
        print(
            f"  num_overloaded_branches:      "
            f"{result.principal_final_metrics['num_overloaded_branches']}"
        )
        print(
            f"  num_hard_overloaded_branches: "
            f"{result.principal_final_metrics['num_hard_overloaded_branches']}"
        )
        print(
            f"  num_outaged_branches:         "
            f"{result.principal_final_metrics['num_outaged_branches']}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
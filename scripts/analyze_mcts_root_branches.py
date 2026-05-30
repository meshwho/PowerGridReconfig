from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.continuation_gate import analyze_root_branches
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MCTS root branches using lookahead continuation gate."
    )

    parser.add_argument("raw_dir", type=str)
    parser.add_argument("--scenario", type=int, default=7)

    parser.add_argument(
        "--prefix-branches",
        type=int,
        nargs="*",
        default=[],
        help="Branches to switch off before running root MCTS analysis.",
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--prior-exponent", type=float, default=0.5)
    parser.add_argument("--pf-alg", type=int, default=3)

    parser.add_argument(
        "--stop-policy",
        type=str,
        default="no_hard_overloads",
        choices=["never", "solved_only", "no_hard_overloads", "always"],
    )

    parser.add_argument(
        "--min-hard-improvement",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--min-soft-improvement",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of root branches to print.",
    )

    parser.add_argument("--min-visits", type=int, default=5)
    parser.add_argument("--min-visit-fraction", type=float, default=0.01)

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    adapter = GridFMAdapter(raw_dir)

    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=args.pf_alg,
        enable_cache=True,
    )

    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=True,
    )

    reward_fn = GridFMReward()

    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=args.max_steps,
    )

    env.reset(args.scenario)

    print("=" * 100)
    print("Analyzing MCTS root branches")
    print("=" * 100)
    print(f"Scenario:        {args.scenario}")
    print(f"Prefix branches: {args.prefix_branches}")
    print(f"Simulations:     {args.simulations}")
    print(f"Depth:           {args.depth}")
    print(f"Top-K:           {args.top_k}")
    print(f"PF alg:          {args.pf_alg}")

    for branch_id in args.prefix_branches:
        action = env.action_by_branch_id(int(branch_id))
        step_result = env.step(action)

        print(
            f"Prefix step: branch={branch_id} | "
            f"reward={step_result.reward:.4f} | "
            f"done={step_result.done} | "
            f"reason={env.termination_reason}"
        )


        if step_result.done:
            break

    if env.current_state is None:
        raise RuntimeError("No current state.")

    print("\nCurrent state:")
    print(
        f"  max_loading={env.current_state.metrics['max_loading_percent']:.4f}% | "
        f"overloaded={env.current_state.metrics['num_overloaded_branches']} | "
        f"hard={env.current_state.metrics['num_hard_overloaded_branches']} | "
        f"outaged={env.current_state.outaged_branch_ids}"
    )

    evaluator = NeuralPolicyValueEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        enable_cache=True,
    )

    config = MCTSConfig(
        num_simulations=args.simulations,
        max_depth=args.depth,
        top_k_actions=args.top_k,
        gamma=args.gamma,
        c_puct=args.c_puct,
        prior_exponent=args.prior_exponent,
        stop_policy=args.stop_policy,
        include_stop_action=True,
    )

    planner = MCTSPlanner(
        config=config,
        evaluator=evaluator,
    )

    result = planner.search_from_env(env)

    decision = analyze_root_branches(
        result=result,
        min_hard_improvement=args.min_hard_improvement,
        min_soft_improvement=args.min_soft_improvement,
        min_visits=args.min_visits,
        min_visit_fraction=args.min_visit_fraction,
    )

    print("\n" + "=" * 100)
    print("Gate decision")
    print("=" * 100)

    print(f"Root penalty:              {decision.root_penalty:.4f}")
    print(f"Root has hard overload:    {decision.root_has_hard_overload}")
    print(f"Best by visits:            action={decision.best_visit_action_id}, branch={decision.best_visit_branch_id}")
    print(f"Best by improvement:       action={decision.best_improvement_action_id}, branch={decision.best_improvement_branch_id}")
    print(f"Best improvement:          {decision.best_improvement:.4f}")
    print(f"Selected action:           {decision.selected_action_id}")
    print(f"Selected branch:           {decision.selected_branch_id}")
    print(f"Selected reason:           {decision.selected_reason}")
    print("\n" + "=" * 100)
    print("Root branch analysis")
    print("=" * 100)

    header = (
        " rank | action | branch | visits | policy | imm_reward | "
        "improvement | allow | best_sequence | final_max | final_over | final_hard"
    )
    print(header)
    print("-" * len(header))

    for rank, branch in enumerate(decision.branches[: args.show], start=1):
        final_max = branch.best_final_metrics.get("max_loading_percent", float("nan"))
        final_over = branch.best_final_metrics.get("num_overloaded_branches", -1)
        final_hard = branch.best_final_metrics.get("num_hard_overloaded_branches", -1)

        sequence = " -> ".join(
            "-" if item is None else str(item)
            for item in branch.best_sequence_branch_ids
        )

        print(
            f"{rank:>5} | "
            f"{branch.action_id:>6} | "
            f"{str(branch.branch_id):>6} | "
            f"{branch.visits:>6} | "
            f"{branch.policy:>6.3f} | "
            f"{branch.immediate_reward:>10.3f} | "
            f"{branch.improvement:>11.3f} | "
            f"{str(branch.allow):>5} | "
            f"{sequence:<30} | "
            f"{float(final_max):>9.3f} | "
            f"{int(final_over):>10} | "
            f"{int(final_hard):>10}"
        )
        print(f"      confidence_ok: {branch.confidence_ok}")
        print(f"      reason:        {branch.reason}")

    print("\nCaches:")
    print("  Power flow:", backend.cache_info())
    print("  Action space:", action_space.cache_info())
    print("  Evaluator:", evaluator.cache_info())

    print("\nDone.")


if __name__ == "__main__":
    main()
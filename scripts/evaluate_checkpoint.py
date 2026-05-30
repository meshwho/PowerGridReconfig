from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner


def compute_safety_score(row: dict) -> float:
    """
    Operational checkpoint score.

    Higher is better.

    Priority:
        1. solved is best;
        2. handoff_to_redispatch is acceptable;
        3. max_steps_reached is bad;
        4. remaining hard overload is heavily penalized;
        5. remaining overload and high final loading are also penalized.
    """

    score = 0.0

    reason = row.get("termination_reason")
    solved = bool(row.get("solved", False))

    final_loading = float(row.get("final_max_loading_percent", 999.0))
    overloaded = int(row.get("final_num_overloaded_branches", 99))
    hard = int(row.get("final_num_hard_overloaded_branches", 99))
    discounted_return = float(row.get("discounted_return", 0.0))

    if solved:
        score += 1000.0
    elif reason == "handoff_to_redispatch":
        score += 500.0
    elif reason == "max_steps_reached":
        score -= 300.0
    elif reason == "power_flow_failed":
        score -= 1000.0
    else:
        score -= 100.0

    score -= 300.0 * hard
    score -= 50.0 * overloaded

    if final_loading > 100.0:
        score -= 5.0 * (final_loading - 100.0)

    # Small contribution from reward, but do not let reward dominate safety.
    score += 0.05 * discounted_return

    return float(score)

def run_episode(
    scenario_id: int,
    adapter: GridFMAdapter,
    backend: GridFMPowerFlowBackend,
    action_space: GridFMActionSpace,
    reward_fn: GridFMReward,
    planner: MCTSPlanner,
    max_steps: int,
    gamma: float,
) -> dict:
    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=max_steps,
    )

    env.reset(scenario_id)

    total_reward = 0.0
    discounted_return = 0.0
    discount = 1.0

    actions = []
    branches = []
    rewards = []

    for _ in range(max_steps):
        if env.done:
            break

        result = planner.search_from_env(env)

        if result.best_action_id is None:
            break

        action_id = int(result.best_action_id)
        branch_id = result.best_branch_id

        step_result = env.step(action_id)

        reward = float(step_result.reward)

        actions.append(action_id)
        branches.append(branch_id)
        rewards.append(reward)

        total_reward += reward
        discounted_return += discount * reward
        discount *= gamma

        if step_result.done:
            break

    final_state = env.current_state

    if final_state is None:
        final_max_loading = float("nan")
        final_overloaded = -1
        final_hard = -1
        final_outaged = -1
    else:
        final_max_loading = float(final_state.metrics["max_loading_percent"])
        final_overloaded = int(final_state.metrics["num_overloaded_branches"])
        final_hard = int(final_state.metrics["num_hard_overloaded_branches"])
        final_outaged = int(final_state.metrics["num_outaged_branches"])

    row = {
        "scenario_id": int(scenario_id),
        "steps": len(actions),
        "actions": str(actions),
        "branches": str(branches),
        "rewards": str([round(x, 4) for x in rewards]),
        "total_reward": float(total_reward),
        "discounted_return": float(discounted_return),
        "done": bool(env.done),
        "solved": bool(env.solved),
        "termination_reason": env.termination_reason,
        "final_max_loading_percent": final_max_loading,
        "final_num_overloaded_branches": final_overloaded,
        "final_num_hard_overloaded_branches": final_hard,
        "final_num_outaged_branches": final_outaged,
    }

    row["safety_score"] = compute_safety_score(row)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a neural policy-value checkpoint with deterministic MCTS."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument(
        "--transitions",
        type=str,
        required=True,
        help="Transitions CSV used only to select scenario IDs.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Policy-value checkpoint to evaluate.",
    )

    parser.add_argument("--simulations", type=int, default=150)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--prior-exponent", type=float, default=0.5)

    parser.add_argument(
        "--stop-policy",
        type=str,
        default="no_hard_overloads",
        choices=["never", "solved_only", "no_hard_overloads", "always"],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    transitions_path = Path(args.transitions)
    checkpoint_path = Path(args.checkpoint)

    print("=" * 100)
    print("Evaluating checkpoint")
    print("=" * 100)

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Transitions:   {transitions_path.resolve()}")
    print(f"Checkpoint:    {checkpoint_path.resolve()}")

    transitions = pd.read_csv(transitions_path)
    scenario_ids = sorted(int(x) for x in transitions["scenario_id"].unique())

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)
    action_space = GridFMActionSpace(require_connected_after_switch=True)
    reward_fn = GridFMReward()

    evaluator = NeuralPolicyValueEvaluator(
        checkpoint_path=checkpoint_path,
        device=args.device,
    )

    config = MCTSConfig(
        num_simulations=args.simulations,
        max_depth=args.depth,
        top_k_actions=args.top_k,
        gamma=args.gamma,
        c_puct=args.c_puct,
        leaf_penalty_weight=0.10,
        include_stop_action=True,
        prior_exponent=args.prior_exponent,
        stop_policy=args.stop_policy,

        # Evaluation must be deterministic.
        use_root_dirichlet_noise=False,
    )

    planner = MCTSPlanner(
        config=config,
        evaluator=evaluator,
    )

    rows = []

    for scenario_id in scenario_ids:
        row = run_episode(
            scenario_id=scenario_id,
            adapter=adapter,
            backend=backend,
            action_space=action_space,
            reward_fn=reward_fn,
            planner=planner,
            max_steps=args.max_steps,
            gamma=args.gamma,
        )

        rows.append(row)

        print(
            f"Scenario {scenario_id:>3} | "
            f"reason={row['termination_reason']} | "
            f"solved={row['solved']} | "
            f"steps={row['steps']} | "
            f"branches={row['branches']} | "
            f"final_loading={row['final_max_loading_percent']:.2f}% | "
            f"overloaded={row['final_num_overloaded_branches']} | "
            f"hard={row['final_num_hard_overloaded_branches']} | "
            f"R={row['discounted_return']:.2f} | "
            f"score={row['safety_score']:.2f}"
        )
    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    print("\nTermination reasons:")
    print(df["termination_reason"].value_counts(dropna=False).to_string())

    print("\nSolved:")
    print(df["solved"].value_counts(dropna=False).to_string())

    print("\nAverage metrics:")
    print(f"  Avg discounted return: {df['discounted_return'].mean():.4f}")
    print(f"  Avg final loading:     {df['final_max_loading_percent'].mean():.4f}%")
    print(f"  Avg overloaded:        {df['final_num_overloaded_branches'].mean():.4f}")
    print(f"  Avg hard overloaded:   {df['final_num_hard_overloaded_branches'].mean():.4f}")
    print(f"  Avg safety score:     {df['safety_score'].mean():.4f}")
    print(f"  Total safety score:   {df['safety_score'].sum():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
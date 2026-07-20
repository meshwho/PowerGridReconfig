from __future__ import annotations

from dataclasses import replace

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner

from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
import time
from grid_topology_ai.search.continuation_gate import (
    analyze_root_branches,
    make_do_nothing_action,
)

def print_state_metrics(prefix: str, env: TopologySwitchingEnv) -> None:
    state = env.current_state

    if state is None:
        print(f"{prefix}: no state")
        return

    print(
        f"{prefix}: "
        f"max_loading={state.metrics['max_loading_percent']:.4f}% | "
        f"overloaded={state.metrics['num_overloaded_branches']} | "
        f"hard={state.metrics['num_hard_overloaded_branches']} | "
        f"outaged={state.outaged_branch_ids}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one MCTS-controlled topology switching episode."
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
        default=300,
        help="Number of MCTS simulations per decision.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="MCTS search depth per decision.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum real episode steps.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-K switch actions considered per MCTS node.",
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
        default=2.0,
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

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional neural policy-value checkpoint for neural-guided MCTS.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for neural evaluator: cpu or cuda.",
    )

    parser.add_argument(
        "--pf-alg",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="PYPOWER power flow algorithm: 1=NR, 2=FDXB, 3=FDBX, 4=GS.",
    )

    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable power flow cache.",
    )

    parser.add_argument(
        "--use-continuation-gate",
        action="store_true",
        help="Use lookahead continuation gate instead of raw max-visit MCTS action.",
    )

    parser.add_argument("--min-hard-improvement", type=float, default=50.0)
    parser.add_argument("--min-soft-improvement", type=float, default=15.0)
    parser.add_argument("--min-gate-visits", type=int, default=5)
    parser.add_argument("--min-gate-visit-fraction", type=float, default=0.01)

    parser.add_argument(
        "--allow-handoff-with-hard-overloads",
        action="store_true",
        help=(
            "Treat action 0 as redispatch handoff even when hard overloads remain."
        ),
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Running MCTS-controlled episode")
    print("=" * 100)

    print(f"Raw directory:  {raw_dir.resolve()}")
    print(f"Scenario:       {args.scenario}")
    print(f"Simulations:    {args.simulations}")
    print(f"Search depth:   {args.depth}")
    print(f"Max steps:      {args.max_steps}")
    print(f"Top-K actions:  {args.top_k}")
    print(f"Gamma:          {args.gamma}")
    print(f"C_PUCT:         {args.c_puct}")
    print(f"Prior exponent: {args.prior_exponent}")
    print(f"Stop policy:    {args.stop_policy}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Device:         {args.device}")
    print(f"PF algorithm:   {args.pf_alg}")
    print(f"Cache enabled:  {not args.disable_cache}")

    print(f"Continuation gate: {args.use_continuation_gate}")
    print(f"Allow hard handoff: {args.allow_handoff_with_hard_overloads}")

    if args.use_continuation_gate:
        print(f"  min hard improvement: {args.min_hard_improvement}")
        print(f"  min soft improvement: {args.min_soft_improvement}")
        print(f"  min gate visits:      {args.min_gate_visits}")
        print(f"  min gate visit frac:  {args.min_gate_visit_fraction}")

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=replace(DEFAULT_PHYSICS_CONFIG, pf_alg=args.pf_alg),
        enable_cache=not args.disable_cache,
    )
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=not args.disable_cache,
    )
    reward_fn = GridFMReward()

    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=args.max_steps,
        allow_handoff_with_hard_overloads=args.allow_handoff_with_hard_overloads,
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
    )

    evaluator = None

    if args.checkpoint is not None:
        evaluator = NeuralPolicyValueEvaluator(
            checkpoint_path=args.checkpoint,
            device=args.device,
            enable_cache=not args.disable_cache,
        )

        print("\nNeural evaluator loaded.")

    planner = MCTSPlanner(
        config=config,
        evaluator=evaluator,
    )

    env.reset(args.scenario)
    episode_start_time = time.perf_counter()

    print()
    print_state_metrics("Initial state", env)

    episode_reward = 0.0
    discounted_episode_return = 0.0
    discount = 1.0

    step_records = []

    for step in range(args.max_steps):
        if env.done:
            break

        print("\n" + "-" * 100)
        print(f"Decision step {step + 1}")
        print("-" * 100)

        search_result = planner.search_from_env(env)

        if search_result.best_action_id is None:
            print("MCTS did not return an action.")
            break

        raw_best_action_id = int(search_result.best_action_id)
        raw_best_branch_id = search_result.best_branch_id

        gate_decision = None

        if args.use_continuation_gate:
            gate_decision = analyze_root_branches(
                result=search_result,
                min_hard_improvement=args.min_hard_improvement,
                min_soft_improvement=args.min_soft_improvement,
                min_visits=args.min_gate_visits,
                min_visit_fraction=args.min_gate_visit_fraction,
            )

            best_action_id = int(gate_decision.selected_action_id)
            best_branch_id = gate_decision.selected_branch_id
        else:
            best_action_id = raw_best_action_id
            best_branch_id = raw_best_branch_id

        print(f"MCTS raw best action ID: {raw_best_action_id}")
        print(f"MCTS raw best branch ID: {raw_best_branch_id}")

        if gate_decision is not None:
            print("\nContinuation gate:")
            print(f"  root_penalty:        {gate_decision.root_penalty:.4f}")
            print(f"  root_hard:           {gate_decision.root_num_hard}")
            print(
                f"  best_by_visits:      "
                f"action={gate_decision.best_visit_action_id}, "
                f"branch={gate_decision.best_visit_branch_id}"
            )
            print(
                f"  best_by_improvement: "
                f"action={gate_decision.best_improvement_action_id}, "
                f"branch={gate_decision.best_improvement_branch_id}, "
                f"improvement={gate_decision.best_improvement:.4f}"
            )
            print(
                f"  selected:            "
                f"action={gate_decision.selected_action_id}, "
                f"branch={gate_decision.selected_branch_id}"
            )
            print(f"  reason:              {gate_decision.selected_reason}")

        print(f"MCTS/gate selected action ID: {best_action_id}")
        print(f"MCTS/gate selected branch ID: {best_branch_id}")

        print("Root policy top actions:")

        root = search_result.root
        rows = []

        for action_id, probability in search_result.policy.items():
            child = root.children.get(action_id)
            action = root.actions_by_id[action_id]

            rows.append(
                {
                    "action_id": action_id,
                    "branch_id": action.branch_id,
                    "probability": probability,
                    "visits": 0 if child is None else child.visit_count,
                    "q_value": 0.0 if child is None else child.mean_value,
                    "first_reward": 0.0 if child is None else child.reward_from_parent,
                }
            )

        rows = sorted(rows, key=lambda row: row["probability"], reverse=True)

        for row in rows[:8]:
            branch = "-" if row["branch_id"] is None else row["branch_id"]

            print(
                f"  action={row['action_id']:>3} | "
                f"branch={branch!s:>4} | "
                f"pi={row['probability']:.4f} | "
                f"visits={row['visits']:>4} | "
                f"q={row['q_value']:.4f} | "
                f"r1={row['first_reward']:.4f}"
            )

        if best_action_id == 0:
            action_to_execute = make_do_nothing_action()
        else:
            action_to_execute = search_result.root.actions_by_id.get(best_action_id)

            if action_to_execute is None:
                action_to_execute = env.action_by_id(best_action_id)

        step_result = env.step(action_to_execute)

        episode_reward += float(step_result.reward)
        discounted_episode_return += discount * float(step_result.reward)
        discount *= args.gamma

        step_records.append(
            {
                "step": step + 1,
                "action_id": best_action_id,
                "branch_id": best_branch_id,
                "reward": float(step_result.reward),
                "done": bool(step_result.done),
                "solved": bool(step_result.solved),
                "power_flow_success": bool(step_result.power_flow_success),
                "termination_reason": step_result.info["termination_reason"],
            }
        )

        print("\nExecuted action:")
        print(f"  action_id:          {best_action_id}")
        print(f"  branch_id:          {best_branch_id}")
        print(f"  reward:             {step_result.reward:.4f}")
        print(f"  done:               {step_result.done}")
        print(f"  solved:             {step_result.solved}")
        print(f"  power_flow_success: {step_result.power_flow_success}")
        print(f"  termination_reason: {step_result.info['termination_reason']}")

        print_state_metrics("State after action", env)

        if step_result.done:
            break

    print("\n" + "=" * 100)
    print("Episode summary")
    print("=" * 100)

    for record in step_records:
        print(
            f"Step {record['step']:>2}: "
            f"action={record['action_id']:>3} | "
            f"branch={str(record['branch_id']):>4} | "
            f"reward={record['reward']:>10.4f} | "
            f"done={record['done']} | "
            f"solved={record['solved']} | "
            f"reason={record['termination_reason']}"
        )

    print()
    print(f"Total reward:              {episode_reward:.4f}")
    print(f"Discounted episode return: {discounted_episode_return:.4f}")
    print(f"Final done:                {env.done}")
    print(f"Final solved:              {env.solved}")
    print(f"Termination reason:        {env.termination_reason}")

    print_state_metrics("Final state", env)

    print("\nPower flow cache:")
    print(backend.cache_info())
    if evaluator is not None:
        print("\nNeural evaluator cache:")
        print(evaluator.cache_info())
    print("\nAction space cache:")
    print(action_space.cache_info())

    episode_elapsed = time.perf_counter() - episode_start_time

    print("\nTiming:")
    print(f"Episode elapsed time: {episode_elapsed:.4f} sec")

    print("\nDone.")


if __name__ == "__main__":
    main()
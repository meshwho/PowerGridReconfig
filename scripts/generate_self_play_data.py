from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner
from grid_topology_ai.self_play.replay_buffer import SelfPlayReplayBuffer

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    GridFMAdapter,
    GridFMState,
)
import time
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator



def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """
    Compute discounted return from each step.

    Example:
        rewards = [r0, r1, r2]

        returns[0] = r0 + gamma*r1 + gamma^2*r2
        returns[1] = r1 + gamma*r2
        returns[2] = r2
    """

    returns = [0.0 for _ in rewards]
    running = 0.0

    for i in reversed(range(len(rewards))):
        running = float(rewards[i]) + gamma * running
        returns[i] = running

    return returns

def select_action_from_policy(
    policy: dict[int, float],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """
    Select self-play action from MCTS visit-count policy.

    temperature = 0:
        deterministic argmax.

    temperature = 1:
        sample according to policy.

    0 < temperature < 1:
        sharper distribution.
    """

    if not policy:
        raise ValueError("Cannot select action from empty policy.")

    action_ids = np.array(list(policy.keys()), dtype=np.int64)
    probabilities = np.array(
        [float(policy[int(action_id)]) for action_id in action_ids],
        dtype=np.float64,
    )

    total = float(probabilities.sum())

    if total <= 0:
        probabilities = np.ones_like(probabilities) / len(probabilities)
    else:
        probabilities = probabilities / total

    if temperature <= 1e-8:
        return int(action_ids[int(np.argmax(probabilities))])

    adjusted = probabilities ** (1.0 / float(temperature))
    adjusted_sum = float(adjusted.sum())

    if adjusted_sum <= 0:
        adjusted = np.ones_like(adjusted) / len(adjusted)
    else:
        adjusted = adjusted / adjusted_sum

    return int(rng.choice(action_ids, p=adjusted))

def state_security_penalty(state: GridFMState) -> float:
    """
    Compute security penalty for the final state.

    This is used only for value target generation.
    Lower penalty means better final state.
    """

    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
    status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")

    loading = state.branch_features[:, loading_idx]
    status = state.branch_features[:, status_idx]

    active_loading = loading[status > 0]

    total_overload = float(np.sum(np.maximum(active_loading - 100.0, 0.0)))
    hard_overload = float(np.sum(np.maximum(active_loading - 120.0, 0.0)))

    num_overloaded = int(state.metrics["num_overloaded_branches"])
    num_hard_overloaded = int(state.metrics["num_hard_overloaded_branches"])

    voltage_penalty = float(state.metrics.get("total_voltage_violation", 0.0))

    penalty = (
        2.0 * total_overload
        + 5.0 * hard_overload
        + 10.0 * num_overloaded
        + 30.0 * num_hard_overloaded
        + 500.0 * voltage_penalty
    )

    return float(penalty)


def terminal_outcome_reward(
    state: GridFMState | None,
    solved: bool,
    termination_reason: str | None,
    terminal_unsolved_penalty: float,
    terminal_handoff_penalty: float,
    terminal_failure_penalty: float,
    terminal_penalty_weight: float,
) -> float:
    """
    Final episode outcome used for AlphaZero-like value targets.

    solved:
        successful topology switching.

    handoff_to_redispatch:
        topology switching helped, but did not fully solve the problem.
        This should be penalized, but not as hard as max_steps failure.

    max_steps_reached:
        topology switching failed to find a proper stopping point.

    power_flow_failed:
        severe failure.
    """

    if solved:
        return 0.0

    if state is None:
        return -float(terminal_failure_penalty)

    penalty = state_security_penalty(state)

    if termination_reason == "handoff_to_redispatch":
        return -float(terminal_handoff_penalty) - (
            float(terminal_penalty_weight) * penalty
        )

    if termination_reason == "power_flow_failed":
        return -float(terminal_failure_penalty) - (
            float(terminal_penalty_weight) * penalty
        )

    return -float(terminal_unsolved_penalty) - (
        float(terminal_penalty_weight) * penalty
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AlphaZero-like self-play data using MCTS."
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
        "--output-dir",
        type=str,
        default="data/self_play/mcts_v0",
        help="Output directory for self-play examples.",
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
        help="MCTS depth per decision.",
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
        help="Top-K actions considered by MCTS.",
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
        help="Exponent for heuristic prior smoothing.",
    )

    parser.add_argument(
        "--terminal-unsolved-penalty",
        type=float,
        default=500.0,
        help="Terminal penalty added when an episode ends without solving the grid.",
    )

    parser.add_argument(
        "--terminal-penalty-weight",
        type=float,
        default=0.10,
        help="Additional penalty weight for remaining final-state violations.",
    )

    parser.add_argument(
        "--stop-policy",
        type=str,
        default="no_hard_overloads",
        choices=["never", "solved_only", "no_hard_overloads", "always"],
        help="When MCTS is allowed to use the stop/handoff action.",
    )

    parser.add_argument(
        "--terminal-handoff-penalty",
        type=float,
        default=150.0,
        help="Terminal penalty for handoff_to_redispatch episodes.",
    )

    parser.add_argument(
        "--terminal-failure-penalty",
        type=float,
        default=1000.0,
        help="Terminal penalty for power flow failure.",
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
        "--use-root-noise",
        action="store_true",
        help="Use AlphaZero-style Dirichlet noise at MCTS root during self-play.",
    )

    parser.add_argument(
        "--root-dirichlet-alpha",
        type=float,
        default=0.30,
        help="Dirichlet alpha for root exploration noise.",
    )

    parser.add_argument(
        "--root-exploration-fraction",
        type=float,
        default=0.25,
        help="Fraction of root prior replaced by Dirichlet noise.",
    )

    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=0.0,
        help=(
            "Temperature for selecting actions from MCTS policy. "
            "0.0 = deterministic argmax, >0 = sampling for exploration."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for self-play action sampling and root noise.",
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
        help="Disable power flow/action/evaluator caches.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    transitions_path = Path(args.transitions)

    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")

    print("=" * 100)
    print("Generating AlphaZero-like self-play data")
    print("=" * 100)

    print(f"Raw directory:  {raw_dir.resolve()}")
    print(f"Transitions:    {transitions_path.resolve()}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Simulations:    {args.simulations}")
    print(f"Search depth:   {args.depth}")
    print(f"Max steps:      {args.max_steps}")
    print(f"Top-K actions:  {args.top_k}")
    print(f"Gamma:          {args.gamma}")
    print(f"C_PUCT:         {args.c_puct}")
    print(f"Prior exponent: {args.prior_exponent}")
    print(f"Terminal unsolved penalty: {args.terminal_unsolved_penalty}")
    print(f"Terminal penalty weight:   {args.terminal_penalty_weight}")
    print(f"Terminal handoff penalty:  {args.terminal_handoff_penalty}")
    print(f"Terminal failure penalty:  {args.terminal_failure_penalty}")
    print(f"Stop policy:               {args.stop_policy}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Device:         {args.device}")
    print(f"Use root noise: {args.use_root_noise}")
    print(f"Root alpha:     {args.root_dirichlet_alpha}")
    print(f"Root epsilon:   {args.root_exploration_fraction}")
    print(f"Temperature:    {args.selection_temperature}")
    print(f"Seed:           {args.seed}")
    print(f"PF algorithm:   {args.pf_alg}")
    print(f"Cache enabled:  {not args.disable_cache}")
    if args.selection_temperature <= 1e-8:
        print("Action selection: deterministic argmax")
    else:
        print("Action selection: sampling from MCTS policy")

    transitions = pd.read_csv(transitions_path)
    scenario_ids = sorted(int(x) for x in transitions["scenario_id"].unique())

    print(f"\nScenario IDs: {scenario_ids}")

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=args.pf_alg,
        enable_cache=not args.disable_cache,
    )
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=not args.disable_cache,
    )
    reward_fn = GridFMReward()

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
        use_root_dirichlet_noise=args.use_root_noise,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        random_seed=args.seed,
    )

    evaluator = None

    if args.checkpoint is not None:
        evaluator = NeuralPolicyValueEvaluator(
            checkpoint_path=args.checkpoint,
            device=args.device,
            enable_cache=not args.disable_cache,
        )
        print("\nNeural evaluator loaded.")

    rng = np.random.default_rng(args.seed)

    planner = MCTSPlanner(
        config=config,
        evaluator=evaluator,
    )

    replay_buffer = SelfPlayReplayBuffer(args.output_dir)

    total_examples = 0
    start_time = time.perf_counter()
    for scenario_id in scenario_ids:
        print("\n" + "=" * 100)
        print(f"Scenario {scenario_id}")
        print("=" * 100)

        env = TopologySwitchingEnv(
            adapter=adapter,
            backend=backend,
            action_space=action_space,
            reward_fn=reward_fn,
            max_steps=args.max_steps,
        )

        env.reset(scenario_id)

        pending_examples = []
        rewards = []

        for step in range(args.max_steps):
            if env.done:
                break

            state_before = env.current_state

            if state_before is None:
                break

            action_mask = env.valid_action_mask()

            search_result = planner.search_from_env(env)

            if search_result.best_action_id is None:
                print("MCTS returned no action. Stop episode.")
                break

            selected_action_id = select_action_from_policy(
                policy=search_result.policy,
                temperature=args.selection_temperature,
                rng=rng,
            )

            selected_action = search_result.root.actions_by_id[selected_action_id]
            selected_branch_id = selected_action.branch_id

            step_result = env.step(selected_action_id)

            rewards.append(float(step_result.reward))

            state_id = f"scenario_{scenario_id:06d}_step_{step:03d}"

            pending_examples.append(
                {
                    "state": state_before,
                    "state_id": state_id,
                    "action_mask": action_mask,
                    "scenario_id": scenario_id,
                    "step": step,
                    "selected_action_id": selected_action_id,
                    "selected_branch_id": selected_branch_id,
                    "step_reward": float(step_result.reward),
                    "visit_counts": search_result.visit_counts,
                    "mcts_policy": search_result.policy,
                    "done": bool(step_result.done),
                    "solved": bool(step_result.solved),
                    "termination_reason": step_result.info["termination_reason"],
                }
            )

            print(
                f"Step {step:02d}: "
                f"action={selected_action_id}, "
                f"branch={selected_branch_id}, "
                f"reward={step_result.reward:.4f}, "
                f"done={step_result.done}, "
                f"solved={step_result.solved}"
            )

            if step_result.done:
                break

        final_done = bool(env.done)
        final_solved = bool(env.solved)
        final_reason = env.termination_reason
        final_state = env.current_state

        terminal_reward = terminal_outcome_reward(
            state=final_state,
            solved=final_solved,
            termination_reason=final_reason,
            terminal_unsolved_penalty=args.terminal_unsolved_penalty,
            terminal_handoff_penalty=args.terminal_handoff_penalty,
            terminal_failure_penalty=args.terminal_failure_penalty,
            terminal_penalty_weight=args.terminal_penalty_weight,
        )

        rewards_with_terminal = [*rewards, terminal_reward]
        returns_with_terminal = discounted_returns(rewards_with_terminal, args.gamma)
        returns = returns_with_terminal[:-1]

        final_return = returns[0] if returns else terminal_reward

        for item, return_from_step in zip(pending_examples, returns):
            replay_buffer.add_example(
                state=item["state"],
                state_id=item["state_id"],
                action_mask=item["action_mask"],
                scenario_id=item["scenario_id"],
                step=item["step"],
                selected_action_id=item["selected_action_id"],
                selected_branch_id=item["selected_branch_id"],
                step_reward=item["step_reward"],
                final_return=final_return,
                discounted_return_from_step=float(return_from_step),
                solved=final_solved,
                done=final_done,
                termination_reason=final_reason,
                visit_counts=item["visit_counts"],
                mcts_policy=item["mcts_policy"],
                extra_metadata={
                    "source": "mcts_self_play",
                    "scenario_id": int(scenario_id),
                    "step": int(item["step"]),
                    "mcts_simulations": int(args.simulations),
                    "mcts_depth": int(args.depth),
                    "mcts_top_k": int(args.top_k),
                },
            )

            total_examples += 1

        print(
            f"Scenario {scenario_id} finished: "
            f"steps={len(rewards)}, "
            f"terminal_reward={terminal_reward:.4f}, "
            f"final_return={final_return:.4f}, "
            f"solved={final_solved}, "
            f"reason={final_reason}"
        )

    examples_path = replay_buffer.save()

    print("\nPower flow cache:")
    print(backend.cache_info())

    print("\nAction space cache:")
    print(action_space.cache_info())

    if evaluator is not None:
        print("\nNeural evaluator cache:")
        print(evaluator.cache_info())

    print("\n" + "=" * 100)
    print("Self-play generation summary")
    print("=" * 100)

    print(f"Total examples: {total_examples}")
    print(f"Saved examples: {examples_path}")
    print(f"States dir:     {replay_buffer.states_dir}")

    elapsed = time.perf_counter() - start_time

    print("\nTiming:")
    print(f"Self-play generation elapsed time: {elapsed:.4f} sec")

    print("\nDone.")


if __name__ == "__main__":
    main()
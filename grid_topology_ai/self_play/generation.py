from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.config import GenerationConfig

GridFMActionSpace = None
TopologySwitchingEnv = None
GridFMPowerFlowBackend = None
GridFMReward = None
GridFMAdapter = None
NeuralPolicyValueEvaluator = None
MCTSConfig = None
MCTSPlanner = None
SelfPlayReplayBuffer = None
analyze_root_branches = None
make_do_nothing_action = None
BRANCH_FEATURE_COLUMNS = None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    raw_dir: Path
    transitions_csv: Path
    output_dir: Path
    checkpoint: Path | None
    config: GenerationConfig
    seed: int
    clear_cache_between_scenarios: bool
    scenario_ids: tuple[int, ...] | None = None
    device: str = "cpu"
    enable_cache: bool = True
    root_dirichlet_alpha: float = 0.30
    root_exploration_fraction: float = 0.25
    min_hard_improvement: float = 50.0
    min_soft_improvement: float = 15.0
    min_gate_visits: int = 5
    min_gate_visit_fraction: float = 0.01


def _ensure_runtime_dependencies() -> None:
    global GridFMActionSpace
    global TopologySwitchingEnv
    global GridFMPowerFlowBackend
    global GridFMReward
    global GridFMAdapter
    global NeuralPolicyValueEvaluator
    global MCTSConfig
    global MCTSPlanner
    global SelfPlayReplayBuffer
    global analyze_root_branches
    global make_do_nothing_action
    global BRANCH_FEATURE_COLUMNS

    if GridFMActionSpace is not None:
        return

    from grid_topology_ai.action_space import GridFMActionSpace as _ActionSpace
    from grid_topology_ai.data_adapter import (
        BRANCH_FEATURE_COLUMNS as _BRANCH_FEATURE_COLUMNS,
    )
    from grid_topology_ai.data_adapter import GridFMAdapter as _Adapter
    from grid_topology_ai.environment import TopologySwitchingEnv as _Env
    from grid_topology_ai.models.neural_evaluator import (
        NeuralPolicyValueEvaluator as _Evaluator,
    )
    from grid_topology_ai.pypower_backend import (
        GridFMPowerFlowBackend as _Backend,
    )
    from grid_topology_ai.reward import GridFMReward as _Reward
    from grid_topology_ai.search.continuation_gate import (
        analyze_root_branches as _analyze_root_branches,
    )
    from grid_topology_ai.search.continuation_gate import (
        make_do_nothing_action as _make_do_nothing_action,
    )
    from grid_topology_ai.search.mcts import MCTSConfig as _MCTSConfig
    from grid_topology_ai.search.mcts import MCTSPlanner as _MCTSPlanner
    from grid_topology_ai.self_play.replay_buffer import (
        SelfPlayReplayBuffer as _ReplayBuffer,
    )

    GridFMActionSpace = _ActionSpace
    TopologySwitchingEnv = _Env
    GridFMPowerFlowBackend = _Backend
    GridFMReward = _Reward
    GridFMAdapter = _Adapter
    NeuralPolicyValueEvaluator = _Evaluator
    MCTSConfig = _MCTSConfig
    MCTSPlanner = _MCTSPlanner
    SelfPlayReplayBuffer = _ReplayBuffer
    analyze_root_branches = _analyze_root_branches
    make_do_nothing_action = _make_do_nothing_action
    BRANCH_FEATURE_COLUMNS = _BRANCH_FEATURE_COLUMNS


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
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


def make_one_hot_policy(action_id: int) -> dict[int, float]:
    return {int(action_id): 1.0}


def state_security_penalty(state: object) -> float:
    if BRANCH_FEATURE_COLUMNS is None:
        raise RuntimeError("Generation runtime dependencies are not loaded.")

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
    state: object | None,
    solved: bool,
    termination_reason: str | None,
    terminal_unsolved_penalty: float,
    terminal_handoff_penalty: float,
    terminal_failure_penalty: float,
    terminal_penalty_weight: float,
) -> float:
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


def _scenario_ids_from_request(request: GenerationRequest) -> list[int]:
    if not request.transitions_csv.exists():
        raise FileNotFoundError(
            f"Transitions file not found: {request.transitions_csv}"
        )

    transitions = pd.read_csv(request.transitions_csv)

    if request.scenario_ids is not None:
        return [int(value) for value in request.scenario_ids]

    return sorted(int(x) for x in transitions["scenario_id"].unique())


def generate_self_play_examples(request: GenerationRequest) -> Path:
    scenario_ids = _scenario_ids_from_request(request)
    _ensure_runtime_dependencies()
    request.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Generating AlphaZero-like self-play data")
    print("=" * 100)
    print(f"Raw directory:  {request.raw_dir.resolve()}")
    print(f"Transitions:    {request.transitions_csv.resolve()}")
    print(f"Output dir:     {request.output_dir}")
    print(f"Simulations:    {request.config.simulations}")
    print(f"Search depth:   {request.config.depth}")
    print(f"Max steps:      {request.config.max_steps}")
    print(f"Top-K actions:  {request.config.top_k}")
    print(f"Gamma:          {request.config.gamma}")
    print(f"C_PUCT:         {request.config.c_puct}")
    print(f"Prior exponent: {request.config.prior_exponent}")
    print(
        f"Terminal unsolved penalty: "
        f"{request.config.terminal_unsolved_penalty}"
    )
    print(
        f"Terminal penalty weight:   "
        f"{request.config.terminal_penalty_weight}"
    )
    print(
        f"Terminal handoff penalty:  "
        f"{request.config.terminal_handoff_penalty}"
    )
    print(
        f"Terminal failure penalty:  "
        f"{request.config.terminal_failure_penalty}"
    )
    print(f"Stop policy:               {request.config.stop_policy}")
    print(f"Checkpoint:     {request.checkpoint}")
    print(f"Device:         {request.device}")
    print(f"Use root noise: {request.config.use_root_noise}")
    print(f"Root alpha:     {request.root_dirichlet_alpha}")
    print(f"Root epsilon:   {request.root_exploration_fraction}")
    print(f"Temperature:    {request.config.selection_temperature}")
    print(f"Seed:           {request.seed}")
    print(f"PF algorithm:   {request.config.pf_alg}")
    print(f"Cache enabled:  {request.enable_cache}")
    print(
        "Clear cache between scenarios: "
        f"{request.clear_cache_between_scenarios}"
    )
    if request.config.selection_temperature <= 1e-8:
        print("Action selection: deterministic argmax")
    else:
        print("Action selection: sampling from MCTS policy")
    print(f"Continuation gate: {request.config.use_continuation_gate}")

    if request.config.use_continuation_gate:
        print(f"  min hard improvement: {request.min_hard_improvement}")
        print(f"  min soft improvement: {request.min_soft_improvement}")
        print(f"  min gate visits:      {request.min_gate_visits}")
        print(f"  min gate visit frac:  {request.min_gate_visit_fraction}")

    print(f"\nScenario IDs: {scenario_ids}")

    adapter = GridFMAdapter(request.raw_dir)
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=request.config.pf_alg,
        enable_cache=request.enable_cache,
    )
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=request.enable_cache,
    )
    reward_fn = GridFMReward()

    mcts_config = MCTSConfig(
        num_simulations=request.config.simulations,
        max_depth=request.config.depth,
        top_k_actions=request.config.top_k,
        gamma=request.config.gamma,
        c_puct=request.config.c_puct,
        leaf_penalty_weight=0.10,
        include_stop_action=True,
        prior_exponent=request.config.prior_exponent,
        stop_policy=request.config.stop_policy,
        use_root_dirichlet_noise=request.config.use_root_noise,
        root_dirichlet_alpha=request.root_dirichlet_alpha,
        root_exploration_fraction=request.root_exploration_fraction,
        random_seed=request.seed,
    )

    evaluator = None

    if request.checkpoint is not None:
        evaluator = NeuralPolicyValueEvaluator(
            checkpoint_path=request.checkpoint,
            device=request.device,
            enable_cache=request.enable_cache,
        )
        print("\nNeural evaluator loaded.")

    rng = np.random.default_rng(request.seed)
    planner = MCTSPlanner(config=mcts_config, evaluator=evaluator)
    replay_buffer = SelfPlayReplayBuffer(request.output_dir)

    total_examples = 0
    start_time = time.perf_counter()
    for scenario_id in scenario_ids:
        print("\n" + "=" * 100)
        print(f"Scenario {scenario_id}")
        print("=" * 100)
        if request.clear_cache_between_scenarios:
            backend.clear_cache()
            action_space.clear_cache()

            if evaluator is not None:
                evaluator.clear_cache()

        env = TopologySwitchingEnv(
            adapter=adapter,
            backend=backend,
            action_space=action_space,
            reward_fn=reward_fn,
            max_steps=request.config.max_steps,
        )
        env.reset(scenario_id)

        pending_examples: list[dict[str, Any]] = []
        rewards: list[float] = []

        for step in range(request.config.max_steps):
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

            raw_selected_action_id = select_action_from_policy(
                policy=search_result.policy,
                temperature=request.config.selection_temperature,
                rng=rng,
            )
            raw_selected_action = search_result.root.actions_by_id[
                raw_selected_action_id
            ]
            raw_selected_branch_id = raw_selected_action.branch_id
            gate_decision = None

            if request.config.use_continuation_gate:
                gate_decision = analyze_root_branches(
                    result=search_result,
                    min_hard_improvement=request.min_hard_improvement,
                    min_soft_improvement=request.min_soft_improvement,
                    min_visits=request.min_gate_visits,
                    min_visit_fraction=request.min_gate_visit_fraction,
                )
                selected_action_id = int(gate_decision.selected_action_id)
                selected_branch_id = gate_decision.selected_branch_id
                policy_target = make_one_hot_policy(selected_action_id)
            else:
                selected_action_id = int(raw_selected_action_id)
                selected_branch_id = raw_selected_branch_id
                policy_target = search_result.policy

            if selected_action_id == 0:
                selected_action = make_do_nothing_action()
            else:
                selected_action = search_result.root.actions_by_id.get(
                    selected_action_id
                )

                if selected_action is None:
                    selected_action = env.action_by_id(selected_action_id)

            step_result = env.step(selected_action)
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
                    "mcts_policy": policy_target,
                    "done": bool(step_result.done),
                    "solved": bool(step_result.solved),
                    "termination_reason": step_result.info[
                        "termination_reason"
                    ],
                    "raw_selected_action_id": raw_selected_action_id,
                    "raw_selected_branch_id": raw_selected_branch_id,
                    "gate_used": bool(request.config.use_continuation_gate),
                    "gate_reason": (
                        None
                        if gate_decision is None
                        else gate_decision.selected_reason
                    ),
                }
            )

            if gate_decision is None:
                print(
                    f"Step {step:02d}: "
                    f"action={selected_action_id}, "
                    f"branch={selected_branch_id}, "
                    f"reward={step_result.reward:.4f}, "
                    f"done={step_result.done}, "
                    f"solved={step_result.solved}"
                )
            else:
                print(
                    f"Step {step:02d}: "
                    f"raw_action={raw_selected_action_id}, "
                    f"raw_branch={raw_selected_branch_id}, "
                    f"gate_action={selected_action_id}, "
                    f"gate_branch={selected_branch_id}, "
                    f"gate_reason={gate_decision.selected_reason}, "
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
            terminal_unsolved_penalty=request.config.terminal_unsolved_penalty,
            terminal_handoff_penalty=request.config.terminal_handoff_penalty,
            terminal_failure_penalty=request.config.terminal_failure_penalty,
            terminal_penalty_weight=request.config.terminal_penalty_weight,
        )

        rewards_with_terminal = [*rewards, terminal_reward]
        returns_with_terminal = discounted_returns(
            rewards_with_terminal,
            request.config.gamma,
        )
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
                    "mcts_simulations": int(request.config.simulations),
                    "mcts_depth": int(request.config.depth),
                    "mcts_top_k": int(request.config.top_k),
                    "use_continuation_gate": bool(
                        request.config.use_continuation_gate
                    ),
                    "raw_selected_action_id": int(
                        item["raw_selected_action_id"]
                    ),
                    "raw_selected_branch_id": (
                        None
                        if item["raw_selected_branch_id"] is None
                        else int(item["raw_selected_branch_id"])
                    ),
                    "gate_reason": item["gate_reason"],
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

    return examples_path

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.config.physics import (
    PhysicsConfig,
    resolve_physics_config,
)
from grid_topology_ai.search.root_policy import (
    require_action_in_policy_support,
    select_policy_action,
)


_RUNTIME_DEPENDENCIES_LOADED = False

GridFMActionSpace = None
TopologySwitchingEnv = None
GridFMPowerFlowBackend = None
GridFMReward = None
GridFMAdapter = None
NeuralPolicyValueEvaluator = None
MCTSConfig = None
MCTSPlanner = None
ExampleWriter = None
analyze_root_branches = None
make_do_nothing_action = None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    raw_dir: Path
    transitions_csv: Path
    output_dir: Path
    checkpoint: Path | None
    config: GenerationConfig
    mcts_seed: int
    action_seed: int
    clear_cache_between_scenarios: bool
    iteration: int = 1
    physics_config: PhysicsConfig | None = None
    scenario_ids: tuple[int, ...] | None = None
    device: str = "cpu"
    enable_cache: bool = True
    root_dirichlet_alpha: float = 0.30
    root_exploration_fraction: float = 0.25
    min_hard_improvement: float = 50.0
    min_soft_improvement: float = 15.0
    min_gate_visits: int = 5
    min_gate_visit_fraction: float = 0.01

    def __post_init__(self) -> None:
        for field_name in ("mcts_seed", "action_seed"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer."
                )
            seed = int(value)
            if seed < 0:
                raise ValueError(
                    f"{field_name} must be a non-negative integer."
                )
            object.__setattr__(self, field_name, seed)

        if (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, (int, np.integer))
        ):
            raise ValueError("iteration must be a positive integer.")
        iteration = int(self.iteration)
        if iteration <= 0:
            raise ValueError("iteration must be a positive integer.")
        object.__setattr__(self, "iteration", iteration)

    @property
    def resolved_physics_config(self) -> PhysicsConfig:
        return resolve_physics_config(
            self.physics_config,
            self.config.pf_alg,
        )


@dataclass(frozen=True, slots=True)
class _GenerationActionDecision:
    selected_action_id: int
    selected_branch_id: int | None
    policy_target: dict[int, float]
    continuation_analysis: Any | None


def _ensure_runtime_dependencies() -> None:
    global _RUNTIME_DEPENDENCIES_LOADED
    global GridFMActionSpace
    global TopologySwitchingEnv
    global GridFMPowerFlowBackend
    global GridFMReward
    global GridFMAdapter
    global NeuralPolicyValueEvaluator
    global MCTSConfig
    global MCTSPlanner
    global ExampleWriter
    global analyze_root_branches
    global make_do_nothing_action

    if _RUNTIME_DEPENDENCIES_LOADED:
        return

    from grid_topology_ai.action_space import GridFMActionSpace as _ActionSpace
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
    from grid_topology_ai.self_play.examples import (
        ExampleWriter as _ExampleWriter,
    )

    GridFMActionSpace = _ActionSpace
    TopologySwitchingEnv = _Env
    GridFMPowerFlowBackend = _Backend
    GridFMReward = _Reward
    GridFMAdapter = _Adapter
    NeuralPolicyValueEvaluator = _Evaluator
    MCTSConfig = _MCTSConfig
    MCTSPlanner = _MCTSPlanner
    ExampleWriter = _ExampleWriter
    analyze_root_branches = _analyze_root_branches
    make_do_nothing_action = _make_do_nothing_action
    _RUNTIME_DEPENDENCIES_LOADED = True


def discounted_returns(
    rewards: list[float],
    gamma: float,
) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def _scenario_seed(stream_seed: int, scenario_id: int) -> int:
    sequence = np.random.SeedSequence(
        [int(stream_seed), int(scenario_id)]
    )
    state = sequence.generate_state(1, dtype=np.uint64)
    return int(state[0])


def _policy_entropy(
    policy: dict[int, float],
) -> tuple[float, float]:
    probabilities = np.asarray(
        [
            float(probability)
            for probability in policy.values()
            if float(probability) > 0.0
        ],
        dtype=np.float64,
    )
    if probabilities.size == 0:
        return 0.0, 0.0

    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "Policy probabilities must have a positive finite sum."
        )
    probabilities = probabilities / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    if probabilities.size <= 1:
        return entropy, 0.0

    normalized = entropy / math.log(int(probabilities.size))
    return entropy, min(1.0, max(0.0, float(normalized)))


def selection_temperature_for_step(
    config: GenerationConfig,
    *,
    iteration: int,
    step: int,
) -> float:
    if iteration <= 0:
        raise ValueError("iteration must be positive.")
    if step < 0:
        raise ValueError("step must be non-negative.")
    if config.selection_temperature <= 0.0:
        return 0.0
    if (
        config.temperature_iterations <= 0
        or config.temperature_steps <= 0
    ):
        return 0.0
    if iteration > config.temperature_iterations:
        return 0.0
    if step >= config.temperature_steps:
        return 0.0
    return float(config.selection_temperature)


def _select_generation_action(
    *,
    search_result: Any,
    temperature: float,
    rng: np.random.Generator,
    use_continuation_gate: bool,
    min_hard_improvement: float,
    min_soft_improvement: float,
    min_gate_visits: int,
    min_gate_visit_fraction: float,
    physics_config: PhysicsConfig | None = None,
    scenario_id: int | None = None,
    step: int | None = None,
) -> _GenerationActionDecision:
    context = (
        "self-play behavior policy "
        f"(scenario_id={scenario_id}, step={step})"
    )
    selection = select_policy_action(
        search_result.policy,
        temperature,
        rng,
        context=context,
    )
    selected_action_id = int(selection.action_id)
    policy_target = dict(selection.policy)
    require_action_in_policy_support(
        selected_action_id,
        policy_target,
        context=context,
    )

    if selected_action_id == 0:
        selected_branch_id = None
    else:
        selected_action = search_result.root.actions_by_id.get(
            selected_action_id
        )
        if selected_action is None:
            raise RuntimeError(
                f"Action {selected_action_id} is present in {context} but "
                "missing from root.actions_by_id."
            )
        selected_branch_id = selected_action.branch_id

    continuation_analysis = None
    if use_continuation_gate:
        continuation_analysis = analyze_root_branches(
            result=search_result,
            min_hard_improvement=min_hard_improvement,
            min_soft_improvement=min_soft_improvement,
            min_visits=min_gate_visits,
            min_visit_fraction=min_gate_visit_fraction,
            physics_config=physics_config,
        )

    return _GenerationActionDecision(
        selected_action_id=selected_action_id,
        selected_branch_id=selected_branch_id,
        policy_target=policy_target,
        continuation_analysis=continuation_analysis,
    )


def _scenario_ids_from_request(
    request: GenerationRequest,
) -> list[int]:
    if not request.transitions_csv.exists():
        raise FileNotFoundError(
            f"Transitions file not found: {request.transitions_csv}"
        )
    transitions = pd.read_csv(request.transitions_csv)
    if request.scenario_ids is not None:
        return [int(value) for value in request.scenario_ids]
    return sorted(
        int(value)
        for value in transitions["scenario_id"].unique()
    )


def _continuation_metadata(
    analysis: Any | None,
    selected_action_id: int,
) -> dict[str, Any]:
    if analysis is None:
        return {
            "continuation_allowed_action_ids": None,
            "continuation_recommended_action_id": None,
            "continuation_recommended_branch_id": None,
            "continuation_recommendation_reason": None,
            "selected_action_allowed_by_continuation": None,
        }

    allowed_action_ids = tuple(
        int(action_id)
        for action_id in getattr(analysis, "allowed_action_ids", ())
    )
    recommended_action_id = getattr(
        analysis,
        "recommended_action_id",
        getattr(analysis, "selected_action_id", None),
    )
    recommended_branch_id = getattr(
        analysis,
        "recommended_branch_id",
        getattr(analysis, "selected_branch_id", None),
    )
    recommendation_reason = getattr(
        analysis,
        "recommendation_reason",
        getattr(analysis, "selected_reason", None),
    )
    return {
        "continuation_allowed_action_ids": list(allowed_action_ids),
        "continuation_recommended_action_id": (
            None
            if recommended_action_id is None
            else int(recommended_action_id)
        ),
        "continuation_recommended_branch_id": (
            None
            if recommended_branch_id is None
            else int(recommended_branch_id)
        ),
        "continuation_recommendation_reason": recommendation_reason,
        "selected_action_allowed_by_continuation": (
            int(selected_action_id) in set(allowed_action_ids)
        ),
    }


def _step_metadata(
    *,
    request: GenerationRequest,
    scenario_id: int,
    step: int,
    scenario_mcts_seed: int,
    scenario_action_seed: int,
    selection_temperature: float,
    selection_mode: str,
    policy_target_entropy: float,
    policy_target_normalized_entropy: float,
    search_result: Any,
    continuation_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "mcts_self_play",
        "scenario_id": int(scenario_id),
        "step": int(step),
        "mcts_stream_seed": int(request.mcts_seed),
        "action_sampling_stream_seed": int(request.action_seed),
        "scenario_mcts_seed": int(scenario_mcts_seed),
        "scenario_action_sampling_seed": int(scenario_action_seed),
        "self_play_iteration": int(request.iteration),
        "selection_temperature": float(selection_temperature),
        "policy_target_entropy": float(policy_target_entropy),
        "policy_target_normalized_entropy": float(
            policy_target_normalized_entropy
        ),
        "selection_mode": selection_mode,
        "temperature_steps": int(request.config.temperature_steps),
        "temperature_iterations": int(
            request.config.temperature_iterations
        ),
        "mcts_simulations": int(request.config.simulations),
        "mcts_depth": int(request.config.depth),
        "mcts_top_k": int(request.config.top_k),
        "mcts_widening_coefficient": float(
            request.config.widening_coefficient
        ),
        "mcts_widening_exponent": float(
            request.config.widening_exponent
        ),
        "mcts_exploration_quota": int(
            request.config.exploration_quota
        ),
        "mcts_legal_action_count": int(
            search_result.root_legal_action_count
        ),
        "mcts_considered_action_count": int(
            search_result.root_considered_action_count
        ),
        "mcts_visited_action_count": int(
            search_result.root_visited_action_count
        ),
        "mcts_action_coverage": float(
            search_result.root_action_coverage
        ),
        "mcts_visited_action_coverage": float(
            search_result.root_visited_action_coverage
        ),
        "pf_alg": request.resolved_physics_config.pf_alg,
        "use_continuation_gate": bool(
            request.config.use_continuation_gate
        ),
        "policy_target_source": (
            "temperature_adjusted_mcts_visit_distribution"
        ),
        "execution_action_source": "policy_target_sampling",
        "mcts_best_action_id": (
            None
            if search_result.best_action_id is None
            else int(search_result.best_action_id)
        ),
        "mcts_best_branch_id": (
            None
            if getattr(search_result, "best_branch_id", None) is None
            else int(search_result.best_branch_id)
        ),
        **continuation_metadata,
    }


_PF_PERFORMANCE_COUNTERS = (
    "hits",
    "misses",
    "exact_cache_hits",
    "tolerant_cache_hits",
    "warm_start_hits",
    "cold_start_misses",
    "stock_runpf_calls",
    "q_limit_resolves",
)


def _power_flow_performance_snapshot(backend: Any) -> dict[str, object]:
    performance_info = getattr(backend, "performance_info", None)
    if callable(performance_info):
        return dict(performance_info())
    cache_info = getattr(backend, "cache_info", None)
    if callable(cache_info):
        return dict(cache_info())
    return {}


def _new_power_flow_performance_summary(enabled: bool) -> dict[str, object]:
    summary: dict[str, object] = {
        "enabled": bool(enabled),
        "scenarios": 0,
        "peak_cache_size": 0,
        "peak_topology_cache_buckets": 0,
        "peak_topology_cache_entries": 0,
    }
    for key in _PF_PERFORMANCE_COUNTERS:
        summary[key] = 0
    return summary


def _record_power_flow_scenario(
    summary: dict[str, object],
    before: dict[str, object],
    after: dict[str, object],
) -> None:
    summary["scenarios"] = int(summary["scenarios"]) + 1
    for key in _PF_PERFORMANCE_COUNTERS:
        delta = max(
            int(after.get(key, 0)) - int(before.get(key, 0)),
            0,
        )
        summary[key] = int(summary[key]) + delta

    summary["peak_cache_size"] = max(
        int(summary["peak_cache_size"]),
        int(after.get("size", 0)),
    )
    summary["peak_topology_cache_buckets"] = max(
        int(summary["peak_topology_cache_buckets"]),
        int(after.get("topology_cache_buckets", 0)),
    )
    summary["peak_topology_cache_entries"] = max(
        int(summary["peak_topology_cache_entries"]),
        int(after.get("topology_cache_entries", 0)),
    )


def _finalize_power_flow_performance_summary(
    summary: dict[str, object],
) -> dict[str, object]:
    result = dict(summary)
    hits = int(result["hits"])
    misses = int(result["misses"])
    lookups = hits + misses
    warm_starts = int(result["warm_start_hits"])
    cold_starts = int(result["cold_start_misses"])
    stock_calls = int(result["stock_runpf_calls"])

    result["hit_rate"] = (
        float(hits) / float(lookups)
        if lookups > 0
        else 0.0
    )
    result["warm_start_rate"] = (
        float(warm_starts) / float(misses)
        if misses > 0
        else 0.0
    )
    result["cold_start_rate"] = (
        float(cold_starts) / float(misses)
        if misses > 0
        else 0.0
    )
    result["solves_per_cache_miss"] = (
        float(stock_calls) / float(misses)
        if misses > 0
        else 0.0
    )
    return result


def _print_generation_settings(
    request: GenerationRequest,
    scenario_ids: list[int],
) -> None:
    print("=" * 100)
    print("Generating AlphaZero-like self-play data")
    print("=" * 100)
    print(f"Raw directory:  {request.raw_dir.resolve()}")
    print(f"Transitions:    {request.transitions_csv.resolve()}")
    print(f"Output dir:     {request.output_dir}")
    print(f"Simulations:    {request.config.simulations}")
    print(f"Search depth:   {request.config.depth}")
    print(f"Max steps:      {request.config.max_steps}")
    print(f"Initial action width: {request.config.top_k}")
    print(f"Widening coefficient: {request.config.widening_coefficient}")
    print(f"Widening exponent:    {request.config.widening_exponent}")
    print(f"Exploration quota:    {request.config.exploration_quota}")
    print(f"Gamma:          {request.config.gamma}")
    print(f"C_PUCT:         {request.config.c_puct}")
    print(f"Prior exponent: {request.config.prior_exponent}")
    print(f"Stop policy:               {request.config.stop_policy}")
    print(
        "Require connected after switch: "
        f"{request.config.require_connected_after_switch}"
    )
    print(
        "Min loading for branch opening: "
        f"{request.config.min_loading_for_switch_percent}"
    )
    print(f"Closeable branch IDs: {request.config.closeable_branch_ids}")
    print(f"Checkpoint:     {request.checkpoint}")
    print(f"Device:         {request.device}")
    print(f"Use root noise: {request.config.use_root_noise}")
    print(f"Root alpha:     {request.root_dirichlet_alpha}")
    print(f"Root epsilon:   {request.root_exploration_fraction}")
    print(f"Self-play iteration: {request.iteration}")
    print(f"Early temperature:  {request.config.selection_temperature}")
    print(f"Temperature steps:  {request.config.temperature_steps}")
    print(f"Temperature iters:  {request.config.temperature_iterations}")
    print(f"MCTS stream seed:   {request.mcts_seed}")
    print(f"Action stream seed: {request.action_seed}")
    print(f"PF algorithm:   {request.resolved_physics_config.pf_alg}")
    print(f"Cache enabled:  {request.enable_cache}")
    print(
        "Clear cache between scenarios: "
        f"{request.clear_cache_between_scenarios}"
    )

    temperature_enabled = (
        request.config.selection_temperature > 1e-8
        and request.config.temperature_steps > 0
        and request.config.temperature_iterations > 0
    )
    if temperature_enabled:
        print(
            "Action selection: scheduled early sampling, "
            "then deterministic argmax"
        )
    else:
        print("Action selection: deterministic argmax")

    print(
        "Continuation analysis: "
        f"{request.config.use_continuation_gate}"
    )
    if request.config.use_continuation_gate:
        print(f"  min hard improvement: {request.min_hard_improvement}")
        print(f"  min soft improvement: {request.min_soft_improvement}")
        print(f"  min gate visits:      {request.min_gate_visits}")
        print(f"  min gate visit frac:  {request.min_gate_visit_fraction}")
    print(f"\nScenario IDs: {scenario_ids}")


def generate_self_play_examples(request: GenerationRequest) -> Path:
    scenario_ids = _scenario_ids_from_request(request)
    _ensure_runtime_dependencies()
    request.output_dir.mkdir(parents=True, exist_ok=True)
    _print_generation_settings(request, scenario_ids)

    adapter = GridFMAdapter(
        request.raw_dir,
        physics_config=request.resolved_physics_config,
    )
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=request.resolved_physics_config,
        enable_cache=request.enable_cache,
    )
    action_config = request.config.action_space_config
    action_space = GridFMActionSpace(
        require_connected_after_switch=(
            action_config.require_connected_after_switch
        ),
        min_loading_for_switch_percent=(
            action_config.min_loading_for_switch_percent
        ),
        closeable_branch_ids=action_config.closeable_branch_ids,
        enable_cache=request.enable_cache,
    )
    reward_fn = GridFMReward(
        physics_config=request.resolved_physics_config,
        discount_factor=request.config.gamma,
    )
    mcts_config = MCTSConfig(
        num_simulations=request.config.simulations,
        max_depth=request.config.depth,
        top_k_actions=request.config.top_k,
        widening_coefficient=request.config.widening_coefficient,
        widening_exponent=request.config.widening_exponent,
        exploration_quota=request.config.exploration_quota,
        gamma=request.config.gamma,
        c_puct=request.config.c_puct,
        include_stop_action=True,
        prior_exponent=request.config.prior_exponent,
        stop_policy=request.config.stop_policy,
        use_root_dirichlet_noise=request.config.use_root_noise,
        root_dirichlet_alpha=request.root_dirichlet_alpha,
        root_exploration_fraction=request.root_exploration_fraction,
        random_seed=request.mcts_seed,
    )

    evaluator = None
    if request.checkpoint is not None:
        evaluator = NeuralPolicyValueEvaluator(
            checkpoint_path=request.checkpoint,
            device=request.device,
            enable_cache=request.enable_cache,
            physics_config=request.resolved_physics_config,
        )
        if (
            evaluator.topology_action_config.contract_fingerprint()
            != action_space.config.contract_fingerprint()
        ):
            raise ValueError(
                "Configured self-play topology action space does not match "
                f"checkpoint {request.checkpoint}."
            )
        print("\nNeural evaluator loaded.")

    planner = MCTSPlanner(
        config=mcts_config,
        evaluator=evaluator,
        physics_config=request.resolved_physics_config,
    )
    example_writer = ExampleWriter(
        request.output_dir,
        physics_config=request.resolved_physics_config,
        action_space_config=action_space.config,
    )

    total_examples = 0
    power_flow_summary = _new_power_flow_performance_summary(
        request.enable_cache
    )
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

        power_flow_before = _power_flow_performance_snapshot(backend)

        scenario_mcts_seed = _scenario_seed(
            request.mcts_seed,
            scenario_id,
        )
        scenario_action_seed = _scenario_seed(
            request.action_seed,
            scenario_id,
        )
        planner.reset_rng(scenario_mcts_seed)
        action_rng = np.random.default_rng(scenario_action_seed)
        print(f"MCTS seed:          {scenario_mcts_seed}")
        print(f"Action sample seed: {scenario_action_seed}")

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
                raise RuntimeError(
                    "Active self-play environment has no current state."
                )

            action_mask = env.valid_action_mask()
            search_result = planner.search_from_env(env)
            if search_result.best_action_id is None:
                env.terminate_no_legal_action()
                print(
                    "MCTS returned no legal action. "
                    "Episode terminated with no_legal_action."
                )
                break

            temperature = selection_temperature_for_step(
                request.config,
                iteration=request.iteration,
                step=step,
            )
            selection_mode = "sample" if temperature > 1e-8 else "argmax"
            decision = _select_generation_action(
                search_result=search_result,
                temperature=temperature,
                rng=action_rng,
                use_continuation_gate=(
                    request.config.use_continuation_gate
                ),
                min_hard_improvement=request.min_hard_improvement,
                min_soft_improvement=request.min_soft_improvement,
                min_gate_visits=request.min_gate_visits,
                min_gate_visit_fraction=request.min_gate_visit_fraction,
                scenario_id=scenario_id,
                step=step,
                physics_config=request.resolved_physics_config,
            )
            policy_entropy, normalized_entropy = _policy_entropy(
                decision.policy_target
            )
            require_action_in_policy_support(
                decision.selected_action_id,
                decision.policy_target,
                context=(
                    "self-play policy target "
                    f"(scenario_id={scenario_id}, step={step})"
                ),
            )

            if decision.selected_action_id == 0:
                selected_action = make_do_nothing_action()
            else:
                selected_action = search_result.root.actions_by_id[
                    decision.selected_action_id
                ]

            step_result = env.step(selected_action)
            rewards.append(float(step_result.reward))
            continuation = _continuation_metadata(
                decision.continuation_analysis,
                decision.selected_action_id,
            )
            pending_examples.append(
                {
                    "state": state_before,
                    "action_mask": action_mask,
                    "scenario_id": scenario_id,
                    "step": step,
                    "selected_action_id": decision.selected_action_id,
                    "selected_branch_id": decision.selected_branch_id,
                    "step_reward": float(step_result.reward),
                    "visit_counts": search_result.visit_counts,
                    "mcts_policy": decision.policy_target,
                    "selection_temperature": float(temperature),
                    "selection_mode": selection_mode,
                    "policy_target_entropy": float(policy_entropy),
                    "policy_target_normalized_entropy": float(
                        normalized_entropy
                    ),
                    "mcts_legal_action_count": int(
                        search_result.root_legal_action_count
                    ),
                    "mcts_considered_action_count": int(
                        search_result.root_considered_action_count
                    ),
                    "mcts_visited_action_count": int(
                        search_result.root_visited_action_count
                    ),
                    "mcts_action_coverage": float(
                        search_result.root_action_coverage
                    ),
                    "mcts_visited_action_coverage": float(
                        search_result.root_visited_action_coverage
                    ),
                    "extra_metadata": _step_metadata(
                        request=request,
                        scenario_id=scenario_id,
                        step=step,
                        scenario_mcts_seed=scenario_mcts_seed,
                        scenario_action_seed=scenario_action_seed,
                        selection_temperature=temperature,
                        selection_mode=selection_mode,
                        policy_target_entropy=policy_entropy,
                        policy_target_normalized_entropy=normalized_entropy,
                        search_result=search_result,
                        continuation_metadata=continuation,
                    ),
                }
            )

            print(
                f"Step {step:02d}: "
                f"action={decision.selected_action_id}, "
                f"branch={decision.selected_branch_id}, "
                f"temperature={temperature:.4f}, "
                f"selection={selection_mode}, "
                "continuation_recommendation="
                f"{continuation['continuation_recommended_action_id']}, "
                "continuation_reason="
                f"{continuation['continuation_recommendation_reason']}, "
                "coverage="
                f"{search_result.root_considered_action_count}/"
                f"{search_result.root_legal_action_count} "
                f"({search_result.root_action_coverage:.1%}), "
                "visited="
                f"{search_result.root_visited_action_count}/"
                f"{search_result.root_legal_action_count} "
                f"({search_result.root_visited_action_coverage:.1%}), "
                f"reward={step_result.reward:.4f}, "
                f"done={step_result.done}, "
                f"solved={step_result.solved}"
            )
            if step_result.done:
                break

        final_done = bool(env.done)
        final_reason = env.termination_reason
        final_evidence = env.terminal_outcome_evidence
        if not final_done or final_reason is None or final_evidence is None:
            raise RuntimeError(
                "Self-play episode ended without validated terminal "
                f"outcome evidence for scenario {scenario_id}."
            )

        returns = discounted_returns(rewards, request.config.gamma)
        final_return = returns[0] if returns else 0.0
        if pending_examples:
            total_examples += example_writer.add_episode(
                pending_examples,
                final_return=final_return,
                returns_from_step=returns,
                solved=bool(env.solved),
                done=final_done,
                termination_reason=final_reason,
                terminal_outcome_evidence=final_evidence,
                iteration=request.iteration,
            )

        print(
            f"Scenario {scenario_id} finished: "
            f"steps={len(rewards)}, "
            f"final_return={final_return:.4f}, "
            f"solved={env.solved}, "
            f"reason={final_reason}"
        )

        _record_power_flow_scenario(
            power_flow_summary,
            power_flow_before,
            _power_flow_performance_snapshot(backend),
        )

    examples_path = example_writer.save()

    print("\nPower flow cache:")
    print(_finalize_power_flow_performance_summary(power_flow_summary))
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
    print(f"States dir:     {example_writer.states_dir}")
    elapsed = time.perf_counter() - start_time
    print(f"\nTiming:\nSelf-play generation elapsed time: {elapsed:.4f} sec")
    print("\nDone.")
    return examples_path
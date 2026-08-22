from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Sequence


_NATIVE_MATH_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_EXACT_L1_CACHE_MAX_MB_ENV = "POWERGRID_EXACT_L1_CACHE_MAX_MB"
_WORKER_INIT_CONCURRENCY_ENV = "POWERGRID_TEACHER_INIT_CONCURRENCY"


def _configure_cache_runtime_from_cli() -> None:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--exact-cache-max-mb",
        type=float,
        default=None,
    )

    parsed, remaining = parser.parse_known_args(
        sys.argv[1:]
    )

    if parsed.exact_cache_max_mb is not None:
        max_mb = float(parsed.exact_cache_max_mb)

        if not math.isfinite(max_mb) or max_mb <= 0.0:
            raise SystemExit(
                "--exact-cache-max-mb must be a positive finite number."
            )

        os.environ[_EXACT_L1_CACHE_MAX_MB_ENV] = (
            f"{max_mb:.12g}"
        )

    sys.argv[:] = [
        sys.argv[0],
        *remaining,
    ]


def _configure_native_math_threads() -> None:
    """Avoid nested native thread pools inside multiprocessing workers."""

    for name in _NATIVE_MATH_THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")


# These must run before importing the numerical/cache stack.
_configure_cache_runtime_from_cli()
_configure_native_math_threads()

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grid_topology_ai.cache import DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES
from grid_topology_ai.actions import GridFMAction, GridFMActionSpace
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.teacher_config import (
    ensure_teacher_checkpoint_config,
    teacher_run_id,
    teacher_source_identity,
)
from grid_topology_ai.data import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.physics.utility import state_utility
from grid_topology_ai.physics.lodf import LODFStructureCache, lodf_loading_safety_score
from grid_topology_ai.physics.objective import (
    TerminalOutcomeEvidence,
    redispatch_status_for_reason,
)
from grid_topology_ai.physics.objective import assess_physical_state
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend
from grid_topology_ai.physics.redispatch import (
    MinimalRedispatchResult,
    empty_redispatch_diagnostics,
    run_minimal_ac_redispatch,
)
from grid_topology_ai.runtime import (
    ensure_runtime_scenario_store,
)
from grid_topology_ai.runtime import build_memory_mapped_teacher_context
from grid_topology_ai.physics.utility import GridFMReward
from grid_topology_ai.search.continuation_gate import make_do_nothing_action
from grid_topology_ai.search.impact_beam_search import (
    ImpactBeamSearchConfig,
    ImpactBeamSearchPlanner,
    LODFScreenedImpactBeamSearchPlanner,
    ImpactBeamSearchResult,
    safety_score,
)
from grid_topology_ai.search.impact_beam_search import switch_count
from grid_topology_ai.state import GridFMStateStore
from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
    termination_reason_value,
    validate_outcome_invariants,
)
from grid_topology_ai.topology_actions import (
    action_layout_fingerprint,
    action_layout_to_list,
    build_branch_action_slots,
)
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows

# ======================================================================================
# Worker-global context
# ======================================================================================

_WORKER_CONTEXT: dict[str, Any] | None = None


def _require_worker_context() -> dict[str, Any]:
    global _WORKER_CONTEXT

    if _WORKER_CONTEXT is None:
        raise RuntimeError(
            "Worker context is not initialized. "
            "This should not happen when using ProcessPoolExecutor initializer."
        )

    return _WORKER_CONTEXT




# ======================================================================================
# Small helpers
# ======================================================================================


def compute_auto_reward_scale_from_rows(
    rows: list[dict],
    quantile: float = 0.95,
    min_scale: float = 1.0,
) -> float:
    """
    Compute reward scale from generated step rewards.

    This scale is used only for value_target normalization,
    not for teacher search and not for action selection.
    """

    rewards = []

    for row in rows:
        if "step_reward" not in row:
            continue

        value = float(row["step_reward"])

        if math.isfinite(value):
            rewards.append(abs(value))

    if not rewards:
        return float(min_scale)

    rewards_sorted = sorted(rewards)

    q = min(max(float(quantile), 0.0), 1.0)
    index = int(round(q * (len(rewards_sorted) - 1)))

    scale = float(rewards_sorted[index])

    return max(scale, float(min_scale))


def add_normalized_value_targets_to_rows(
    rows: list[dict],
    gamma: float,
    reward_scale: float,
    group_keys: tuple[str, ...] = ("scenario_id",),
) -> None:
    """
    Add normalized value_target to generated teacher rows.

    Existing raw reward fields stay unchanged:
    - step_reward
    - discounted_return_from_step
    - final_return

    New value target:
        r_norm_t = tanh(step_reward_t / reward_scale)

        value_target_t =
            sum_k gamma^k * r_norm_{t+k}
            /
            sum_k gamma^k

    The denominator is important: it keeps value_target in [-1, 1],
    compatible with the Tanh value head.
    """

    if reward_scale <= 0:
        raise ValueError(f"reward_scale must be positive, got {reward_scale}")

    groups: dict[tuple, list[dict]] = {}

    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    for _, group_rows in groups.items():
        group_rows.sort(key=lambda r: int(r.get("step", 0)))

        normalized_rewards = [
            math.tanh(float(row.get("step_reward", 0.0)) / float(reward_scale))
            for row in group_rows
        ]

        n = len(group_rows)

        for i, row in enumerate(group_rows):
            weighted_sum = 0.0
            weight_sum = 0.0
            discount = 1.0

            for j in range(i, n):
                weighted_sum += discount * normalized_rewards[j]
                weight_sum += discount
                discount *= float(gamma)

            value_target = weighted_sum / max(weight_sum, 1e-12)

            row["value_target"] = float(value_target)
            row["value_target_mode"] = "tanh_step_reward_discounted_average"
            row["value_reward_scale"] = float(reward_scale)
            row["value_gamma"] = float(gamma)
            row["value_horizon_normalized"] = True

def discounted_returns(
    rewards: list[float],
    gamma: float,
) -> list[float]:
    """
    Compute discounted returns from every step.
    """

    returns = [0.0 for _ in rewards]
    running = 0.0

    for i in reversed(range(len(rewards))):
        running = float(rewards[i]) + float(gamma) * running
        returns[i] = float(running)

    return returns


def make_one_hot_policy(action_id: int) -> dict[int, float]:
    return {int(action_id): 1.0}


def make_policy_from_final_beam(
    result: ImpactBeamSearchResult,
    temperature: float,
) -> tuple[dict[int, float], dict[int, int]]:
    """
    Convert final beam into a policy over first actions.

    For teacher generation we usually use temperature=0, meaning one-hot target.
    """

    best_node = result.best_node

    if not best_node.action_ids:
        return {}, {}

    best_action_id = int(best_node.action_ids[0])

    if temperature <= 1e-12:
        return make_one_hot_policy(best_action_id), {best_action_id: 1}

    best_safety = float(best_node.safety_score)

    weights_by_action: dict[int, float] = {}
    counts_by_action: dict[int, int] = {}

    for node in result.final_beam:
        if not node.action_ids:
            continue

        action_id = int(node.action_ids[0])
        safety_gap = max(float(node.safety_score) - best_safety, 0.0)
        weight = float(np.exp(-safety_gap / float(temperature)))

        weights_by_action[action_id] = weights_by_action.get(action_id, 0.0) + weight
        counts_by_action[action_id] = counts_by_action.get(action_id, 0) + 1

    total = float(sum(weights_by_action.values()))

    if total <= 0.0:
        return make_one_hot_policy(best_action_id), {best_action_id: 1}

    policy = {
        int(action_id): float(weight / total)
        for action_id, weight in weights_by_action.items()
    }

    return policy, counts_by_action


def _force_stop_action_valid(action_mask: np.ndarray) -> np.ndarray:
    """
    Make sure action 0 can be used as handoff target.
    """

    fixed_mask = np.array(action_mask, dtype=bool).copy()

    if fixed_mask.shape[0] > 0:
        fixed_mask[0] = True

    return fixed_mask


def _action_is_valid(
    action_mask: np.ndarray,
    action_id: int,
) -> bool:
    action_id = int(action_id)

    if action_id < 0:
        return False

    if action_id >= int(action_mask.shape[0]):
        return False

    return bool(action_mask[action_id])


def _make_action_for_env(
    env: TopologySwitchingEnv,
    action_id: int,
):
    action_id = int(action_id)

    if action_id == 0:
        return make_do_nothing_action()

    return env.action_by_id(action_id)


def _get_state_hard_count(state) -> int:
    return int(state.metrics["num_hard_overloaded_branches"])


def _get_state_max_loading(state) -> float:
    return float(state.metrics["max_loading_percent"])


def should_continue_teacher_action(
    safety_before: float,
    safety_after: float,
    state_before,
    state_after,
    task: dict[str, Any],
) -> tuple[bool, str, float]:
    """
    Decide whether the teacher should execute the next topology action
    or hand off the remaining problem to redispatch.
    """

    if state_after is None:
        return False, "power_flow_failed", -float("inf")

    safety_before = float(safety_before)
    safety_after = float(safety_after)

    improvement = float(safety_before - safety_after)

    hard_before = _get_state_hard_count(state_before)
    hard_after = _get_state_hard_count(state_after)

    max_before = _get_state_max_loading(state_before)
    max_after = _get_state_max_loading(state_after)

    allow_hard_increase = bool(task["allow_hard_count_increase"])

    if hard_after > hard_before and not allow_hard_increase:
        return (
            False,
            f"hard_count_increase_{hard_before}_to_{hard_after}",
            improvement,
        )

    max_loading_increase_limit = float(task["max_loading_increase_limit"])

    if max_after > max_before + max_loading_increase_limit:
        return (
            False,
            f"max_loading_increase_{max_before:.2f}_to_{max_after:.2f}",
            improvement,
        )

    if hard_before > 0:
        required_improvement = float(task["min_continue_improvement_with_hard"])
    else:
        required_improvement = float(task["min_continue_improvement_without_hard"])

    if hard_after < hard_before and improvement > 0.0:
        return True, "hard_count_reduced", improvement

    if improvement < required_improvement:
        return (
            False,
            f"improvement_too_small_{improvement:.2f}_lt_{required_improvement:.2f}",
            improvement,
        )

    return True, "useful_safety_improvement", improvement


def make_handoff_step_item(
    step_idx: int,
    state_before,
    action_mask: np.ndarray,
    safety_before: float,
    reason: str,
) -> dict[str, Any]:
    """
    Create a training example for action 0 = handoff to redispatch.
    """

    fixed_action_mask = _force_stop_action_valid(action_mask)

    return {
        "step": int(step_idx),
        "state": state_before,
        "action_mask": fixed_action_mask,
        "selected_action_id": 0,
        "selected_branch_id": None,
        "policy_target": make_one_hot_policy(0),
        "visit_counts": {0: 1},
        "safety_before": float(safety_before),
        "safety_after": float(safety_before),
        "step_reward": 0.0,
        "env_reward": 0.0,
        "done_after_step": True,
        "solved_after_step": False,
        "termination_reason_after_step": (
            TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
        ),
        "teacher_decision_reason": reason,
    }


def _safe_short_sequence(best_node) -> str:
    if hasattr(best_node, "short_sequence"):
        return str(best_node.short_sequence())

    parts = []

    for branch_id in getattr(best_node, "branch_ids", []):
        parts.append("stop" if branch_id is None else str(branch_id))

    return " -> ".join(parts) if parts else "(root)"


# ======================================================================================
# LODF screening
# ======================================================================================





# ======================================================================================
# Scenario processing
# ======================================================================================


def _generate_scenario(scenario_id: int) -> dict[str, Any]:
    ctx = _require_worker_context()

    adapter = ctx["adapter"]
    backend = ctx["backend"]
    action_space = ctx["action_space"]
    reward_fn = ctx["reward_fn"]
    physics_config = ctx["physics_config"]
    state_store = ctx["state_store"]
    task = ctx["task_config"]

    scenario_id = int(scenario_id)

    try:
        search_env = TopologySwitchingEnv(
            adapter=adapter,
            backend=backend,
            action_space=action_space,
            reward_fn=reward_fn,
            max_steps=int(task["max_steps"]),
        )

        initial_state = search_env.reset(scenario_id)
        initial_safety = safety_score(initial_state, physics_config=physics_config)

        planner_config = ImpactBeamSearchConfig(
            max_depth=int(task["depth"]),
            beam_width=int(task["beam_width"]),
            candidate_pool_size=int(task["candidate_pool"]),
            top_k_actions=int(task["top_k"]),
            gamma=float(task["gamma"]),
            include_stop_action=True,
            allow_hard_count_increase=bool(task["allow_hard_count_increase"]),
            show_progress=False,
            progress_update_every=10,
        )

        if bool(task.get("use_lodf_screening", False)):
            planner = LODFScreenedImpactBeamSearchPlanner(
                config=planner_config,
                lodf_screen_top_k=int(task["lodf_screen_top_k"]),
                lodf_min_candidate_count=int(task["lodf_min_candidate_count"]),
                physics_config=physics_config,
                lodf_structure_cache=ctx.get(_RUNTIME_LODF_STRUCTURE_CACHE),
            )
        else:
            planner = ImpactBeamSearchPlanner(
                planner_config,
                physics_config=physics_config,
            )

        print(
            f"[worker {os.getpid()}] scenario {scenario_id}: beam search start",
            flush=True,
        )

        search_started = time.perf_counter()
        result = planner.search(env=search_env, scenario_id=scenario_id)
        result, selection_diagnostics = _redispatch_aware_selection(
            result,
            task_config=task,
        )
        _SELECTION_PROVENANCE_BY_SCENARIO[scenario_id] = _selection_provenance(
            result,
            selection_diagnostics,
        )

        print(
            f"[worker {os.getpid()}] scenario {scenario_id}: "
            f"beam search done in "
            f"{time.perf_counter() - search_started:.1f}s | "
            f"evaluated={result.evaluated_actions}",
            flush=True,
        )
        best = result.best_node

        if not best.action_ids:
            clear_worker_caches_if_needed()
            return {
                "ok": False,
                "scenario_id": scenario_id,
                "reason": "no_teacher_action_found",
                "traceback": None,
            }

        final_teacher_safety = float(best.safety_score)
        total_safety_improvement = float(initial_safety - final_teacher_safety)

        if total_safety_improvement < float(task["min_safety_improvement"]):
            clear_worker_caches_if_needed()
            return {
                "ok": False,
                "scenario_id": scenario_id,
                "reason": (
                    f"safety_improvement {total_safety_improvement:.4f} "
                    f"< {float(task['min_safety_improvement']):.4f}"
                ),
                "traceback": None,
            }

        root_policy_target, root_visit_counts = make_policy_from_final_beam(
            result=result,
            temperature=float(task["soft_policy_temperature"]),
        )

        if not root_policy_target:
            clear_worker_caches_if_needed()
            return {
                "ok": False,
                "scenario_id": scenario_id,
                "reason": "empty_root_policy_target",
                "traceback": None,
            }

        replay_env = TopologySwitchingEnv(
            adapter=adapter,
            backend=backend,
            action_space=action_space,
            reward_fn=reward_fn,
            max_steps=int(task["max_steps"]),
        )
        replay_env.reset(scenario_id)

        step_items: list[dict[str, Any]] = []
        step_rewards: list[float] = []
        max_teacher_steps = min(
            len(best.action_ids),
            int(task["max_teacher_steps"]),
        )
        handoff_added = False
        handoff_reason: str | None = None

        for step_idx in range(max_teacher_steps):
            if replay_env.done:
                break

            state_before = replay_env.current_state
            if state_before is None:
                break

            action_mask = replay_env.operational_action_mask()
            selected_action_id = int(best.action_ids[step_idx])
            selected_branch_id = best.branch_ids[step_idx]

            if not _selected_teacher_action_is_valid(
                action_mask, selected_action_id
            ):
                if bool(task["add_handoff_example"]):
                    safety_before = safety_score(
                        state_before,
                        physics_config=physics_config,
                    )
                    step_items.append(
                        make_handoff_step_item(
                            step_idx=step_idx,
                            state_before=state_before,
                            action_mask=action_mask,
                            safety_before=safety_before,
                            reason=f"teacher_action_invalid_{selected_action_id}",
                        )
                    )
                    step_rewards.append(0.0)
                    handoff_added = True
                    handoff_reason = f"teacher_action_invalid_{selected_action_id}"
                break

            safety_before = safety_score(
                state_before,
                physics_config=physics_config,
            )
            candidate_env = replay_env.clone()
            selected_action = _make_action_for_env(
                env=candidate_env,
                action_id=selected_action_id,
            )
            step_result = candidate_env.step(selected_action)
            next_state = step_result.next_state

            if next_state is None:
                safety_after = safety_before + float(task["power_flow_failure_penalty"])
            else:
                safety_after = safety_score(next_state, physics_config=physics_config)

            continue_action, continue_reason, step_improvement = (
                _selected_teacher_replay_decision(
                    safety_before=safety_before,
                    safety_after=safety_after,
                    state_before=state_before,
                    state_after=next_state,
                    task=task,
                )
            )

            if not continue_action:
                if bool(task["add_handoff_example"]):
                    step_items.append(
                        make_handoff_step_item(
                            step_idx=step_idx,
                            state_before=state_before,
                            action_mask=action_mask,
                            safety_before=safety_before,
                            reason=continue_reason,
                        )
                    )
                    step_rewards.append(0.0)
                    handoff_added = True
                    handoff_reason = continue_reason
                break

            replay_env = candidate_env
            env_reward = float(step_result.reward)

            if step_idx == 0 and bool(task["use_soft_root_policy"]):
                policy_target = root_policy_target
                visit_counts = root_visit_counts
            else:
                policy_target = make_one_hot_policy(selected_action_id)
                visit_counts = {int(selected_action_id): 1}

            step_items.append(
                {
                    "step": int(step_idx),
                    "state": state_before,
                    "action_mask": action_mask,
                    "selected_action_id": int(selected_action_id),
                    "selected_branch_id": (
                        None if selected_branch_id is None else int(selected_branch_id)
                    ),
                    "policy_target": policy_target,
                    "visit_counts": visit_counts,
                    "safety_before": float(safety_before),
                    "safety_after": float(safety_after),
                    "step_reward": float(step_improvement),
                    "env_reward": float(env_reward),
                    "done_after_step": bool(step_result.done),
                    "solved_after_step": bool(step_result.solved),
                    "termination_reason_after_step": step_result.info.get(
                        "termination_reason"
                    ),
                    "teacher_decision_reason": continue_reason,
                }
            )
            step_rewards.append(float(step_improvement))

            if step_result.done:
                break

        if (
            bool(task["add_handoff_example"])
            and not handoff_added
            and not replay_env.done
        ):
            final_teacher_state = replay_env.current_state
            if final_teacher_state is not None:
                final_action_mask = replay_env.operational_action_mask()
                final_safety_before = safety_score(
                    final_teacher_state,
                    physics_config=physics_config,
                )
                final_stop_step = len(step_items)
                step_items.append(
                    make_handoff_step_item(
                        step_idx=final_stop_step,
                        state_before=final_teacher_state,
                        action_mask=final_action_mask,
                        safety_before=final_safety_before,
                        reason="terminal_handoff_after_useful_sequence",
                    )
                )
                step_rewards.append(0.0)
                handoff_added = True
                handoff_reason = "terminal_handoff_after_useful_sequence"

        if not step_items:
            clear_worker_caches_if_needed()
            return {
                "ok": False,
                "scenario_id": scenario_id,
                "reason": "no_replay_steps_saved",
                "traceback": None,
            }

        returns = discounted_returns(
            rewards=step_rewards,
            gamma=float(task["gamma"]),
        )
        final_state = replay_env.current_state

        if final_state is None:
            final_safety = float("inf")
            final_max_loading = float("inf")
            final_num_hard = 10**9
            final_num_overloaded = 10**9
        else:
            final_safety = safety_score(final_state, physics_config=physics_config)
            final_max_loading = float(final_state.metrics["max_loading_percent"])
            final_num_hard = int(final_state.metrics["num_hard_overloaded_branches"])
            final_num_overloaded = int(final_state.metrics["num_overloaded_branches"])

        if handoff_added:
            episode_done = True
            episode_solved = False
            episode_reason = TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
        elif replay_env.done:
            episode_done = True
            episode_solved = bool(replay_env.solved)
            episode_reason = parse_termination_reason(
                replay_env.termination_reason,
                allow_none=False,
            )
        else:
            episode_done = True
            episode_solved = False
            episode_reason = TerminationReason.TEACHER_DEPTH_LIMIT

        validate_outcome_invariants(
            solved=episode_solved,
            termination_reason=episode_reason,
        )

        rows: list[dict[str, Any]] = []
        final_return = (
            float(returns[0])
            if returns
            else float(total_safety_improvement)
        )

        for item, return_from_step in zip(step_items, returns):
            step_idx = int(item["step"])
            state_id = f"impact_teacher_scenario_{scenario_id:06d}_step_{step_idx:03d}"
            state_path = state_store.save_state(
                state=item["state"],
                state_id=state_id,
                action_mask=item["action_mask"],
                extra_metadata={
                    "physics_config": physics_config.to_dict(),
                    "source": "impact_beam_teacher_multistep_fast",
                    "scenario_id": int(scenario_id),
                    "step": int(step_idx),
                    "initial_safety": float(initial_safety),
                    "teacher_final_safety": float(final_teacher_safety),
                    "replay_final_safety": float(final_safety),
                    "total_safety_improvement": float(total_safety_improvement),
                    "safety_before": float(item["safety_before"]),
                    "safety_after": float(item["safety_after"]),
                    "step_safety_improvement": float(item["step_reward"]),
                    "env_reward": float(item["env_reward"]),
                    "selected_action_id": int(item["selected_action_id"]),
                    "selected_branch_id": item["selected_branch_id"],
                    "teacher_decision_reason": item.get("teacher_decision_reason"),
                    "handoff_added": bool(handoff_added),
                    "handoff_reason": handoff_reason,
                    "episode_done": bool(episode_done),
                    "episode_solved": bool(episode_solved),
                    "episode_termination_reason": termination_reason_value(episode_reason),
                    "step_done": bool(item.get("done_after_step", False)),
                    "step_solved": bool(item.get("solved_after_step", False)),
                    "step_termination_reason": termination_reason_value(
                        parse_termination_reason(item.get("termination_reason_after_step"))
                    ),
                    "best_sequence_action_ids": [int(x) for x in best.action_ids],
                    "best_sequence_branch_ids": [
                        None if x is None else int(x) for x in best.branch_ids
                    ],
                    "best_max_loading_percent": float(best.max_loading_percent),
                    "best_num_hard_overloaded": int(best.num_hard_overloaded),
                    "best_num_overloaded": int(best.num_overloaded),
                    "best_total_hard_overload": float(best.total_hard_overload),
                    "best_squared_hard_overload": float(best.squared_hard_overload),
                    "best_total_overload": float(best.total_overload),
                    "replay_final_max_loading_percent": float(final_max_loading),
                    "replay_final_num_hard_overloaded": int(final_num_hard),
                    "replay_final_num_overloaded": int(final_num_overloaded),
                    "beam_depth": int(task["depth"]),
                    "beam_width": int(task["beam_width"]),
                    "candidate_pool": int(task["candidate_pool"]),
                    "top_k": int(task["top_k"]),
                    "soft_policy_temperature": float(task["soft_policy_temperature"]),
                    "use_soft_root_policy": bool(task["use_soft_root_policy"]),
                    "use_lodf_screening": bool(task.get("use_lodf_screening", False)),
                    "lodf_screen_top_k": int(task.get("lodf_screen_top_k", 0)),
                    "evaluated_actions": int(result.evaluated_actions),
                },
            )
            rows.append(
                {
                    "state_id": state_id,
                    "state_path": str(state_path),
                    "scenario_id": int(scenario_id),
                    "step": int(step_idx),
                    "selected_action_id": int(item["selected_action_id"]),
                    "selected_branch_id": item["selected_branch_id"],
                    "step_reward": float(item["step_reward"]),
                    "final_return": float(final_return),
                    "discounted_return_from_step": float(return_from_step),
                    "solved": bool(episode_solved),
                    "done": bool(episode_done),
                    "termination_reason": termination_reason_value(episode_reason),
                    "step_solved": bool(item.get("solved_after_step", False)),
                    "step_done": bool(item.get("done_after_step", False)),
                    "step_termination_reason": termination_reason_value(
                        parse_termination_reason(item.get("termination_reason_after_step"))
                    ),
                    "physics_config": json.dumps(
                        physics_config.to_dict(), sort_keys=True, separators=(",", ":")
                    ),
                    "visit_counts_json": json.dumps(
                        {str(k): int(v) for k, v in item["visit_counts"].items()}
                    ),
                    "mcts_policy_json": json.dumps(
                        {str(k): float(v) for k, v in item["policy_target"].items()}
                    ),
                }
            )

        clear_worker_caches_if_needed()
        return {
            "ok": True,
            "scenario_id": int(scenario_id),
            "rows": rows,
            "summary": {
                "num_examples": int(len(rows)),
                "first_action": int(best.action_ids[0]),
                "first_branch": (
                    None if best.branch_ids[0] is None else int(best.branch_ids[0])
                ),
                "initial_safety": float(initial_safety),
                "teacher_final_safety": float(final_teacher_safety),
                "replay_final_safety": float(final_safety),
                "total_safety_improvement": float(total_safety_improvement),
                "final_hard": int(final_num_hard),
                "final_overloaded": int(final_num_overloaded),
                "final_max_loading": float(final_max_loading),
                "sequence": _safe_short_sequence(best),
                "evaluated_actions": int(result.evaluated_actions),
                "handoff_added": bool(handoff_added),
                "handoff_reason": handoff_reason,
            },
        }
    except Exception:
        clear_worker_caches_if_needed()
        return {
            "ok": False,
            "scenario_id": scenario_id,
            "reason": "exception",
            "traceback": traceback.format_exc(),
        }




# ======================================================================================
# IO / CLI helpers
# ======================================================================================


def load_scenario_ids(
    transitions_path: Path,
    limit: int | None,
) -> list[int]:
    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")

    transitions = pd.read_csv(transitions_path)
    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions file must contain scenario_id column: {transitions_path}"
        )

    scenario_ids = [
        int(x)
        for x in transitions["scenario_id"].drop_duplicates().tolist()
    ]
    if limit is not None:
        scenario_ids = scenario_ids[: int(limit)]
    return scenario_ids


def chunk_list(values: list[int], batch_size: int) -> list[list[int]]:
    batch_size = max(int(batch_size), 1)
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def _console_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(str(message))
    else:
        print(str(message))


def print_success(result: dict[str, Any]) -> None:
    summary = result["summary"]
    _console_write(
        f"Scenario {result['scenario_id']}: saved | "
        f"examples={summary['num_examples']} | "
        f"first_action={summary['first_action']} | "
        f"first_branch={summary['first_branch']} | "
        f"safety {summary['initial_safety']:.2f} -> "
        f"{summary['teacher_final_safety']:.2f} | "
        f"improvement={summary['total_safety_improvement']:.2f} | "
        f"final_hard={summary['final_hard']} | "
        f"final_over={summary['final_overloaded']} | "
        f"final_max={summary['final_max_loading']:.2f}% | "
        f"eval={summary['evaluated_actions']} | "
        f"handoff={summary['handoff_added']} | "
        f"seq={summary['sequence']}"
    )


def print_failure(result: dict[str, Any]) -> None:
    _console_write(
        f"Scenario {result['scenario_id']}: skipped | "
        f"reason={result['reason']}"
    )
    if result.get("traceback"):
        _console_write(result["traceback"])


def make_task_config(args: argparse.Namespace) -> dict[str, Any]:
    physics_config = replace(
        DEFAULT_PHYSICS_CONFIG,
        pf_alg=int(args.pf_alg),
        max_iterations=int(args.pf_max_iter),
    )
    depth = int(args.depth)
    max_teacher_steps = int(args.max_teacher_steps)
    if depth <= 0 or max_teacher_steps <= 0:
        raise ValueError("Teacher depth and max_teacher_steps must be positive.")
    return {
        "depth": min(depth, max_teacher_steps),
        "beam_width": int(args.beam_width),
        "candidate_pool": int(args.candidate_pool),
        "top_k": int(args.top_k),
        "gamma": float(args.gamma),
        "pf_alg": int(args.pf_alg),
        "pf_max_iter": physics_config.max_iterations,
        "physics_config": physics_config.to_dict(),
        "max_steps": int(args.max_steps),
        "max_teacher_steps": max_teacher_steps,
        "soft_policy_temperature": float(args.soft_policy_temperature),
        "use_soft_root_policy": bool(args.use_soft_root_policy),
        "min_safety_improvement": 0.0,
        "allow_hard_count_increase": bool(args.allow_hard_count_increase),
        "disable_cache": bool(args.disable_cache),
        "power_flow_failure_penalty": float(args.power_flow_failure_penalty),
        "min_continue_improvement_with_hard": float(
            args.min_continue_improvement_with_hard
        ),
        "min_continue_improvement_without_hard": float(
            args.min_continue_improvement_without_hard
        ),
        "max_loading_increase_limit": float(args.max_loading_increase_limit),
        "add_handoff_example": bool(args.add_handoff_example),
        "use_lodf_screening": bool(args.use_lodf_screening),
        "lodf_screen_top_k": int(args.lodf_screen_top_k),
        "lodf_min_candidate_count": int(args.lodf_min_candidate_count),
        "min_meaningful_safety_improvement": _MIN_MEANINGFUL_SAFETY_IMPROVEMENT,
        "terminal_redispatch_relative_epsilon": _TERMINAL_REDISPATCH_RELATIVE_EPSILON,
        "terminal_redispatch_absolute_epsilon_mw": _TERMINAL_REDISPATCH_ABSOLUTE_EPSILON_MW,
    }


def resolve_num_workers(
    num_workers_arg: int,
    num_batches: int,
    task_config: dict[str, Any],
) -> int:
    del num_batches, task_config
    return max(int(num_workers_arg), 1)


def append_scenario_checkpoint(
    checkpoint_path: Path,
    result: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(result.get("ok", False))
    payload = {
        "scenario_id": int(result["scenario_id"]),
        "ok": ok,
        "reason": result.get("reason"),
        "rows": result.get("rows", []) if ok else [],
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    with checkpoint_path.open("ab+") as checkpoint_file:
        checkpoint_file.seek(0, os.SEEK_END)
        size = checkpoint_file.tell()
        if size > 0:
            checkpoint_file.seek(-1, os.SEEK_END)
            last_byte = checkpoint_file.read(1)
            checkpoint_file.seek(0, os.SEEK_END)
            if last_byte != b"\n":
                checkpoint_file.write(b"\n")
        checkpoint_file.write(encoded)
        checkpoint_file.flush()
        os.fsync(checkpoint_file.fileno())


def load_scenario_checkpoints(
    checkpoint_path: Path,
    allowed_scenario_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    allowed = {int(scenario_id) for scenario_id in allowed_scenario_ids}
    results: dict[int, dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return results

    with checkpoint_path.open("r", encoding="utf-8", errors="replace") as checkpoint_file:
        for line_number, raw_line in enumerate(checkpoint_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                scenario_id = int(payload["scenario_id"])
            except Exception:
                print(
                    "Warning: ignoring invalid checkpoint line "
                    f"{line_number}: {checkpoint_path}",
                    flush=True,
                )
                continue
            if scenario_id not in allowed:
                continue
            ok = bool(payload.get("ok", False))
            rows = payload.get("rows", [])
            if ok and (not isinstance(rows, list) or not rows):
                print(
                    "Warning: ignoring incomplete successful "
                    f"checkpoint for scenario {scenario_id}",
                    flush=True,
                )
                continue
            results[scenario_id] = {
                "scenario_id": scenario_id,
                "ok": ok,
                "reason": payload.get("reason"),
                "rows": rows if ok else [],
            }
    return results




def collect_rows_from_checkpoints(
    checkpoint_results: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0
    for scenario_id in sorted(checkpoint_results):
        result = checkpoint_results[scenario_id]
        if bool(result["ok"]):
            rows.extend(result["rows"])
            total_saved += 1
        else:
            total_skipped += 1
    return rows, total_saved, total_skipped


def run_sequential(
    scenario_batches: list[list[int]],
    scenario_ids: Sequence[int],
    raw_dir: Path,
    states_dir: Path,
    task_config: dict[str, Any],
    checkpoint_path: Path,
    verbose_success: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    init_worker_context(
        raw_dir_str=str(raw_dir),
        states_dir_str=str(states_dir),
        task_config=task_config,
        scenario_ids=scenario_ids,
        memory_registry=None,
    )
    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0
    iterator = scenario_batches
    if tqdm is not None:
        iterator = tqdm(
            scenario_batches,
            desc="Teacher batches",
            unit="batch",
            dynamic_ncols=True,
        )

    for batch in iterator:
        results = process_scenario_batch(batch)
        for result in results:
            append_scenario_checkpoint(checkpoint_path=checkpoint_path, result=result)
            if result["ok"]:
                rows.extend(result["rows"])
                total_saved += 1
                if verbose_success:
                    print_success(result)
            else:
                total_skipped += 1
                print_failure(result)
    return rows, total_saved, total_skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fast multi-step teacher generation using persistent worker contexts."
        )
    )
    parser.add_argument("raw_dir", type=str, help="Path to GridFM raw directory.")
    parser.add_argument(
        "--transitions",
        type=str,
        required=True,
        help="Transitions CSV with scenario_id column.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for teacher examples.",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Terminal-utility gamma. The current contract requires 1.0.",
    )
    parser.add_argument("--pf-alg", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--pf-max-iter", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-teacher-steps", type=int, default=4)
    parser.add_argument("--soft-policy-temperature", type=float, default=0.0)
    parser.add_argument("--use-soft-root-policy", action="store_true")
    parser.add_argument("--min-safety-improvement", type=float, default=0.0)
    parser.add_argument("--allow-hard-count-increase", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--power-flow-failure-penalty", type=float, default=1_000_000.0)
    parser.add_argument("--min-continue-improvement-with-hard", type=float, default=100.0)
    parser.add_argument("--min-continue-improvement-without-hard", type=float, default=150.0)
    parser.add_argument("--max-loading-increase-limit", type=float, default=5.0)
    parser.add_argument("--add-handoff-example", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Explicit number of multiprocessing workers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of scenarios per submitted worker task.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--quiet-success",
        action="store_true",
        help="Do not print one line for every successful scenario.",
    )
    parser.add_argument(
        "--use-lodf-screening",
        action="store_true",
        help=(
            "Use LODF/DC screening to prefilter candidate topology actions before "
            "expensive AC PF validation."
        ),
    )
    parser.add_argument(
        "--lodf-screen-top-k",
        type=int,
        default=0,
        help=(
            "Keep only this many LODF-ranked switch actions before AC PF. "
            "Use 0 to disable effective LODF pruning."
        ),
    )
    parser.add_argument(
        "--lodf-min-candidate-count",
        type=int,
        default=8,
        help="Apply LODF screening only if there are at least this many switch candidates.",
    )
    parser.add_argument(
        "--value-target-mode",
        type=str,
        default="tanh_step_reward_discounted_average",
        choices=[
            "legacy_discounted_return",
            "tanh_step_reward_discounted_average",
        ],
        help=(
            "How to create value targets in examples.csv. "
            "legacy_discounted_return keeps old behavior. "
            "tanh_step_reward_discounted_average adds bounded value_target."
        ),
    )
    parser.add_argument(
        "--value-reward-scale",
        type=str,
        default="auto",
        help=(
            "Reward scale for tanh value target normalization. "
            "Use 'auto' to compute it from generated step_reward values, "
            "or pass a positive number for reproducible fixed scaling."
        ),
    )
    parser.add_argument(
        "--value-reward-scale-quantile",
        type=float,
        default=0.95,
        help="Quantile of abs(step_reward) used when --value-reward-scale auto.",
    )

    args = parser.parse_args(argv)
    raw_dir = Path(args.raw_dir)
    transitions_path = Path(args.transitions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.csv"

    scenario_ids = load_scenario_ids(
        transitions_path=transitions_path,
        limit=args.limit,
    )
    task_config = make_task_config(args)
    physics_config = PhysicsConfig.from_mapping(task_config["physics_config"])
    checkpoint_path = output_dir / "teacher_checkpoint.jsonl"
    checkpoint_config_path = output_dir / "teacher_checkpoint_config.json"
    checkpoint_config = {
        "raw_dir": str(raw_dir.resolve()),
        "transitions_path": str(transitions_path.resolve()),
        "scenario_ids": [int(scenario_id) for scenario_id in scenario_ids],
        "task_config": task_config,
    }
    ensure_checkpoint_config(
        config_path=checkpoint_config_path,
        config=checkpoint_config,
    )
    checkpoint_results = load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=scenario_ids,
    )
    completed_scenario_ids = set(checkpoint_results)
    pending_scenario_ids = [
        int(scenario_id)
        for scenario_id in scenario_ids
        if int(scenario_id) not in completed_scenario_ids
    ]
    scenario_batches = chunk_list(
        values=pending_scenario_ids,
        batch_size=int(args.batch_size),
    )
    if scenario_batches:
        resolved_num_workers = resolve_num_workers(
            num_workers_arg=int(args.num_workers),
            num_batches=len(scenario_batches),
            task_config=task_config,
        )
    else:
        resolved_num_workers = 0

    print("=" * 100)
    print("Generating multi-step impact-beam teacher examples, FAST")
    print("=" * 100)
    print(f"Raw directory:        {raw_dir.resolve()}")
    print(f"Transitions:          {transitions_path.resolve()}")
    print(f"Output dir:           {output_dir}")
    print(f"States dir:           {states_dir}")
    print(f"Examples CSV:         {examples_path}")
    print(f"Checkpoint:           {checkpoint_path}")
    print(f"Restored scenarios:   {len(completed_scenario_ids)}")
    print(f"Pending scenarios:    {len(pending_scenario_ids)}")
    print(f"Scenarios:            {len(scenario_ids)}")
    print(f"Batches:              {len(scenario_batches)}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Depth:                {args.depth}")
    print(f"Beam width:           {args.beam_width}")
    print(f"Candidate pool:       {args.candidate_pool}")
    print(f"Top-K actions:        {args.top_k}")
    print(f"Gamma:                {args.gamma}")
    print(f"PF algorithm:         {args.pf_alg}")
    print(f"PF max iter:          {args.pf_max_iter}")
    print(f"Max teacher steps:    {args.max_teacher_steps}")
    print(f"Soft root policy:     {args.use_soft_root_policy}")
    print(f"Soft policy temp:     {args.soft_policy_temperature}")
    print(f"Min safety improve:   {args.min_safety_improvement}")
    print(f"Continue hard:        {args.min_continue_improvement_with_hard}")
    print(f"Continue no hard:     {args.min_continue_improvement_without_hard}")
    print(f"Max loading increase: {args.max_loading_increase_limit}")
    print(f"Allow hard increase:  {args.allow_hard_count_increase}")
    print(f"Cache enabled:        {not args.disable_cache}")
    print(f"Num workers arg:      {args.num_workers}")
    print(f"Resolved workers:     {resolved_num_workers}")
    print(f"Use LODF screening:   {args.use_lodf_screening}")
    print(f"LODF screen top-k:    {args.lodf_screen_top_k}")
    print(f"LODF min candidates:  {args.lodf_min_candidate_count}")
    print(f"Quiet success:        {args.quiet_success}")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    for required_name in ["bus_data.parquet", "branch_data.parquet", "gen_data.parquet"]:
        required_path = raw_dir / required_name
        if not required_path.exists():
            raise FileNotFoundError(f"Required raw file not found: {required_path}")

    verbose_success = not bool(args.quiet_success)
    if pending_scenario_ids:
        if int(resolved_num_workers) <= 1:
            run_sequential(
                scenario_batches=scenario_batches,
                scenario_ids=pending_scenario_ids,
                raw_dir=raw_dir,
                states_dir=states_dir,
                task_config=task_config,
                checkpoint_path=checkpoint_path,
                verbose_success=verbose_success,
            )
        else:
            run_parallel(
                scenario_batches=scenario_batches,
                scenario_ids=pending_scenario_ids,
                raw_dir=raw_dir,
                states_dir=states_dir,
                task_config=task_config,
                checkpoint_path=checkpoint_path,
                num_workers=int(resolved_num_workers),
                verbose_success=verbose_success,
            )
    else:
        print("\nAll requested scenarios are already present in the checkpoint.")

    checkpoint_results = load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=scenario_ids,
    )
    missing_scenario_ids = sorted(
        set(int(value) for value in scenario_ids) - set(checkpoint_results)
    )
    if missing_scenario_ids:
        raise RuntimeError(
            "Teacher stopped before all scenarios were checkpointed. Missing scenarios: "
            f"{missing_scenario_ids[:20]} (total {len(missing_scenario_ids)}). "
            "Run the same command again to resume."
        )

    rows, total_saved, total_skipped = collect_rows_from_checkpoints(
        checkpoint_results
    )
    if not rows:
        raise RuntimeError("No teacher examples were generated.")

    add_outcome_value_targets_to_rows(
        rows=rows,
        gamma=float(args.gamma),
        group_keys=("scenario_id",),
    )
    print("Outcome value target mode: alphazero_terminal_utility")
    print(f"Outcome gamma:             {args.gamma}")

    if args.value_target_mode == "tanh_step_reward_discounted_average":
        if str(args.value_reward_scale).lower().strip() == "auto":
            value_reward_scale = compute_auto_reward_scale_from_rows(
                rows=rows,
                quantile=float(args.value_reward_scale_quantile),
                min_scale=1.0,
            )
        else:
            value_reward_scale = float(args.value_reward_scale)
            if value_reward_scale <= 0:
                raise ValueError(
                    f"--value-reward-scale must be positive, got {value_reward_scale}"
                )
        add_normalized_value_targets_to_rows(
            rows=rows,
            gamma=float(args.gamma),
            reward_scale=float(value_reward_scale),
            group_keys=("scenario_id",),
        )
        print(f"Value target mode:  {args.value_target_mode}")
        print(f"Value reward scale: {value_reward_scale}")
        print(f"Value gamma:        {args.gamma}")

    examples_df = pd.DataFrame(rows)
    examples_df = examples_df.sort_values(
        ["scenario_id", "step"],
        ascending=[True, True],
    )
    examples_temp_path = examples_path.with_suffix(".csv.tmp")
    examples_df.to_csv(examples_temp_path, index=False)
    examples_temp_path.replace(examples_path)

    print("\n" + "=" * 100)
    print("Fast multi-step impact teacher generation summary")
    print("=" * 100)
    print(f"Saved scenarios: {total_saved}")
    print(f"Skipped:         {total_skipped}")
    print(f"Saved examples:  {len(examples_df)}")
    print(f"Examples CSV:    {examples_path}")
    print(f"States dir:      {states_dir}")
    print("\nStep distribution:")
    print(examples_df.groupby("step").size().to_string())
    print("\nAction 0 / handoff examples:")
    print(int((examples_df["selected_action_id"] == 0).sum()))
    print("\nTermination reasons:")
    print(examples_df["termination_reason"].value_counts(dropna=False).to_string())
    print("\nDone.")
    return 0


_REDISPATCH_ROW_FIELDS = (
    "redispatch_attempted",
    "redispatch_opf_success",
    "redispatch_validated",
    "redispatch_l1_mw",
    "redispatch_up_mw",
    "redispatch_down_mw",
    "redispatch_max_generator_delta_mw",
    "redispatch_message",
)
_SELECTION_PROVENANCE_BY_SCENARIO: dict[int, dict[str, object]] = {}




def _selected_teacher_action_is_valid(
    action_mask: np.ndarray,
    action_id: int,
) -> bool:
    if not _action_is_valid(action_mask, action_id):
        raise RuntimeError(
            "Beam-selected teacher action became invalid during replay: "
            f"action_id={int(action_id)}."
        )
    return True


def _selected_teacher_replay_decision(
    safety_before: float,
    safety_after: float,
    state_before,
    state_after,
    task: dict[str, Any],
) -> tuple[bool, str, float]:
    del state_before, task
    if state_after is None:
        raise RuntimeError(
            "Beam-selected teacher trajectory hit a power-flow failure during replay."
        )
    improvement = float(safety_before) - float(safety_after)
    return True, "selected_by_beam_search", improvement



















def _set_redispatch_diagnostics(
    rows: list[dict[str, Any]],
    result: MinimalRedispatchResult | None,
) -> None:
    diagnostics = empty_redispatch_diagnostics() if result is None else result.diagnostics()
    for row in rows:
        row.update(diagnostics)


def _replay_terminal_evidence(
    scenario_id: int,
    rows: list[dict[str, Any]],
) -> TerminalOutcomeEvidence:
    reasons = {
        parse_termination_reason(row.get("termination_reason"), allow_none=False)
        for row in rows
    }
    if reasons == {TerminationReason.HANDOFF_TO_REDISPATCH}:
        for row in rows:
            row["termination_reason"] = (
                TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value
            )
    ctx = _require_worker_context()
    _set_redispatch_diagnostics(rows, None)
    ordered_rows = sorted(rows, key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered_rows]
    if steps != list(range(len(steps))):
        raise ValueError(
            f"Teacher scenario {scenario_id} has non-contiguous steps: {steps}."
        )

    solved_values = {bool(row["solved"]) for row in ordered_rows}
    reason_values = {
        parse_termination_reason(row.get("termination_reason"), allow_none=False)
        for row in ordered_rows
    }
    if len(solved_values) != 1 or len(reason_values) != 1:
        raise ValueError(
            f"Teacher scenario {scenario_id} contains mixed terminal outcomes."
        )

    solved = solved_values.pop()
    recorded_reason = reason_values.pop()
    assert recorded_reason is not None
    env = TopologySwitchingEnv(
        adapter=ctx["adapter"],
        backend=ctx["backend"],
        action_space=ctx["action_space"],
        reward_fn=ctx["reward_fn"],
        max_steps=int(ctx["task_config"]["max_steps"]),
    )
    env.reset(int(scenario_id))

    for row in ordered_rows:
        action_id = int(row["selected_action_id"])
        if action_id == 0:
            break
        if env.done:
            raise ValueError(
                f"Teacher scenario {scenario_id} contains an action after termination."
            )
        step_result = env.step(action_id)
        if not step_result.power_flow_success:
            raise ValueError(
                f"Teacher scenario {scenario_id} replay hit a power-flow failure."
            )

    if (
        env.done
        and env.terminal_outcome_evidence is not None
        and env.terminal_outcome_evidence.solved is solved
        and env.terminal_outcome_evidence.termination_reason is recorded_reason
    ):
        return env.terminal_outcome_evidence

    final_state = env.current_state
    if final_state is None:
        raise ValueError(
            f"Teacher scenario {scenario_id} has no terminal state for provenance."
        )

    assessment = assess_physical_state(final_state.metrics)
    topology_value = state_utility(final_state, physics_config=ctx["physics_config"])
    reason = recorded_reason
    if (
        reason is TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
        and not assessment.hard_overload_free
    ):
        reason = TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD

    if (
        reason is TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
        and assessment.hard_overload_free
    ):
        redispatch_result = run_minimal_ac_redispatch(ctx["backend"], final_state)
        _set_redispatch_diagnostics(rows, redispatch_result)
        if redispatch_result.validated:
            assert redispatch_result.assessment is not None
            validated_reason = TerminationReason.REDISPATCH_VALIDATED
            return TerminalOutcomeEvidence(
                solved=False,
                termination_reason=validated_reason,
                assessment=assessment,
                redispatch_status=redispatch_status_for_reason(validated_reason),
                topology_utility=topology_value,
                redispatch_assessment=redispatch_result.assessment,
            )

    return TerminalOutcomeEvidence(
        solved=solved,
        termination_reason=reason,
        assessment=assessment,
        redispatch_status=redispatch_status_for_reason(reason),
        topology_utility=topology_value,
    )


def _load_state_file(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    if "metadata_json" not in arrays:
        raise ValueError(f"State file is missing metadata_json: {path}")
    metadata = json.loads(str(np.asarray(arrays["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"State metadata_json must contain an object: {path}")
    return arrays, metadata


def _write_state_metadata(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    arrays["metadata_json"] = np.array(
        json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    temp_path = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(temp_path, **arrays)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _json_feature_columns(row: dict[str, Any]) -> None:
    for field in ("bus_feature_columns", "branch_feature_columns"):
        value = row.get(field)
        if isinstance(value, str):
            continue
        row[field] = json.dumps(value, separators=(",", ":"), allow_nan=False)


def _finalize_success_result(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Successful teacher result must contain rows.")
    scenario_id = int(result["scenario_id"])
    if any(int(row["scenario_id"]) != scenario_id for row in rows):
        raise ValueError(
            f"Teacher result for scenario {scenario_id} contains mixed scenario IDs."
        )

    selection_provenance = _SELECTION_PROVENANCE_BY_SCENARIO.pop(scenario_id, None)
    if selection_provenance is None:
        raise RuntimeError(
            f"Teacher scenario {scenario_id} is missing trajectory selection provenance."
        )

    evidence = _replay_terminal_evidence(scenario_id, rows)
    run_id = _worker_run_id()
    iteration = 1
    episode_id = f"{run_id}_scenario_{scenario_id:06d}"
    reason_value = evidence.termination_reason.value
    evidence_json = evidence.to_json()
    evidence_mapping = evidence.to_dict()
    ctx = _require_worker_context()

    for row in rows:
        state_path = Path(str(row["state_path"]))
        arrays, metadata = _load_state_file(state_path)
        branch_ids = np.asarray(arrays["branch_ids"], dtype=np.int64)
        layout = build_branch_action_slots(branch_ids)
        action_data = {
            "topology_action_config": ctx["action_space"].config.to_contract_dict(),
            "action_layout": action_layout_to_list(layout),
            "action_layout_fingerprint": action_layout_fingerprint(layout),
        }
        row.update(
            {
                "run_id": run_id,
                "iteration": iteration,
                "episode_id": episode_id,
                "solved": evidence.solved,
                "done": True,
                "termination_reason": reason_value,
                "terminal_outcome_evidence_json": evidence_json,
                **selection_provenance,
                **{
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if key != "action_layout_fingerprint" else value
                    for key, value in action_data.items()
                },
            }
        )
        _json_feature_columns(row)
        if int(row["selected_action_id"]) == 0:
            row["step_termination_reason"] = reason_value

        metadata.update(
            {
                "run_id": run_id,
                "iteration": iteration,
                "episode_id": episode_id,
                "episode_done": True,
                "episode_solved": evidence.solved,
                "episode_termination_reason": reason_value,
                "terminal_outcome_evidence": evidence_mapping,
                **selection_provenance,
                **{field: row.get(field) for field in _REDISPATCH_ROW_FIELDS},
                **action_data,
            }
        )
        if int(row["selected_action_id"]) == 0:
            metadata["step_termination_reason"] = reason_value
        _write_state_metadata(state_path, arrays, metadata)
    return result


def process_scenario_batch(
    scenario_ids: list[int],
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        result = process_one_scenario_fast(int(scenario_id))
        scenario_id = int(result["scenario_id"])
        if bool(result.get("ok", False)):
            result = _finalize_success_result(result)
        else:
            _SELECTION_PROVENANCE_BY_SCENARIO.pop(scenario_id, None)
        finalized.append(result)
    return finalized


def process_one_scenario_fast(scenario_id: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = _generate_scenario(int(scenario_id))
    result["runtime_seconds"] = time.perf_counter() - started
    return result




# ======================================================================================
# Redispatch-aware final layer
# ======================================================================================


_TEACHER_SELECTION_MODE = "redispatch_aware_epsilon_minimum_switch"
_TERMINAL_REDISPATCH_RELATIVE_EPSILON = 0.01
_TERMINAL_REDISPATCH_ABSOLUTE_EPSILON_MW = 1.0
_MIN_MEANINGFUL_SAFETY_IMPROVEMENT = 1.0
_TOLERANCE = 1e-9

@dataclass(frozen=True)
class _TerminalCandidate:
    node: Any
    redispatch_l1_mw: float




def _action_key(node: Any) -> tuple[int, ...]:
    return tuple(int(action_id) for action_id in node.action_ids)


def _terminal_candidate_key(candidate: _TerminalCandidate) -> tuple[object, ...]:
    return (
        switch_count(candidate.node),
        float(candidate.redispatch_l1_mw),
        float(candidate.node.safety_score),
        _action_key(candidate.node),
    )


def _same_terminal_objectives(
    left: _TerminalCandidate,
    right: _TerminalCandidate,
) -> bool:
    return (
        switch_count(left.node) == switch_count(right.node)
        and abs(left.redispatch_l1_mw - right.redispatch_l1_mw) <= _TOLERANCE
    )


def _terminal_dominates(
    left: _TerminalCandidate,
    right: _TerminalCandidate,
) -> bool:
    left_switches = switch_count(left.node)
    right_switches = switch_count(right.node)
    left_redispatch = float(left.redispatch_l1_mw)
    right_redispatch = float(right.redispatch_l1_mw)
    no_worse = (
        left_switches <= right_switches
        and left_redispatch <= right_redispatch + _TOLERANCE
    )
    strictly_better = (
        left_switches < right_switches
        or left_redispatch < right_redispatch - _TOLERANCE
    )
    return no_worse and strictly_better


def _terminal_pareto_front(
    candidates: Sequence[_TerminalCandidate],
) -> list[_TerminalCandidate]:
    unique: list[_TerminalCandidate] = []
    for candidate in candidates:
        duplicate_index = next(
            (
                index
                for index, other in enumerate(unique)
                if _same_terminal_objectives(candidate, other)
            ),
            None,
        )
        if duplicate_index is None:
            unique.append(candidate)
            continue
        if _terminal_candidate_key(candidate) < _terminal_candidate_key(
            unique[duplicate_index]
        ):
            unique[duplicate_index] = candidate

    front = [
        candidate
        for candidate in unique
        if not any(
            other is not candidate and _terminal_dominates(other, candidate)
            for other in unique
        )
    ]
    return sorted(front, key=_terminal_candidate_key)


def _with_handoff(node: Any) -> Any:
    if node.action_ids and int(node.action_ids[-1]) == 0:
        return node
    return replace(
        node,
        action_ids=[*node.action_ids, 0],
        branch_ids=[*node.branch_ids, None],
        done=True,
        solved=False,
        termination_reason=TerminationReason.HANDOFF_TO_REDISPATCH,
    )


def _terminal_candidate(node: Any) -> _TerminalCandidate | None:
    state = node.env.current_state
    if state is None:
        return None
    assessment = assess_physical_state(state.metrics)
    if assessment.physically_secure:
        return _TerminalCandidate(node=node, redispatch_l1_mw=0.0)
    if not assessment.hard_overload_free:
        return None
    if node.done:
        reason = parse_termination_reason(node.termination_reason)
        if reason not in {
            TerminationReason.HANDOFF_TO_REDISPATCH,
            TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        }:
            return None

    redispatch_result = run_minimal_ac_redispatch(node.env.backend, state)
    if (
        not redispatch_result.validated
        or redispatch_result.redispatch_l1_mw is None
    ):
        return None
    redispatch_l1_mw = float(redispatch_result.redispatch_l1_mw)
    if not math.isfinite(redispatch_l1_mw) or redispatch_l1_mw < 0.0:
        return None
    return _TerminalCandidate(
        node=_with_handoff(node),
        redispatch_l1_mw=redispatch_l1_mw,
    )


def _root_node(result) -> Any | None:
    roots = [
        node
        for node in result.pareto_front
        if switch_count(node) == 0 and not node.action_ids
    ]
    if not roots:
        return None
    return min(roots, key=lambda node: float(node.safety_score))


def _retained_physical_improvement(
    *,
    root_safety: float,
    best_physical_safety: float,
    selected_safety: float,
) -> float:
    available = max(float(root_safety) - float(best_physical_safety), 0.0)
    if available <= _TOLERANCE:
        return 1.0
    retained = (float(root_safety) - float(selected_safety)) / available
    return float(min(max(retained, 0.0), 1.0))


def _select_terminal_candidate(
    candidates: Sequence[_TerminalCandidate],
    *,
    relative_epsilon: float,
    absolute_epsilon_mw: float,
) -> tuple[_TerminalCandidate, list[_TerminalCandidate], list[_TerminalCandidate]]:
    front = _terminal_pareto_front(candidates)
    if not front:
        raise ValueError("Terminal redispatch selection requires at least one candidate.")
    best_redispatch = min(candidate.redispatch_l1_mw for candidate in front)
    threshold = (
        best_redispatch * (1.0 + float(relative_epsilon))
        + float(absolute_epsilon_mw)
    )
    pool = [
        candidate
        for candidate in front
        if candidate.redispatch_l1_mw <= threshold + _TOLERANCE
    ]
    selected = min(pool, key=_terminal_candidate_key)
    return selected, front, sorted(pool, key=_terminal_candidate_key)


def _redispatch_aware_selection(
    result,
    *,
    task_config: dict[str, Any],
) -> tuple[Any, dict[str, object]]:
    terminal_candidates = [
        candidate
        for node in result.pareto_front
        if (candidate := _terminal_candidate(node)) is not None
    ]
    diagnostics: dict[str, object] = {
        "terminal_redispatch_relative_epsilon": float(
            task_config["terminal_redispatch_relative_epsilon"]
        ),
        "terminal_redispatch_absolute_epsilon_mw": float(
            task_config["terminal_redispatch_absolute_epsilon_mw"]
        ),
        "min_meaningful_safety_improvement": float(
            task_config["min_meaningful_safety_improvement"]
        ),
        "teacher_terminal_selection_applied": False,
        "teacher_terminal_candidate_count": int(len(terminal_candidates)),
        "teacher_terminal_pareto_front_size": 0,
    }
    root = _root_node(result)
    root_safety = (
        float(root.safety_score)
        if root is not None
        else float(result.selected_safety)
    )

    if terminal_candidates:
        selected, terminal_front, terminal_pool = _select_terminal_candidate(
            terminal_candidates,
            relative_epsilon=float(
                task_config["terminal_redispatch_relative_epsilon"]
            ),
            absolute_epsilon_mw=float(
                task_config["terminal_redispatch_absolute_epsilon_mw"]
            ),
        )
        diagnostics["teacher_terminal_selection_applied"] = True
        diagnostics["teacher_terminal_pareto_front_size"] = int(len(terminal_front))
        retained = _retained_physical_improvement(
            root_safety=root_safety,
            best_physical_safety=float(result.best_physical_safety),
            selected_safety=float(selected.node.safety_score),
        )
        updated = replace(
            result,
            best_node=selected.node,
            final_beam=[
                candidate.node
                for candidate in terminal_pool[: result.config.beam_width]
            ],
            selected_safety=float(selected.node.safety_score),
            selected_switch_count=int(switch_count(selected.node)),
            retained_improvement_fraction=retained,
        )
        return updated, diagnostics

    meaningful_improvement = float(root_safety) - float(result.selected_safety)
    minimum = float(task_config["min_meaningful_safety_improvement"])
    if (
        root is not None
        and not bool(result.best_node.solved)
        and meaningful_improvement < minimum
    ):
        result = replace(
            result,
            best_node=root,
            final_beam=[root],
            selected_safety=float(root.safety_score),
            selected_switch_count=0,
            retained_improvement_fraction=0.0,
        )
    return result, diagnostics


def _selection_provenance(
    result,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    return {
        "teacher_selection_mode": _TEACHER_SELECTION_MODE,
        "relative_physical_epsilon": float(result.config.relative_physical_epsilon),
        "teacher_best_physical_safety": float(result.best_physical_safety),
        "teacher_selected_safety": float(result.selected_safety),
        "teacher_selected_switch_count": int(result.selected_switch_count),
        "teacher_retained_improvement_fraction": float(
            result.retained_improvement_fraction
        ),
        "teacher_pareto_front_size": int(len(result.pareto_front)),
        **diagnostics,
    }







# ======================================================================================
# Runtime execution
# ======================================================================================


_RUNTIME_LODF_STRUCTURE_CACHE = "_redispatch_lodf_structure_cache"
_RUNTIME_SCENARIO_STORE_DIR = "_redispatch_runtime_scenario_store_dir"
_RUNTIME_WORKER_INIT_SEMAPHORE = "_redispatch_worker_init_semaphore"



def _native_math_thread_summary() -> str:
    return ", ".join(
        f"{name}={os.environ.get(name, '<unset>')}"
        for name in _NATIVE_MATH_THREAD_ENV_VARS
    )

def _worker_init_concurrency() -> int:
    raw_value = os.environ.get(
        _WORKER_INIT_CONCURRENCY_ENV,
        "1",
    ).strip()

    try:
        concurrency = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{_WORKER_INIT_CONCURRENCY_ENV} must be a positive integer, "
            f"got {raw_value!r}."
        ) from exc

    if concurrency <= 0:
        raise ValueError(
            f"{_WORKER_INIT_CONCURRENCY_ENV} must be >= 1, "
            f"got {concurrency}."
        )

    return concurrency

def _worker_run_id() -> str:
    ctx = _require_worker_context()
    states_dir = Path(ctx["state_store"].output_dir)
    return teacher_run_id(states_dir, ctx["task_config"])


def ensure_checkpoint_config(
    config_path: Path,
    config: dict[str, Any],
) -> None:
    bound_config = dict(config)
    raw_dir = bound_config.get("raw_dir")
    transitions_path = bound_config.get("transitions_path")

    if raw_dir is None or transitions_path is None:
        raise ValueError(
            "Teacher checkpoint config requires raw_dir and transitions_path."
        )

    bound_config["source_identity"] = teacher_source_identity(
        raw_dir,
        transitions_path,
    )
    ensure_teacher_checkpoint_config(config_path, bound_config)















def clear_worker_caches_if_needed() -> None:
    """Bounded caches and process lifetime replace global cache clearing."""

    return None




def init_worker_context(
    raw_dir_str: str,
    states_dir_str: str,
    task_config: dict[str, Any],
    scenario_ids: Sequence[int],
    memory_registry=None,
) -> None:
    del memory_registry

    global _WORKER_CONTEXT

    runtime_task_config = dict(task_config)

    init_semaphore = runtime_task_config.pop(
        _RUNTIME_WORKER_INIT_SEMAPHORE,
        None,
    )
    store_dir = runtime_task_config.pop(
        _RUNTIME_SCENARIO_STORE_DIR,
        None,
    )

    if store_dir is None:
        store_dir = ensure_runtime_scenario_store(
            raw_dir_str
        )

    def build_context() -> dict[str, Any]:
        return build_memory_mapped_teacher_context(
            runtime_store_dir=store_dir,
            states_dir=states_dir_str,
            task_config=runtime_task_config,
            scenario_ids=scenario_ids,
            memory_registry=None,
        )

    if init_semaphore is None:
        _WORKER_CONTEXT = build_context()
    else:
        init_semaphore.acquire()
        try:
            _WORKER_CONTEXT = build_context()
            gc.collect()
        finally:
            init_semaphore.release()

    ctx = _require_worker_context()
    ctx.pop("memory_registry", None)

    ctx[_RUNTIME_LODF_STRUCTURE_CACHE] = (
        None
        if bool(
            runtime_task_config.get(
                "disable_cache",
                False,
            )
        )
        else LODFStructureCache()
    )


def _partition_batches(
    scenario_batches: Sequence[Sequence[int]],
    worker_count: int,
) -> list[list[list[int]]]:
    workers = max(int(worker_count), 1)
    shards: list[list[list[int]]] = [[] for _ in range(workers)]

    for index, batch in enumerate(scenario_batches):
        shards[index % workers].append(
            [int(value) for value in batch]
        )

    return [shard for shard in shards if shard]


def _shard_scenario_ids(
    shard_batches: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(scenario_id)
                for batch in shard_batches
                for scenario_id in batch
            }
        )
    )


def _handle_batch_results(
    batch_results: Sequence[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    checkpoint_path,
    verbose_success: bool,
) -> tuple[int, int]:
    del verbose_success

    saved = 0
    skipped = 0

    for result in batch_results:
        append_scenario_checkpoint(
            checkpoint_path=checkpoint_path,
            result=result,
        )

        if result["ok"]:
            rows.extend(result["rows"])
            saved += 1
        else:
            skipped += 1

    return saved, skipped


def _run_timed_batch(
    process_batch: Callable[[list[int]], list[dict[str, Any]]],
    batch: list[int],
) -> tuple[list[dict[str, Any]], float]:
    """Run one worker batch quietly and return its actual worker wall time."""

    started = time.perf_counter()

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            results = process_batch(batch)

    return results, time.perf_counter() - started




def _scenario_runtime_line(
    result: dict[str, Any],
) -> str:
    scenario_id = int(result["scenario_id"])
    seconds = float(result.get("runtime_seconds", 0.0))

    if bool(result.get("ok", False)):
        status = "saved"
    else:
        reason = result.get("reason")
        status = (
            "skipped"
            if reason is None
            else f"skipped ({reason})"
        )

    return f"scenario {scenario_id} | {seconds:.1f}s | {status}"


def run_parallel(
    scenario_batches: list[list[int]],
    scenario_ids: Sequence[int],
    raw_dir,
    states_dir,
    task_config: dict[str, Any],
    checkpoint_path,
    num_workers: int,
    verbose_success: bool,
):
    if not scenario_batches:
        return [], 0, 0

    store_dir = ensure_runtime_scenario_store(
        Path(raw_dir)
    )

    runtime_task_config = dict(task_config)
    runtime_task_config[
        _RUNTIME_SCENARIO_STORE_DIR
    ] = str(store_dir)

    print(
        f"Memory-mapped runtime store: {store_dir}"
    )

    print(
        "Exact L1 PF cache:          "
        f"{DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES / (1024.0 * 1024.0):.1f} "
        "MiB / worker"
    )

    print(
        "Native math threads:       "
        f"{_native_math_thread_summary()}"
    )

    workers = min(
        max(int(num_workers), 1),
        len(scenario_batches),
    )

    init_concurrency = min(
        _worker_init_concurrency(),
        workers,
    )

    print(
        f"Worker init concurrency: {init_concurrency}"
    )

    if init_concurrency < workers:
        runtime_task_config[
            _RUNTIME_WORKER_INIT_SEMAPHORE
        ] = mp.BoundedSemaphore(
            init_concurrency
        )

    shards = _partition_batches(
        scenario_batches,
        workers,
    )

    workers = len(shards)

    shard_sizes = [
        _shard_scenario_ids(shard)
        for shard in shards
    ]

    print(f"\nParallel sharded mode: {workers} workers")
    print(f"Batches:                {len(scenario_batches)}")

    executors: list[ProcessPoolExecutor] = []
    futures = []
    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0
    completed_batches = 0

    try:
        for shard_batches, shard_scenarios in zip(
            shards,
            shard_sizes,
        ):
            executor = ProcessPoolExecutor(
                max_workers=1,
                initializer=init_worker_context,
                initargs=(
                    str(raw_dir),
                    str(states_dir),
                    runtime_task_config,
                    shard_scenarios,
                    None,
                ),
            )

            executors.append(executor)

            for batch in shard_batches:
                futures.append(
                    executor.submit(
                        _run_timed_batch,
                        process_scenario_batch,
                        batch,
                    )
                )

        progress_bar = None
        iterator = as_completed(futures)

        if tqdm is not None:
            progress_bar = tqdm(
                iterator,
                total=len(futures),
                desc="Teacher batches",
                unit="batch",
                dynamic_ncols=True,
            )
            iterator = progress_bar

        for future in iterator:
            batch_results, batch_seconds = future.result()

            saved, skipped = _handle_batch_results(
                batch_results,
                rows=rows,
                checkpoint_path=checkpoint_path,
                verbose_success=verbose_success,
            )

            total_saved += saved
            total_skipped += skipped
            completed_batches += 1

            for result in batch_results:
                line = _scenario_runtime_line(result)

                if progress_bar is not None:
                    progress_bar.write(line)
                else:
                    print(line, flush=True)

            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "worker": f"{batch_seconds:.1f}s",
                        "saved": saved,
                        "skipped": skipped,
                    },
                    refresh=True,
                )
            else:
                print(
                    f"Teacher batch "
                    f"{completed_batches}/{len(futures)} | "
                    f"worker={batch_seconds:.1f}s | "
                    f"saved={saved} | skipped={skipped}",
                    flush=True,
                )
    finally:
        for executor in executors:
            try:
                executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
            except Exception:
                pass

    return rows, total_saved, total_skipped


if __name__ == "__main__":
    main()

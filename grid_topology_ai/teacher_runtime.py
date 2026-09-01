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
from dataclasses import replace
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

    parsed, remaining = parser.parse_known_args(sys.argv[1:])

    if parsed.exact_cache_max_mb is not None:
        max_mb = float(parsed.exact_cache_max_mb)

        if not math.isfinite(max_mb) or max_mb <= 0.0:
            raise SystemExit("--exact-cache-max-mb must be a positive finite number.")

        os.environ[_EXACT_L1_CACHE_MAX_MB_ENV] = f"{max_mb:.12g}"

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
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.physics.lodf import LODFStructureCache
from grid_topology_ai.physics.objective import (
    RedispatchStatus,
    TerminalOutcomeEvidence,
    assess_physical_state,
)
from grid_topology_ai.physics.redispatch import (
    MinimalRedispatchResult,
    empty_redispatch_diagnostics,
    run_minimal_ac_redispatch,
)
from grid_topology_ai.physics.utility import state_utility
from grid_topology_ai.runtime import (
    build_memory_mapped_teacher_context,
    ensure_runtime_scenario_store,
)
from grid_topology_ai.search.teacher import (
    _TERMINAL_REDISPATCH_ABSOLUTE_EPSILON_MW,
    _TERMINAL_REDISPATCH_RELATIVE_EPSILON,
    _redispatch_aware_selection,
    _safe_short_sequence,
    _selection_provenance,
    ImpactBeamSearchConfig,
    ImpactBeamSearchPlanner,
    LODFScreenedImpactBeamSearchPlanner,
    ensure_teacher_checkpoint_config,
    make_one_hot_policy,
    make_policy_from_final_beam,
    safety_score,
    teacher_run_id,
    teacher_source_identity,
)
from grid_topology_ai.termination import (
    TerminationReason,
    classify_teacher_outcome,
    parse_termination_reason,
    termination_reason_value,
    validate_outcome_invariants,
)
from grid_topology_ai.actions import (
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


def _force_stop_action_valid(action_mask: np.ndarray) -> np.ndarray:
    """Make sure action 0 can be used as a terminal training target."""

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
    return env.action_by_id(int(action_id))


def make_terminal_step_item(
    step_idx: int,
    state_before,
    action_mask: np.ndarray,
    safety_before: float,
    *,
    solved: bool,
    termination_reason: TerminationReason,
    reason: str,
) -> dict[str, Any]:
    """Create an action-0 row for a terminal state with no topology action."""

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
        "solved_after_step": bool(solved),
        "termination_reason_after_step": termination_reason,
        "teacher_decision_reason": reason,
    }


def make_handoff_step_item(
    step_idx: int,
    state_before,
    action_mask: np.ndarray,
    safety_before: float,
    reason: str,
) -> dict[str, Any]:
    """Create a training example for action 0 = handoff to redispatch."""

    return make_terminal_step_item(
        step_idx=step_idx,
        state_before=state_before,
        action_mask=action_mask,
        safety_before=safety_before,
        solved=False,
        termination_reason=TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        reason=reason,
    )


def _initial_redispatch_diagnostics(
    result: MinimalRedispatchResult,
) -> dict[str, object]:
    return {
        "initial_redispatch_attempted": True,
        "initial_redispatch_opf_success": bool(result.opf_success),
        "initial_redispatch_validated": bool(result.validated),
        "initial_redispatch_l1_mw": result.redispatch_l1_mw,
        "initial_redispatch_up_mw": result.redispatch_up_mw,
        "initial_redispatch_down_mw": result.redispatch_down_mw,
        "initial_redispatch_max_generator_delta_mw": (
            result.redispatch_max_generator_delta_mw
        ),
    }


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
        initial_redispatch_result = run_minimal_ac_redispatch(backend, initial_state)
        initial_redispatch = _initial_redispatch_diagnostics(
            initial_redispatch_result
        )

        planner_config = ImpactBeamSearchConfig(
            max_depth=int(task["depth"]),
            beam_width=int(task["beam_width"]),
            candidate_pool_size=int(task["candidate_pool"]),
            top_k_actions=int(task["top_k"]),
            redispatch_candidates_per_switch_count=int(
                task["redispatch_candidates_per_switch_count"]
            ),
            gamma=float(task["gamma"]),
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
            initial_redispatch_result=initial_redispatch_result,
        )
        selected_redispatch_result = selection_diagnostics.pop(
            "_selected_redispatch_result",
            None,
        )
        selection_provenance = _selection_provenance(
            result,
            selection_diagnostics,
        )
        _SELECTION_PROVENANCE_BY_SCENARIO[scenario_id] = selection_provenance

        print(
            f"[worker {os.getpid()}] scenario {scenario_id}: "
            f"beam search done in "
            f"{time.perf_counter() - search_started:.1f}s | "
            f"evaluated={result.evaluated_actions}",
            flush=True,
        )
        best = result.best_node

        terminal_handoff_selected = bool(
            best.action_ids
            and best.branch_ids
            and int(best.action_ids[-1]) == 0
            and best.branch_ids[-1] is None
        )
        topology_action_ids = (
            best.action_ids[:-1] if terminal_handoff_selected else best.action_ids
        )
        topology_branch_ids = (
            best.branch_ids[:-1] if terminal_handoff_selected else best.branch_ids
        )

        final_teacher_safety = float(best.safety_score)
        total_safety_improvement = float(initial_safety - final_teacher_safety)

        if topology_action_ids:
            root_policy_target, root_visit_counts = make_policy_from_final_beam(
                result=result,
                temperature=float(task["soft_policy_temperature"]),
            )
            if not root_policy_target:
                raise RuntimeError(
                    "Teacher topology trajectory has no root policy target."
                )
        else:
            root_policy_target, root_visit_counts = {}, {}

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
            len(topology_action_ids),
            int(task["max_teacher_steps"]),
        )
        handoff_added = False
        handoff_reason: str | None = None
        zero_action_episode_reason: TerminationReason | None = None
        zero_action_episode_solved = False

        for step_idx in range(max_teacher_steps):
            if replay_env.done:
                break

            state_before = replay_env.current_state
            if state_before is None:
                break

            action_mask = replay_env.operational_action_mask()
            selected_action_id = int(topology_action_ids[step_idx])
            selected_branch_id = topology_branch_ids[step_idx]

            _selected_teacher_action_is_valid(action_mask, selected_action_id)

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
                raise RuntimeError(
                    "Beam-selected teacher trajectory hit a power-flow failure during replay."
                )

            safety_after = safety_score(next_state, physics_config=physics_config)
            step_improvement = float(safety_before - safety_after)
            continue_reason = "selected_by_beam_search"

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

        if terminal_handoff_selected and not handoff_added:
            final_teacher_state = replay_env.current_state
            if final_teacher_state is not None:
                final_action_mask = action_space.operational_action_mask(
                    final_teacher_state
                )
                final_safety_before = safety_score(
                    final_teacher_state,
                    physics_config=physics_config,
                )
                final_stop_step = len(step_items)
                terminal_handoff_reason = "terminal_redispatch_selected"
                step_items.append(
                    make_handoff_step_item(
                        step_idx=final_stop_step,
                        state_before=final_teacher_state,
                        action_mask=final_action_mask,
                        safety_before=final_safety_before,
                        reason=terminal_handoff_reason,
                    )
                )
                step_rewards.append(0.0)
                handoff_added = True
                handoff_reason = terminal_handoff_reason

        if not step_items and not topology_action_ids:
            root_state = replay_env.current_state
            if root_state is None:
                raise RuntimeError(
                    "Zero-action teacher trajectory has no root state."
                )
            root_assessment = assess_physical_state(root_state.metrics)
            zero_action_episode_solved = bool(root_assessment.physically_secure)
            zero_action_episode_reason = (
                TerminationReason.SOLVED
                if zero_action_episode_solved
                else TerminationReason.MAX_STEPS_REACHED
            )
            zero_action_decision_reason = (
                "initial_topology_solved"
                if zero_action_episode_solved
                else "terminal_redispatch_unavailable"
            )
            step_items.append(
                make_terminal_step_item(
                    step_idx=0,
                    state_before=root_state,
                    action_mask=action_space.operational_action_mask(root_state),
                    safety_before=safety_score(
                        root_state,
                        physics_config=physics_config,
                    ),
                    solved=zero_action_episode_solved,
                    termination_reason=zero_action_episode_reason,
                    reason=zero_action_decision_reason,
                )
            )
            step_rewards.append(0.0)

        if not step_items:
            raise RuntimeError(
                "Selected teacher trajectory produced no replay steps."
            )

        returns = discounted_returns(
            rewards=step_rewards,
            gamma=float(task["gamma"]),
        )
        final_state = replay_env.current_state

        if final_state is None:
            raise RuntimeError(
                "Selected teacher trajectory has no final replay state."
            )

        final_safety = safety_score(final_state, physics_config=physics_config)
        final_max_loading = float(final_state.metrics["max_loading_percent"])
        final_num_hard = int(final_state.metrics["num_hard_overloaded_branches"])
        final_num_overloaded = int(final_state.metrics["num_overloaded_branches"])

        if zero_action_episode_reason is not None:
            episode_done = True
            episode_solved = bool(zero_action_episode_solved)
            episode_reason = zero_action_episode_reason
        elif handoff_added:
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

        terminal_evidence, terminal_redispatch_result = _terminal_evidence_from_state(
            final_state,
            recorded_solved=episode_solved,
            recorded_reason=episode_reason,
            redispatch_result=selected_redispatch_result,
        )
        redispatch_diagnostics = (
            empty_redispatch_diagnostics()
            if terminal_redispatch_result is None
            else terminal_redispatch_result.diagnostics()
        )
        teacher_outcome_value = classify_teacher_outcome(
            topology_solved=terminal_evidence.solved,
            redispatch_validated=bool(redispatch_diagnostics["redispatch_validated"]),
        ).value
        run_id = _worker_run_id()
        iteration = 1
        episode_id = f"{run_id}_scenario_{scenario_id:06d}"
        diagnostic_reason_value = terminal_evidence.termination_reason.value
        evidence_mapping = terminal_evidence.to_dict()

        rows: list[dict[str, Any]] = []
        final_return = float(returns[0]) if returns else float(total_safety_improvement)

        for item, return_from_step in zip(step_items, returns):
            step_idx = int(item["step"])
            state = item["state"]
            state_id = f"impact_teacher_scenario_{scenario_id:06d}_step_{step_idx:03d}"
            layout = build_branch_action_slots(
                np.asarray(state.branch_ids, dtype=np.int64)
            )
            action_data = {
                "topology_action_config": action_space.config.to_contract_dict(),
                "action_layout": action_layout_to_list(layout),
                "action_layout_fingerprint": action_layout_fingerprint(layout),
            }
            step_diagnostic_reason = termination_reason_value(
                parse_termination_reason(item.get("termination_reason_after_step"))
            )
            if int(item["selected_action_id"]) == 0:
                step_diagnostic_reason = diagnostic_reason_value

            state_metadata = {
                "physics_config": physics_config.to_dict(),
                "source": "impact_beam_teacher_multistep_fast",
                "scenario_id": int(scenario_id),
                "step": int(step_idx),
                "initial_safety": float(initial_safety),
                **initial_redispatch,
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
                "episode_done": True,
                "episode_solved": terminal_evidence.solved,
                "episode_teacher_outcome": teacher_outcome_value,
                "episode_diagnostic_termination_reason": diagnostic_reason_value,
                "terminal_outcome_evidence": evidence_mapping,
                **selection_provenance,
                **redispatch_diagnostics,
                **action_data,
                "step_done": bool(item.get("done_after_step", False)),
                "step_solved": bool(item.get("solved_after_step", False)),
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
                "run_id": run_id,
                "iteration": iteration,
                "episode_id": episode_id,
            }
            if step_diagnostic_reason is not None:
                state_metadata["step_diagnostic_termination_reason"] = (
                    step_diagnostic_reason
                )

            state_path = state_store.save_state(
                state=state,
                state_id=state_id,
                action_mask=item["action_mask"],
                extra_metadata=state_metadata,
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
                        parse_termination_reason(
                            item.get("termination_reason_after_step")
                        )
                    ),
                    **initial_redispatch,
                    "physics_config": json.dumps(
                        physics_config.to_dict(), sort_keys=True, separators=(",", ":")
                    ),
                    "visit_counts_json": json.dumps(
                        {str(k): int(v) for k, v in item["visit_counts"].items()}
                    ),
                    "mcts_policy_json": json.dumps(
                        {str(k): float(v) for k, v in item["policy_target"].items()}
                    ),
                    **{
                        key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if key != "action_layout_fingerprint"
                        else value
                        for key, value in action_data.items()
                    },
                }
            )

        _set_redispatch_diagnostics(rows, terminal_redispatch_result)
        clear_worker_caches_if_needed()
        first_action = int(best.action_ids[0]) if best.action_ids else 0
        first_branch = (
            None
            if not best.branch_ids or best.branch_ids[0] is None
            else int(best.branch_ids[0])
        )
        return {
            "ok": True,
            "scenario_id": int(scenario_id),
            "rows": rows,
            "_terminal_outcome_evidence": terminal_evidence,
            "summary": {
                "num_examples": int(len(rows)),
                "first_action": first_action,
                "first_branch": first_branch,
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
    difficulty_class: str | None = None,
) -> list[int]:
    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")

    transitions = pd.read_csv(transitions_path)
    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions file must contain scenario_id column: {transitions_path}"
        )

    if difficulty_class is not None:
        if "difficulty_class" not in transitions.columns:
            raise ValueError(
                "Transitions file must contain difficulty_class column when "
                f"--difficulty-class is used: {transitions_path}"
            )
        normalized = (
            transitions["difficulty_class"].astype(str).str.strip().str.lower()
        )
        requested = str(difficulty_class).strip().lower()
        transitions = transitions.loc[normalized == requested]
        if transitions.empty:
            raise ValueError(
                f"No {requested!r} scenarios found in transitions file: "
                f"{transitions_path}"
            )

    scenario_ids = [
        int(x) for x in transitions["scenario_id"].drop_duplicates().tolist()
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
        f"Scenario {result['scenario_id']}: skipped | reason={result['reason']}"
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
    redispatch_candidates_per_switch_count = int(
        args.redispatch_candidates_per_switch_count
    )
    if depth <= 0 or max_teacher_steps <= 0:
        raise ValueError("Teacher depth and max_teacher_steps must be positive.")
    if redispatch_candidates_per_switch_count <= 0:
        raise ValueError(
            "redispatch_candidates_per_switch_count must be positive."
        )
    return {
        "depth": min(depth, max_teacher_steps),
        "beam_width": int(args.beam_width),
        "candidate_pool": int(args.candidate_pool),
        "top_k": int(args.top_k),
        "redispatch_candidates_per_switch_count": (
            redispatch_candidates_per_switch_count
        ),
        "gamma": float(args.gamma),
        "pf_alg": int(args.pf_alg),
        "pf_max_iter": physics_config.max_iterations,
        "physics_config": physics_config.to_dict(),
        "max_steps": int(args.max_steps),
        "max_teacher_steps": max_teacher_steps,
        "soft_policy_temperature": float(args.soft_policy_temperature),
        "use_soft_root_policy": bool(args.use_soft_root_policy),
        "allow_hard_count_increase": bool(args.allow_hard_count_increase),
        "disable_cache": bool(args.disable_cache),
        "use_lodf_screening": bool(args.use_lodf_screening),
        "lodf_screen_top_k": int(args.lodf_screen_top_k),
        "lodf_min_candidate_count": int(args.lodf_min_candidate_count),
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

    with checkpoint_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as checkpoint_file:
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
        "--difficulty-class",
        choices=["simple", "medium", "hard"],
        default=None,
        help="Restrict teacher generation to one difficulty_class in the transitions CSV.",
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
        "--redispatch-candidates-per-switch-count",
        type=int,
        default=5,
        help=(
            "Keep this many lowest-J topology candidates for each switch count "
            "before terminal redispatch."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Terminal-utility gamma. The current contract requires 1.0.",
    )
    parser.add_argument("--pf-alg", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--pf-max-iter", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-teacher-steps", type=int, default=4)
    parser.add_argument("--soft-policy-temperature", type=float, default=0.0)
    parser.add_argument("--use-soft-root-policy", action="store_true")
    parser.add_argument("--allow-hard-count-increase", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
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
        difficulty_class=args.difficulty_class,
    )
    task_config = make_task_config(args)
    checkpoint_path = output_dir / "teacher_checkpoint.jsonl"
    checkpoint_config_path = output_dir / "teacher_checkpoint_config.json"
    checkpoint_config = {
        "raw_dir": str(raw_dir.resolve()),
        "transitions_path": str(transitions_path.resolve()),
        "difficulty_class": args.difficulty_class,
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
    print(f"Difficulty class:     {args.difficulty_class or 'all'}")
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
    print(
        "Redispatch candidates/switch count: "
        f"{args.redispatch_candidates_per_switch_count}"
    )
    print(f"Gamma:                {args.gamma}")
    print(f"PF algorithm:         {args.pf_alg}")
    print(f"PF max iter:          {args.pf_max_iter}")
    print(f"Max teacher steps:    {args.max_teacher_steps}")
    print(f"Soft root policy:     {args.use_soft_root_policy}")
    print(f"Soft policy temp:     {args.soft_policy_temperature}")
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
    for required_name in [
        "bus_data.parquet",
        "branch_data.parquet",
        "gen_data.parquet",
    ]:
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

    rows, total_saved, total_skipped = collect_rows_from_checkpoints(checkpoint_results)
    if not rows:
        raise RuntimeError("No teacher examples were generated.")

    add_outcome_value_targets_to_rows(
        rows=rows,
        gamma=float(args.gamma),
        group_keys=("scenario_id",),
    )
    print("Outcome value target mode: alphazero_terminal_utility")
    print(f"Outcome gamma:             {args.gamma}")

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
    print("\nTeacher outcomes:")
    print(examples_df["teacher_outcome"].value_counts(dropna=False).to_string())
    print("\nDone.")
    return 0


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


def _set_redispatch_diagnostics(
    rows: list[dict[str, Any]],
    result: MinimalRedispatchResult | None,
) -> None:
    diagnostics = (
        empty_redispatch_diagnostics() if result is None else result.diagnostics()
    )
    for row in rows:
        row.update(diagnostics)


def _terminal_evidence_from_state(
    final_state,
    *,
    recorded_solved: bool,
    recorded_reason: TerminationReason,
    redispatch_result: MinimalRedispatchResult | None = None,
) -> tuple[TerminalOutcomeEvidence, MinimalRedispatchResult | None]:
    ctx = _require_worker_context()
    assessment = assess_physical_state(final_state.metrics)
    topology_value = state_utility(
        final_state,
        physics_config=ctx["physics_config"],
    )

    if assessment.physically_secure:
        return (
            TerminalOutcomeEvidence(
                solved=True,
                termination_reason=TerminationReason.SOLVED,
                assessment=assessment,
                redispatch_status=RedispatchStatus.NOT_REQUESTED,
                topology_utility=topology_value,
            ),
            None,
        )

    if redispatch_result is None:
        redispatch_result = run_minimal_ac_redispatch(
            ctx["backend"],
            final_state,
        )

    if redispatch_result.validated:
        assert redispatch_result.assessment is not None
        return (
            TerminalOutcomeEvidence(
                solved=False,
                termination_reason=TerminationReason.REDISPATCH_VALIDATED,
                assessment=assessment,
                redispatch_status=RedispatchStatus.VALIDATED,
                topology_utility=topology_value,
                redispatch_assessment=redispatch_result.assessment,
            ),
            redispatch_result,
        )

    if recorded_solved:
        raise ValueError(
            "Teacher trajectory was recorded solved but its final state is insecure."
        )

    return (
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=recorded_reason,
            assessment=assessment,
            redispatch_status=RedispatchStatus.REQUESTED,
            topology_utility=topology_value,
        ),
        redispatch_result,
    )


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

    evidence = result.pop("_terminal_outcome_evidence", None)
    if not isinstance(evidence, TerminalOutcomeEvidence):
        raise RuntimeError(
            f"Teacher scenario {scenario_id} is missing terminal outcome evidence."
        )

    redispatch_validated = bool(rows[0].get("redispatch_validated", False))
    teacher_outcome = classify_teacher_outcome(
        topology_solved=evidence.solved,
        redispatch_validated=redispatch_validated,
    )
    teacher_outcome_value = teacher_outcome.value

    run_id = _worker_run_id()
    iteration = 1
    episode_id = f"{run_id}_scenario_{scenario_id:06d}"
    diagnostic_reason_value = evidence.termination_reason.value
    evidence_json = evidence.to_json()

    for row in rows:
        step_diagnostic_reason = row.pop("step_termination_reason", None)
        if int(row["selected_action_id"]) == 0:
            step_diagnostic_reason = diagnostic_reason_value

        row.update(
            {
                "run_id": run_id,
                "iteration": iteration,
                "episode_id": episode_id,
                "solved": evidence.solved,
                "done": True,
                "teacher_outcome": teacher_outcome_value,
                "diagnostic_termination_reason": diagnostic_reason_value,
                "terminal_outcome_evidence_json": evidence_json,
                **selection_provenance,
            }
        )
        row.pop("termination_reason", None)
        if step_diagnostic_reason is not None:
            row["step_diagnostic_termination_reason"] = step_diagnostic_reason
        _json_feature_columns(row)
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
            f"{_WORKER_INIT_CONCURRENCY_ENV} must be >= 1, got {concurrency}."
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
        store_dir = ensure_runtime_scenario_store(raw_dir_str)

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
        shards[index % workers].append([int(value) for value in batch])

    return [shard for shard in shards if shard]


def _shard_scenario_ids(
    shard_batches: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    return tuple(
        sorted({int(scenario_id) for batch in shard_batches for scenario_id in batch})
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
        status = "skipped" if reason is None else f"skipped ({reason})"

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

    store_dir = ensure_runtime_scenario_store(Path(raw_dir))

    runtime_task_config = dict(task_config)
    runtime_task_config[_RUNTIME_SCENARIO_STORE_DIR] = str(store_dir)

    print(f"Memory-mapped runtime store: {store_dir}")

    print(
        "Exact L1 PF cache:          "
        f"{DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES / (1024.0 * 1024.0):.1f} "
        "MiB / worker"
    )

    print(f"Native math threads:       {_native_math_thread_summary()}")

    workers = min(
        max(int(num_workers), 1),
        len(scenario_batches),
    )

    init_concurrency = min(
        _worker_init_concurrency(),
        workers,
    )

    print(f"Worker init concurrency: {init_concurrency}")

    if init_concurrency < workers:
        runtime_task_config[_RUNTIME_WORKER_INIT_SEMAPHORE] = mp.BoundedSemaphore(
            init_concurrency
        )

    shards = _partition_batches(
        scenario_batches,
        workers,
    )

    workers = len(shards)

    shard_sizes = [_shard_scenario_ids(shard) for shard in shards]

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

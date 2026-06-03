from __future__ import annotations

import argparse
import json
import traceback
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.continuation_gate import make_do_nothing_action
from grid_topology_ai.search.impact_beam_search import (
    ImpactBeamSearchConfig,
    ImpactBeamSearchPlanner,
    ImpactBeamSearchResult,
    safety_score,
)
from grid_topology_ai.state_store import GridFMStateStore


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

def get_process_memory_mb() -> float | None:
    """
    Return current process RSS memory in MB.

    psutil is optional. If it is not installed, return None.
    """

    try:
        import psutil
    except Exception:
        return None

    try:
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def clear_worker_caches(reason: str = "manual") -> None:
    """
    Clear worker-local caches and force Python garbage collection.
    """

    ctx = _require_worker_context()

    backend = ctx.get("backend")
    action_space = ctx.get("action_space")

    memory_before = get_process_memory_mb()

    if hasattr(backend, "clear_cache"):
        backend.clear_cache()

    if hasattr(action_space, "clear_cache"):
        action_space.clear_cache()

    gc.collect()

    memory_after = get_process_memory_mb()

    if bool(ctx["task_config"].get("print_memory_events", False)):
        before_text = "unknown" if memory_before is None else f"{memory_before:.1f} MB"
        after_text = "unknown" if memory_after is None else f"{memory_after:.1f} MB"

        print(
            f"[worker {os.getpid()}] cache clear ({reason}) | "
            f"memory {before_text} -> {after_text}",
            flush=True,
        )

def init_worker_context(
    raw_dir_str: str,
    states_dir_str: str,
    task_config: dict[str, Any],
) -> None:
    """
    Initialize heavy objects once per worker process.

    This is the main speed optimization.

    Old slow behavior:
        every scenario -> GridFMAdapter(raw_dir) -> read parquet again

    New fast behavior:
        every worker -> GridFMAdapter(raw_dir) once
        worker then processes many scenarios using the same adapter/backend/action_space
    """

    global _WORKER_CONTEXT

    raw_dir = Path(raw_dir_str)
    states_dir = Path(states_dir_str)

    adapter = GridFMAdapter(raw_dir)

    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=int(task_config["pf_alg"]),
        max_iter=int(task_config["pf_max_iter"]),
        enable_cache=not bool(task_config["disable_cache"]),
    )

    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=not bool(task_config["disable_cache"]),
    )

    reward_fn = GridFMReward()
    state_store = GridFMStateStore(states_dir)

    _WORKER_CONTEXT = {
        "adapter": adapter,
        "backend": backend,
        "action_space": action_space,
        "reward_fn": reward_fn,
        "state_store": state_store,
        "task_config": task_config,
        "processed_in_worker": 0,
    }


def clear_worker_caches_if_needed() -> None:
    """
    Prevent long-running workers from accumulating too much cache memory.

    Two mechanisms:
    1. periodic cache clearing every N scenarios;
    2. memory threshold guard using current process RSS.
    """

    ctx = _require_worker_context()
    cfg = ctx["task_config"]

    ctx["processed_in_worker"] = int(ctx.get("processed_in_worker", 0)) + 1
    processed = int(ctx["processed_in_worker"])

    every = int(cfg["clear_caches_every"])

    should_clear_periodically = every > 0 and processed % every == 0

    memory_mb = get_process_memory_mb()
    max_memory_mb = float(cfg.get("max_worker_memory_mb", 0.0))

    should_clear_by_memory = (
        memory_mb is not None
        and max_memory_mb > 0.0
        and memory_mb >= max_memory_mb
    )

    if should_clear_by_memory:
        clear_worker_caches(
            reason=f"memory_guard_{memory_mb:.1f}_mb_ge_{max_memory_mb:.1f}_mb"
        )
        return

    if should_clear_periodically:
        clear_worker_caches(reason=f"periodic_every_{every}")


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
        "termination_reason_after_step": "handoff_to_redispatch_teacher",
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
# Scenario processing
# ======================================================================================


def process_one_scenario_fast(scenario_id: int) -> dict[str, Any]:
    """
    Process one scenario using worker-global adapter/backend/action_space.

    This function does NOT create GridFMAdapter.
    That is the main difference from generate_impact_teacher_parallel.py.
    """

    ctx = _require_worker_context()

    adapter = ctx["adapter"]
    backend = ctx["backend"]
    action_space = ctx["action_space"]
    reward_fn = ctx["reward_fn"]
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
        initial_safety = safety_score(initial_state)

        planner_config = ImpactBeamSearchConfig(
            max_depth=int(task["depth"]),
            beam_width=int(task["beam_width"]),
            candidate_pool_size=int(task["candidate_pool"]),
            top_k_actions=int(task["top_k"]),
            gamma=float(task["gamma"]),
            include_stop_action=True,
            allow_hard_count_increase=bool(task["allow_hard_count_increase"]),
            show_progress=False,
            progress_update_every=1,
        )

        planner = ImpactBeamSearchPlanner(planner_config)

        result = planner.search(
            env=search_env,
            scenario_id=scenario_id,
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

            action_mask = replay_env.valid_action_mask()

            selected_action_id = int(best.action_ids[step_idx])
            selected_branch_id = best.branch_ids[step_idx]

            if not _action_is_valid(action_mask, selected_action_id):
                if bool(task["add_handoff_example"]):
                    safety_before = safety_score(state_before)

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

            safety_before = safety_score(state_before)

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
                safety_after = safety_score(next_state)

            continue_action, continue_reason, step_improvement = (
                should_continue_teacher_action(
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

        # Add final handoff after useful sequence.
        if (
            bool(task["add_handoff_example"])
            and not handoff_added
            and not replay_env.done
        ):
            final_teacher_state = replay_env.current_state

            if final_teacher_state is not None:
                final_action_mask = replay_env.valid_action_mask()
                final_safety_before = safety_score(final_teacher_state)

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
            final_safety = safety_score(final_state)
            final_max_loading = float(final_state.metrics["max_loading_percent"])
            final_num_hard = int(final_state.metrics["num_hard_overloaded_branches"])
            final_num_overloaded = int(final_state.metrics["num_overloaded_branches"])

        rows: list[dict[str, Any]] = []

        final_return = float(returns[0]) if returns else float(total_safety_improvement)

        for item, return_from_step in zip(step_items, returns):
            step_idx = int(item["step"])

            state_id = (
                f"impact_teacher_scenario_{scenario_id:06d}_step_{step_idx:03d}"
            )

            state_path = state_store.save_state(
                state=item["state"],
                state_id=state_id,
                action_mask=item["action_mask"],
                extra_metadata={
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
                    "best_sequence_action_ids": [int(x) for x in best.action_ids],
                    "best_sequence_branch_ids": [
                        None if x is None else int(x)
                        for x in best.branch_ids
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
                    "solved": bool(item.get("solved_after_step", replay_env.solved)),
                    "done": bool(item.get("done_after_step", replay_env.done)),
                    "termination_reason": (
                        item.get("termination_reason_after_step")
                        or (
                            "handoff_to_redispatch_teacher"
                            if handoff_added
                            else (replay_env.termination_reason or "teacher_depth_limit")
                        )
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


def process_scenario_batch(scenario_ids: list[int]) -> list[dict[str, Any]]:
    """
    Process a batch of scenarios inside one worker.

    Batch processing reduces ProcessPool overhead.
    """

    results: list[dict[str, Any]] = []

    for scenario_id in scenario_ids:
        results.append(process_one_scenario_fast(int(scenario_id)))

    return results


# ======================================================================================
# IO / CLI helpers
# ======================================================================================


def load_scenario_ids(
    transitions_path: Path,
    limit: int | None,
) -> list[int]:
    """
    Read scenario IDs from transitions CSV, preserving file order.
    """

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


def chunk_list(
    values: list[int],
    batch_size: int,
) -> list[list[int]]:
    batch_size = max(int(batch_size), 1)

    return [
        values[i : i + batch_size]
        for i in range(0, len(values), batch_size)
    ]


def print_success(result: dict[str, Any]) -> None:
    summary = result["summary"]

    print(
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
    print(
        f"Scenario {result['scenario_id']}: skipped | "
        f"reason={result['reason']}"
    )

    if result.get("traceback"):
        print(result["traceback"])


def make_task_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "depth": int(args.depth),
        "beam_width": int(args.beam_width),
        "candidate_pool": int(args.candidate_pool),
        "top_k": int(args.top_k),
        "gamma": float(args.gamma),
        "pf_alg": int(args.pf_alg),
        "pf_max_iter": int(args.pf_max_iter),
        "max_steps": int(args.max_steps),
        "max_teacher_steps": int(args.max_teacher_steps),
        "soft_policy_temperature": float(args.soft_policy_temperature),
        "use_soft_root_policy": bool(args.use_soft_root_policy),
        "min_safety_improvement": float(args.min_safety_improvement),
        "allow_hard_count_increase": bool(args.allow_hard_count_increase),
        "disable_cache": bool(args.disable_cache),
        "clear_caches_every": int(args.clear_caches_every),
        "max_worker_memory_mb": float(args.max_worker_memory_mb),
        "print_memory_events": bool(args.print_memory_events),
        "power_flow_failure_penalty": float(args.power_flow_failure_penalty),
        "min_continue_improvement_with_hard": float(
            args.min_continue_improvement_with_hard
        ),
        "min_continue_improvement_without_hard": float(
            args.min_continue_improvement_without_hard
        ),
        "max_loading_increase_limit": float(args.max_loading_increase_limit),
        "add_handoff_example": bool(args.add_handoff_example),
        "max_tasks_per_child": int(args.max_tasks_per_child),
    }


def run_sequential(
    scenario_batches: list[list[int]],
    raw_dir: Path,
    states_dir: Path,
    task_config: dict[str, Any],
    verbose_success: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Sequential mode with one persistent context in the main process.
    Useful for lowest RAM usage and debugging.
    """

    init_worker_context(
        raw_dir_str=str(raw_dir),
        states_dir_str=str(states_dir),
        task_config=task_config,
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
            if result["ok"]:
                rows.extend(result["rows"])
                total_saved += 1

                if verbose_success:
                    print_success(result)
            else:
                total_skipped += 1
                print_failure(result)

    return rows, total_saved, total_skipped


def run_parallel(
    scenario_batches: list[list[int]],
    raw_dir: Path,
    states_dir: Path,
    task_config: dict[str, Any],
    num_workers: int,
    verbose_success: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Parallel mode with persistent per-worker contexts.
    """

    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0

    print(f"\nParallel fast mode: {num_workers} workers")
    print(f"Batches:            {len(scenario_batches)}")

    executor_kwargs = {
        "max_workers": int(num_workers),
        "initializer": init_worker_context,
        "initargs": (str(raw_dir), str(states_dir), task_config),
    }

    max_tasks_per_child = int(task_config.get("max_tasks_per_child", 0))

    if max_tasks_per_child > 0:
        executor_kwargs["max_tasks_per_child"] = max_tasks_per_child

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = [
            executor.submit(process_scenario_batch, batch)
            for batch in scenario_batches
        ]

        iterator = as_completed(futures)

        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="Teacher batches",
                unit="batch",
                dynamic_ncols=True,
            )

        for future in iterator:
            batch_results = future.result()

            for result in batch_results:
                if result["ok"]:
                    rows.extend(result["rows"])
                    total_saved += 1

                    if verbose_success:
                        print_success(result)
                else:
                    total_skipped += 1
                    print_failure(result)

    return rows, total_saved, total_skipped


# ======================================================================================
# Main
# ======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast multi-step teacher generation using persistent worker contexts."
        )
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw directory.",
    )

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
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--pf-alg", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--pf-max-iter", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=5)

    parser.add_argument(
        "--max-teacher-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--soft-policy-temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--use-soft-root-policy",
        action="store_true",
    )

    parser.add_argument(
        "--min-safety-improvement",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--allow-hard-count-increase",
        action="store_true",
    )

    parser.add_argument(
        "--disable-cache",
        action="store_true",
    )

    parser.add_argument(
        "--clear-caches-every",
        type=int,
        default=50,
        help=(
            "Clear backend/action-space caches after this many scenarios per worker. "
            "Use 0 to never clear caches."
        ),
    )

    parser.add_argument(
        "--power-flow-failure-penalty",
        type=float,
        default=1_000_000.0,
    )

    parser.add_argument(
        "--min-continue-improvement-with-hard",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--min-continue-improvement-without-hard",
        type=float,
        default=150.0,
    )

    parser.add_argument(
        "--max-loading-increase-limit",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--add-handoff-example",
        action="store_true",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of scenarios per submitted worker task.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--quiet-success",
        action="store_true",
        help="Do not print one line for every successful scenario.",
    )

    parser.add_argument(
        "--max-worker-memory-mb",
        type=float,
        default=0.0,
        help=(
            "If > 0, worker clears backend/action-space caches when its RSS memory "
            "reaches this value in MB."
        ),
    )

    parser.add_argument(
        "--print-memory-events",
        action="store_true",
        help="Print memory before/after cache clearing events.",
    )

    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=0,
        help=(
            "Restart each worker process after this many submitted batches. "
            "Use 0 to disable. This is the strongest protection against memory leaks."
        ),
    )

    args = parser.parse_args()

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

    scenario_batches = chunk_list(
        values=scenario_ids,
        batch_size=int(args.batch_size),
    )

    task_config = make_task_config(args)

    print("=" * 100)
    print("Generating multi-step impact-beam teacher examples, FAST")
    print("=" * 100)
    print(f"Raw directory:        {raw_dir.resolve()}")
    print(f"Transitions:          {transitions_path.resolve()}")
    print(f"Output dir:           {output_dir}")
    print(f"States dir:           {states_dir}")
    print(f"Examples CSV:         {examples_path}")
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
    print(f"Clear caches every:   {args.clear_caches_every}")
    print(f"Max worker memory MB: {args.max_worker_memory_mb}")
    print(f"Print memory events:  {args.print_memory_events}")
    print(f"Max tasks per child:  {args.max_tasks_per_child}")
    print(f"Num workers:          {args.num_workers}")
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

    if int(args.num_workers) <= 1:
        rows, total_saved, total_skipped = run_sequential(
            scenario_batches=scenario_batches,
            raw_dir=raw_dir,
            states_dir=states_dir,
            task_config=task_config,
            verbose_success=verbose_success,
        )
    else:
        rows, total_saved, total_skipped = run_parallel(
            scenario_batches=scenario_batches,
            raw_dir=raw_dir,
            states_dir=states_dir,
            task_config=task_config,
            num_workers=int(args.num_workers),
            verbose_success=verbose_success,
        )

    if not rows:
        raise RuntimeError("No teacher examples were generated.")

    examples_df = pd.DataFrame(rows)
    examples_df = examples_df.sort_values(
        ["scenario_id", "step"],
        ascending=[True, True],
    )

    examples_df.to_csv(examples_path, index=False)

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


if __name__ == "__main__":
    main()
from __future__ import annotations

import math
import argparse
import gc
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grid_topology_ai.action_space import GridFMAction, GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter
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


# ======================================================================================
# Memory helpers
# ======================================================================================


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


def get_system_available_memory_mb() -> float | None:
    """
    Return available system RAM in MB.

    psutil is optional. If unavailable, return None.
    """

    try:
        import psutil
    except Exception:
        return None

    try:
        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        return None


def get_cpu_load_percent() -> float | None:
    """
    Return current total CPU load percent.

    psutil is optional. If unavailable, return None.
    """

    try:
        import psutil
    except Exception:
        return None

    try:
        return float(psutil.cpu_percent(interval=0.2))
    except Exception:
        return None


def update_worker_memory_registry() -> None:
    """
    Update shared memory registry for the current worker.
    """

    ctx = _require_worker_context()
    registry = ctx.get("memory_registry")

    if registry is None:
        return

    memory_mb = get_process_memory_mb()

    if memory_mb is None:
        return

    pid = int(os.getpid())

    try:
        registry[pid] = {
            "rss_mb": float(memory_mb),
            "timestamp": float(time.time()),
        }
    except Exception:
        return


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


def maybe_clear_heaviest_worker_for_global_memory() -> None:
    """
    Cooperative global memory guard.

    If system free memory is below a configured threshold, the currently running
    worker checks whether it is the heaviest registered worker. If yes, it clears
    its own local caches.

    ProcessPoolExecutor does not provide a reliable way to directly command a
    different running child process to clean its memory. This cooperative guard is
    therefore checked after every processed scenario.
    """

    ctx = _require_worker_context()
    cfg = ctx["task_config"]

    min_free_mb = float(cfg.get("min_free_system_memory_mb", 0.0))

    if min_free_mb <= 0.0:
        return

    available_mb = get_system_available_memory_mb()

    if available_mb is None:
        return

    update_worker_memory_registry()

    if available_mb >= min_free_mb:
        return

    registry = ctx.get("memory_registry")

    if registry is None:
        return

    now = time.time()
    max_age_sec = float(cfg.get("memory_registry_max_age_sec", 120.0))

    heaviest_pid: int | None = None
    heaviest_mb = -1.0

    try:
        for pid_raw, info in list(registry.items()):
            pid = int(pid_raw)
            rss_mb = float(info.get("rss_mb", 0.0))
            timestamp = float(info.get("timestamp", 0.0))

            if now - timestamp > max_age_sec:
                continue

            if rss_mb > heaviest_mb:
                heaviest_mb = rss_mb
                heaviest_pid = pid
    except Exception:
        return

    current_pid = int(os.getpid())

    if heaviest_pid == current_pid:
        clear_worker_caches(
            reason=(
                f"global_memory_low_available_{available_mb:.1f}_mb_"
                f"lt_{min_free_mb:.1f}_mb_heaviest_{heaviest_mb:.1f}_mb"
            )
        )
        update_worker_memory_registry()


# ======================================================================================
# Worker initialization
# ======================================================================================


def init_worker_context(
    raw_dir_str: str,
    states_dir_str: str,
    task_config: dict[str, Any],
    memory_registry=None,
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
        "memory_registry": memory_registry,
    }

    update_worker_memory_registry()


def clear_worker_caches_if_needed() -> None:
    """
    Prevent long-running workers from accumulating too much cache memory.

    Mechanisms:
    1. periodic cache clearing every N scenarios;
    2. per-worker RSS threshold guard;
    3. cooperative global system-memory guard.
    """

    ctx = _require_worker_context()
    cfg = ctx["task_config"]

    ctx["processed_in_worker"] = int(ctx.get("processed_in_worker", 0)) + 1
    processed = int(ctx["processed_in_worker"])

    update_worker_memory_registry()
    maybe_clear_heaviest_worker_for_global_memory()

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
            reason=f"worker_memory_guard_{memory_mb:.1f}_mb_ge_{max_memory_mb:.1f}_mb"
        )
        update_worker_memory_registry()
        return

    if should_clear_periodically:
        clear_worker_caches(reason=f"periodic_every_{every}")
        update_worker_memory_registry()


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

def _terminal_value_from_outcome(
    solved: bool,
    done: bool,
    termination_reason: str | None,
) -> tuple[float | None, str]:
    """
    Convert final episode status into AlphaZero-like terminal value.

    Returns:
        terminal_value:
            +1.0 for solved episodes
             0.0 for handoff to redispatch
            -1.0 for failed / exhausted episodes
            None for non-terminal truncated episodes

        outcome_class:
            stable textual label saved into examples.csv
    """

    reason = "" if termination_reason is None else str(termination_reason)

    if bool(solved) or reason == "solved":
        return 1.0, "solved"

    if reason in {
        "handoff_to_redispatch",
        "handoff_to_redispatch_teacher",
        "handoff_to_redispatch_with_hard_overload",
    }:
        return 0.0, "handoff_to_redispatch_teacher"

    if reason in {
        "max_steps_reached",
        "power_flow_failed",
        "non_convergence",
        "unsafe_stop_with_hard_overload",
    }:
        return -1.0, reason

    # Важно: если эпизод не terminal, не надо делать вид,
    # что мы знаем итог. Иначе outcome target станет ложным.
    if not bool(done):
        return None, "truncated_non_terminal"

    # Любой неизвестный terminal без solved/handoff считаем плохим исходом.
    return -1.0, reason or "unknown_terminal_failure"


def add_outcome_value_targets_to_rows(
    rows: list[dict],
    gamma: float,
    group_keys: tuple[str, ...] = ("scenario_id",),
) -> None:
    """
    Add AlphaZero-like outcome value targets to generated teacher rows.

    For each scenario:
        solved                        -> +1.0
        handoff_to_redispatch_teacher ->  0.0
        max_steps_reached / failure   -> -1.0

    For each step:
        outcome_value_target_t = terminal_value * gamma ** steps_to_terminal

    If the episode is not truly terminal, outcome_value_target is set to NaN.
    Then GraphSelfPlayDataset will safely fall back to dense value_target.
    """

    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    groups: dict[tuple, list[dict]] = {}

    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    for _, group_rows in groups.items():
        group_rows.sort(key=lambda r: int(r.get("step", 0)))

        if not group_rows:
            continue

        terminal_row = group_rows[-1]

        terminal_value, outcome_class = _terminal_value_from_outcome(
            solved=bool(terminal_row.get("solved", False)),
            done=bool(terminal_row.get("done", False)),
            termination_reason=terminal_row.get("termination_reason"),
        )

        n = len(group_rows)

        for position, row in enumerate(group_rows):
            steps_to_terminal = n - position

            row["outcome_class"] = outcome_class
            row["outcome_steps_to_terminal"] = int(steps_to_terminal)
            row["outcome_value_target_mode"] = "alphazero_discounted"
            row["outcome_gamma"] = float(gamma)

            if terminal_value is None:
                row["outcome_value_target"] = float("nan")
            else:
                row["outcome_value_target"] = float(
                    terminal_value * (float(gamma) ** steps_to_terminal)
                )

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
# LODF screening
# ======================================================================================


class LODFScreenedImpactBeamSearchPlanner(ImpactBeamSearchPlanner):
    """
    ImpactBeamSearchPlanner with optional LODF-based candidate screening.

    Important:
        LODF is used only before expensive AC PF.
        Final children are still evaluated through env.step(), so teacher examples
        remain AC-validated.
    """

    def __init__(
        self,
        config: ImpactBeamSearchConfig,
        lodf_screen_top_k: int,
        lodf_min_candidate_count: int = 1,
    ):
        super().__init__(config)

        self.lodf_screen_top_k = int(lodf_screen_top_k)
        self.lodf_min_candidate_count = int(lodf_min_candidate_count)

    def _candidate_actions(
        self,
        env: TopologySwitchingEnv,
    ) -> list[GridFMAction]:
        base_actions = super()._candidate_actions(env)

        if self.lodf_screen_top_k <= 0:
            return base_actions

        state = env.current_state

        if state is None:
            return base_actions

        stop_actions = [
            action
            for action in base_actions
            if action.action_type == "do_nothing"
        ]

        switch_actions = [
            action
            for action in base_actions
            if action.action_type == "switch_off_branch"
        ]

        if len(switch_actions) < self.lodf_min_candidate_count:
            return base_actions

        if len(switch_actions) <= self.lodf_screen_top_k:
            return base_actions

        try:
            ranked_switch_actions = rank_actions_by_lodf_screening(
                state=state,
                actions=switch_actions,
            )
        except Exception:
            # LODF must never break teacher generation.
            # Fall back to original loading-based candidate order.
            ranked_switch_actions = switch_actions

        selected_switch_actions = ranked_switch_actions[: self.lodf_screen_top_k]

        return [*stop_actions, *selected_switch_actions]


def rank_actions_by_lodf_screening(
    state,
    actions: list[GridFMAction],
) -> list[GridFMAction]:
    """
    Rank switch-off actions by approximate post-contingency DC/LODF safety.

    Lower score is better.

    This is only a screening heuristic. The selected actions are still checked
    later by full AC PF through env.step().
    """

    if not actions:
        return actions

    status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")
    x_idx = BRANCH_FEATURE_COLUMNS.index("x")
    pf_idx = BRANCH_FEATURE_COLUMNS.index("pf")
    rate_idx = BRANCH_FEATURE_COLUMNS.index("rate_a")
    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    branch_features = state.branch_features
    edge_index = state.edge_index.astype(int)

    num_branches = int(branch_features.shape[0])
    num_buses = int(state.bus_features.shape[0])

    if num_branches <= 1 or num_buses <= 1:
        return actions

    status = branch_features[:, status_idx].astype(float)
    x = branch_features[:, x_idx].astype(float)
    pf = branch_features[:, pf_idx].astype(float)
    rate = branch_features[:, rate_idx].astype(float)

    active_mask = (
        (status > 0.0)
        & np.isfinite(x)
        & (np.abs(x) > 1e-9)
        & np.isfinite(rate)
        & (rate > 1e-9)
    )

    active_positions = np.where(active_mask)[0]

    if len(active_positions) <= 1:
        return actions

    active_pos_to_row = {
        int(branch_pos): int(row)
        for row, branch_pos in enumerate(active_positions.tolist())
    }

    active_from = edge_index[0, active_positions].astype(int)
    active_to = edge_index[1, active_positions].astype(int)

    if np.any(active_from < 0) or np.any(active_to < 0):
        return actions

    if np.any(active_from >= num_buses) or np.any(active_to >= num_buses):
        return actions

    active_x = x[active_positions]
    active_b = 1.0 / active_x

    m = len(active_positions)
    n = num_buses

    incidence = np.zeros((m, n), dtype=np.float64)
    incidence[np.arange(m), active_from] = 1.0
    incidence[np.arange(m), active_to] = -1.0

    if n <= 1:
        return actions

    # Remove slack bus 0. This is enough for screening; final validation is AC PF.
    incidence_red = incidence[:, 1:]

    # Bbus = A^T diag(b) A.
    bbus = incidence_red.T @ (active_b[:, None] * incidence_red)

    try:
        bbus_inv = np.linalg.pinv(bbus, rcond=1e-10)
    except Exception:
        return actions

    # PTDF for branch-to-branch transactions.
    # H[l, k] = flow on line l caused by a transaction from from_bus(k) to to_bus(k).
    h = (active_b[:, None] * incidence_red) @ bbus_inv @ incidence_red.T

    diag_h = np.diag(h)
    denom = 1.0 - diag_h

    active_pf = pf[active_positions]
    active_rate = rate[active_positions]

    scored: list[tuple[float, GridFMAction]] = []

    for action in actions:
        branch_pos = int(getattr(action, "branch_pos", -1))

        k = active_pos_to_row.get(branch_pos)

        if k is None:
            # Inactive or invalid branch - make it unattractive.
            scored.append((float("inf"), action))
            continue

        if not np.isfinite(denom[k]) or abs(float(denom[k])) < 1e-9:
            scored.append((float("inf"), action))
            continue

        lodf_col = h[:, k] / denom[k]

        flow_after = active_pf + lodf_col * active_pf[k]
        flow_after[k] = 0.0

        loading_after = np.divide(
            np.abs(flow_after),
            active_rate,
            out=np.zeros_like(flow_after, dtype=np.float64),
            where=active_rate > 1e-9,
        ) * 100.0

        loading_after = np.nan_to_num(
            loading_after,
            nan=0.0,
            posinf=1e9,
            neginf=1e9,
        )

        score = lodf_loading_safety_score(loading_after)

        # Small tie-breaker: prefer currently loaded lines if LODF score is equal.
        current_loading = float(branch_features[branch_pos, loading_idx])
        score -= 1e-4 * current_loading

        scored.append((float(score), action))

    scored.sort(key=lambda x: x[0])

    return [action for _, action in scored]


def lodf_loading_safety_score(loading_percent: np.ndarray) -> float:
    """
    Approximate safety score based only on predicted DC loading.

    This mirrors the overload philosophy of safety_score(), but without voltage
    and reactive power terms. It is intentionally conservative.
    """

    loading = np.asarray(loading_percent, dtype=np.float64)

    overload = np.maximum(loading - 100.0, 0.0)
    hard = np.maximum(loading - 120.0, 0.0)

    num_overloaded = float(np.sum(loading > 100.0))
    num_hard = float(np.sum(loading > 120.0))

    hard_sq = float(np.sum(hard * hard))
    hard_sum = float(np.sum(hard))
    over_sum = float(np.sum(overload))
    max_hard = float(np.max(hard)) if hard.size else 0.0
    max_over = float(np.max(overload)) if overload.size else 0.0

    return float(
        3.0 * hard_sq
        + 1500.0 * num_hard
        + 50.0 * hard_sum
        + 30.0 * max_hard
        + 5.0 * over_sum
        + 100.0 * num_overloaded
        + 2.0 * max_over
    )


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

        if bool(task.get("use_lodf_screening", False)):
            planner = LODFScreenedImpactBeamSearchPlanner(
                config=planner_config,
                lodf_screen_top_k=int(task["lodf_screen_top_k"]),
                lodf_min_candidate_count=int(task["lodf_min_candidate_count"]),
            )
        else:
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


def _console_write(message: str) -> None:
    """
    Print without breaking tqdm bars when tqdm is active.
    """

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
        "min_free_system_memory_mb": float(args.min_free_system_memory_mb),
        "memory_registry_max_age_sec": float(args.memory_registry_max_age_sec),
        "auto_worker_memory_mb": float(args.auto_worker_memory_mb),
        "auto_worker_memory_reserve_mb": float(args.auto_worker_memory_reserve_mb),
        "auto_worker_cpu_util_target": float(args.auto_worker_cpu_util_target),
        "use_lodf_screening": bool(args.use_lodf_screening),
        "lodf_screen_top_k": int(args.lodf_screen_top_k),
        "lodf_min_candidate_count": int(args.lodf_min_candidate_count),
        "auto_worker_cpu_mode": str(args.auto_worker_cpu_mode),
        "auto_worker_cpu_fraction": float(args.auto_worker_cpu_fraction),
        "auto_worker_max": int(args.auto_worker_max),
    }



def resolve_num_workers(
    num_workers_arg: str,
    num_batches: int,
    task_config: dict[str, Any],
) -> int:
    """
    Resolve --num-workers.

    Supports:
        --num-workers 4
        --num-workers auto

    Auto mode is intentionally adaptive:
    - CPU cap can use physical or logical cores;
    - memory cap uses currently available RAM minus reserve;
    - optional auto_worker_max can limit aggressive auto selection.
    """

    value = str(num_workers_arg).strip().lower()

    if value != "auto":
        return max(int(value), 1)

    try:
        import psutil
    except Exception:
        fallback = max((os.cpu_count() or 2) - 1, 1)
        return min(fallback, int(num_batches))

    logical_cpu = psutil.cpu_count(logical=True) or (os.cpu_count() or 2)
    physical_cpu = psutil.cpu_count(logical=False) or logical_cpu

    cpu_mode = str(task_config.get("auto_worker_cpu_mode", "logical")).lower()
    cpu_fraction = float(task_config.get("auto_worker_cpu_fraction", 0.85))
    cpu_fraction = min(max(cpu_fraction, 0.1), 1.0)

    if cpu_mode == "physical":
        base_cpu_count = int(physical_cpu)
    else:
        base_cpu_count = int(logical_cpu)

    cpu_cap = max(int(base_cpu_count * cpu_fraction), 1)

    cpu_load = get_cpu_load_percent()
    target_cpu = float(task_config.get("auto_worker_cpu_util_target", 85.0))

    # If the machine is already busy before starting, reduce the cap.
    # Do not reduce it too aggressively, because teacher load is bursty.
    if cpu_load is not None and cpu_load > target_cpu:
        cpu_cap = max(int(cpu_cap * 0.75), 1)

    available_mb = get_system_available_memory_mb()

    estimated_worker_mb = float(task_config.get("auto_worker_memory_mb", 1000.0))
    reserve_mb = float(task_config.get("auto_worker_memory_reserve_mb", 2048.0))

    if available_mb is None:
        memory_cap = cpu_cap
    else:
        usable_mb = max(float(available_mb) - reserve_mb, 0.0)
        memory_cap = max(int(usable_mb // max(estimated_worker_mb, 1.0)), 1)

    auto_worker_max = int(task_config.get("auto_worker_max", 0))

    workers = max(
        1,
        min(
            int(num_batches),
            int(cpu_cap),
            int(memory_cap),
        ),
    )

    if auto_worker_max > 0:
        workers = min(workers, int(auto_worker_max))

    print("")
    print("Auto worker selection:")
    print(f"  logical CPU:        {logical_cpu}")
    print(f"  physical CPU:       {physical_cpu}")
    print(f"  CPU mode:           {cpu_mode}")
    print(f"  CPU fraction:       {cpu_fraction}")
    print(f"  current CPU load:   {cpu_load}")
    print(f"  available RAM MB:   {available_mb}")
    print(f"  reserve RAM MB:     {reserve_mb}")
    print(f"  worker RAM MB est:  {estimated_worker_mb}")
    print(f"  CPU cap:            {cpu_cap}")
    print(f"  memory cap:         {memory_cap}")
    print(f"  auto worker max:    {auto_worker_max}")
    print(f"  selected workers:   {workers}")
    print("")

    return workers


# ======================================================================================
# Execution modes
# ======================================================================================


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

    manager = None
    memory_registry = None

    if float(task_config.get("min_free_system_memory_mb", 0.0)) > 0.0:
        manager = mp.Manager()
        memory_registry = manager.dict()

    executor_kwargs = {
        "max_workers": int(num_workers),
        "initializer": init_worker_context,
        "initargs": (str(raw_dir), str(states_dir), task_config, memory_registry),
    }

    max_tasks_per_child = int(task_config.get("max_tasks_per_child", 0))

    if max_tasks_per_child > 0:
        executor_kwargs["max_tasks_per_child"] = max_tasks_per_child

    try:
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
    finally:
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass

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
        type=str,
        default="2",
        help="Number of workers or 'auto'.",
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
            "Use 0 to disable. On Windows this may interact badly with long queues; "
            "prefer 0 plus memory guards for long production runs."
        ),
    )

    parser.add_argument(
        "--min-free-system-memory-mb",
        type=float,
        default=0.0,
        help=(
            "If > 0, workers cooperatively clear caches when available system "
            "RAM drops below this value."
        ),
    )

    parser.add_argument(
        "--memory-registry-max-age-sec",
        type=float,
        default=120.0,
        help="Ignore stale worker memory records older than this many seconds.",
    )

    parser.add_argument(
        "--auto-worker-memory-mb",
        type=float,
        default=1200.0,
        help="Estimated RAM usage per worker for --num-workers auto.",
    )

    parser.add_argument(
        "--auto-worker-memory-reserve-mb",
        type=float,
        default=2048.0,
        help="RAM reserve kept free when using --num-workers auto.",
    )

    parser.add_argument(
        "--auto-worker-cpu-util-target",
        type=float,
        default=85.0,
        help="If current CPU load is above this percent, auto workers are reduced.",
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
        "--auto-worker-cpu-mode",
        type=str,
        default="logical",
        choices=["physical", "logical"],
        help="CPU cap mode for --num-workers auto.",
    )

    parser.add_argument(
        "--auto-worker-cpu-fraction",
        type=float,
        default=0.85,
        help=(
            "Fraction of selected CPU count allowed for --num-workers auto. "
            "Example: 0.85 of 8 logical CPUs -> 6 workers."
        ),
    )

    parser.add_argument(
        "--auto-worker-max",
        type=int,
        default=0,
        help="Optional hard upper limit for --num-workers auto. Use 0 for no explicit limit.",
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
        default="7000",
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

    resolved_num_workers = resolve_num_workers(
        num_workers_arg=str(args.num_workers),
        num_batches=len(scenario_batches),
        task_config=task_config,
    )

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
    print(f"Min free RAM MB:      {args.min_free_system_memory_mb}")
    print(f"Print memory events:  {args.print_memory_events}")
    print(f"Max tasks per child:  {args.max_tasks_per_child}")
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

    if int(resolved_num_workers) <= 1:
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
            num_workers=int(resolved_num_workers),
            verbose_success=verbose_success,
        )

    if not rows:
        raise RuntimeError("No teacher examples were generated.")


    add_outcome_value_targets_to_rows(
        rows=rows,
        gamma=float(args.gamma),
        group_keys=("scenario_id",),
    )

    print("Outcome value target mode: alphazero_discounted")
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

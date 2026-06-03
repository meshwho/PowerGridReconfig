from __future__ import annotations

import argparse
import json
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


def discounted_returns(
    rewards: list[float],
    gamma: float,
) -> list[float]:
    """
    Compute discounted returns from every step.

    Example:
        rewards = [r0, r1, r2]

        return[0] = r0 + gamma*r1 + gamma^2*r2
        return[1] = r1 + gamma*r2
        return[2] = r2
    """

    returns = [0.0 for _ in rewards]
    running = 0.0

    for i in reversed(range(len(rewards))):
        running = float(rewards[i]) + float(gamma) * running
        returns[i] = float(running)

    return returns


def make_one_hot_policy(action_id: int) -> dict[int, float]:
    """
    Create deterministic policy target.
    """

    return {int(action_id): 1.0}


def make_policy_from_final_beam(
    result: ImpactBeamSearchResult,
    temperature: float,
) -> tuple[dict[int, float], dict[int, int]]:
    """
    Convert final beam into a policy target over FIRST actions.

    temperature <= 0:
        one-hot policy for the best sequence first action.

    temperature > 0:
        soft policy. Final-beam trajectories close to the best safety score
        receive higher weights.

    This is used only for step 0. For later states, we use one-hot targets
    from the selected teacher sequence, because we do not run a new beam search
    for every intermediate state.
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


def load_scenario_ids(
    transitions_path: Path,
    limit: int | None,
) -> list[int]:
    """
    Read scenario IDs from transitions CSV.
    """

    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")

    transitions = pd.read_csv(transitions_path)

    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions file must contain scenario_id column: {transitions_path}"
        )

    scenario_ids = sorted(int(x) for x in transitions["scenario_id"].unique())

    if limit is not None:
        scenario_ids = scenario_ids[: int(limit)]

    return scenario_ids


def _action_is_valid(
    action_mask: np.ndarray,
    action_id: int,
) -> bool:
    """
    Check whether action_id is valid according to current state's action mask.
    """

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
    """
    Convert action_id to an executable environment action.
    """

    action_id = int(action_id)

    if action_id == 0:
        return make_do_nothing_action()

    return env.action_by_id(action_id)

def _force_stop_action_valid(action_mask: np.ndarray) -> np.ndarray:
    """
    Ensure that action 0 can be used as a teacher handoff target.

    This is important because the training dataset masks target_policy by
    action_mask. If action 0 is False in the mask, the stop/handoff target
    would be erased during training.
    """

    fixed_mask = np.array(action_mask, dtype=bool).copy()

    if fixed_mask.shape[0] > 0:
        fixed_mask[0] = True

    return fixed_mask


def _get_state_hard_count(state) -> int:
    return int(state.metrics["num_hard_overloaded_branches"])


def _get_state_overloaded_count(state) -> int:
    return int(state.metrics["num_overloaded_branches"])


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

    Returns:
        continue_action:
            True  -> save/execute topology action;
            False -> save action 0 as handoff_to_redispatch.

        reason:
            Human-readable decision reason.

        improvement:
            safety_before - safety_after.
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

    # Different thresholds are useful:
    # - while hard overload exists, we allow smaller useful steps;
    # - after hard overload is cleared, we require stronger improvement,
    #   otherwise redispatch is preferred.
    if hard_before > 0:
        required_improvement = float(task["min_continue_improvement_with_hard"])
    else:
        required_improvement = float(task["min_continue_improvement_without_hard"])

    # Always accept a step that reduces hard-overload count and improves safety.
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

def process_one_scenario_worker(task: dict[str, Any]) -> dict[str, Any]:
    """
    Process one scenario.

    This function is self-contained so it can run inside a separate process
    on Windows.

    For each scenario:
        1. reset environment;
        2. run impact beam search once;
        3. take the best sequence;
        4. replay the sequence from the initial state;
        5. save one training example for every step of that sequence.

    This is the important difference from the previous teacher generator:
    it creates multi-step imitation data, not only state_0 -> first_action.
    """

    scenario_id = int(task["scenario_id"])
    raw_dir = Path(str(task["raw_dir"]))
    states_dir = Path(str(task["states_dir"]))

    try:
        adapter = GridFMAdapter(raw_dir)

        backend = GridFMPowerFlowBackend(
            adapter=adapter,
            pf_alg=int(task["pf_alg"]),
            enable_cache=not bool(task["disable_cache"]),
        )

        action_space = GridFMActionSpace(
            require_connected_after_switch=True,
            enable_cache=not bool(task["disable_cache"]),
        )

        reward_fn = GridFMReward()

        # ------------------------------------------------------------------
        # First environment: used by impact beam search.
        # ------------------------------------------------------------------
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
            return {
                "ok": False,
                "scenario_id": scenario_id,
                "reason": "no_teacher_action_found",
                "traceback": None,
            }

        final_teacher_safety = float(best.safety_score)
        total_safety_improvement = float(initial_safety - final_teacher_safety)

        if total_safety_improvement < float(task["min_safety_improvement"]):
            return {
                "ok": False,
                "scenario_id": scenario_id,
                "reason": (
                    f"safety_improvement {total_safety_improvement:.4f} "
                    f"< {float(task['min_safety_improvement']):.4f}"
                ),
                "traceback": None,
            }

        # Root soft policy can be useful for step 0.
        root_policy_target, root_visit_counts = make_policy_from_final_beam(
            result=result,
            temperature=float(task["soft_policy_temperature"]),
        )

        if not root_policy_target:
            return {
                "ok": False,
                "scenario_id": scenario_id,
                "reason": "empty_root_policy_target",
                "traceback": None,
            }

        # ------------------------------------------------------------------
        # Second environment: replay best teacher sequence step by step.
        # We save a training example before every action.
        # ------------------------------------------------------------------
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

            # --------------------------------------------------------------
            # Test the candidate teacher action on a cloned environment first.
            # We do not mutate replay_env until we know that the action is useful.
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # The action is useful. Accept the cloned environment as the new
            # replay environment.
            # --------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Add final handoff example after the useful topology sequence.
        # ------------------------------------------------------------------
        # If the teacher executed all useful planned topology actions and did
        # not explicitly stop yet, teach the network that the next decision is
        # handoff_to_redispatch.
        #
        # This prevents the policy from learning:
        #     "always continue switching until max_steps_reached"
        #
        # Instead, it learns:
        #     "perform useful topology actions, then stop/handoff".
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

        state_store = GridFMStateStore(states_dir)

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
                    "source": "impact_beam_teacher_multistep",
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
                    "best_sequence_action_ids": [int(x) for x in best.action_ids],
                    "best_sequence_branch_ids": [
                        None if x is None else int(x)
                        for x in best.branch_ids
                    ],
                    "best_max_loading_percent": float(best.max_loading_percent),
                    "best_num_hard_overloaded": int(best.num_hard_overloaded),
                    "best_num_overloaded": int(best.num_overloaded),
                    "best_total_hard_overload": float(best.total_hard_overload),
                    "best_squared_hard_overload": float(
                        best.squared_hard_overload
                    ),
                    "best_total_overload": float(best.total_overload),
                    "replay_final_max_loading_percent": float(final_max_loading),
                    "replay_final_num_hard_overloaded": int(final_num_hard),
                    "replay_final_num_overloaded": int(final_num_overloaded),
                    "beam_depth": int(task["depth"]),
                    "beam_width": int(task["beam_width"]),
                    "candidate_pool": int(task["candidate_pool"]),
                    "top_k": int(task["top_k"]),
                    "soft_policy_temperature": float(
                        task["soft_policy_temperature"]
                    ),
                    "use_soft_root_policy": bool(task["use_soft_root_policy"]),
                    "evaluated_actions": int(result.evaluated_actions),
                    "teacher_decision_reason": item.get("teacher_decision_reason"),
                    "handoff_added": bool(handoff_added),
                    "handoff_reason": handoff_reason,
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
                "sequence": best.short_sequence(),
                "evaluated_actions": int(result.evaluated_actions),
            },
        }

    except Exception:
        return {
            "ok": False,
            "scenario_id": scenario_id,
            "reason": "exception",
            "traceback": traceback.format_exc(),
        }


def build_tasks(
    scenario_ids: list[int],
    raw_dir: Path,
    states_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """
    Build serializable worker tasks.
    """

    tasks: list[dict[str, Any]] = []

    for scenario_id in scenario_ids:
        tasks.append(
            {
                "scenario_id": int(scenario_id),
                "raw_dir": str(raw_dir),
                "states_dir": str(states_dir),
                "depth": int(args.depth),
                "beam_width": int(args.beam_width),
                "candidate_pool": int(args.candidate_pool),
                "top_k": int(args.top_k),
                "gamma": float(args.gamma),
                "pf_alg": int(args.pf_alg),
                "max_steps": int(args.max_steps),
                "max_teacher_steps": int(args.max_teacher_steps),
                "soft_policy_temperature": float(args.soft_policy_temperature),
                "use_soft_root_policy": bool(args.use_soft_root_policy),
                "min_safety_improvement": float(args.min_safety_improvement),
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
            }
        )

    return tasks


def print_success(result: dict[str, Any]) -> None:
    """
    Print one successful scenario result.
    """

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
        f"seq={summary['sequence']}"
    )


def print_failure(result: dict[str, Any]) -> None:
    """
    Print one failed/skipped scenario result.
    """

    print(
        f"Scenario {result['scenario_id']}: skipped | "
        f"reason={result['reason']}"
    )

    if result.get("traceback"):
        print(result["traceback"])


def run_sequential(
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Sequential generation.
    """

    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0

    iterator = tasks

    if tqdm is not None:
        iterator = tqdm(
            tasks,
            desc="Teacher scenarios",
            unit="scenario",
            dynamic_ncols=True,
        )

    for task in iterator:
        result = process_one_scenario_worker(task)

        if result["ok"]:
            rows.extend(result["rows"])
            total_saved += 1
            print_success(result)
        else:
            total_skipped += 1
            print_failure(result)

    return rows, total_saved, total_skipped


def run_parallel(
    tasks: list[dict[str, Any]],
    num_workers: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Parallel generation over scenarios.

    Each worker handles a full impact beam search for one scenario and saves
    all step states for that scenario.
    """

    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0

    print(f"\nParallel mode: {num_workers} workers")

    with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
        futures = [
            executor.submit(process_one_scenario_worker, task)
            for task in tasks
        ]

        iterator = as_completed(futures)

        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="Teacher scenarios",
                unit="scenario",
                dynamic_ncols=True,
            )

        for future in iterator:
            result = future.result()

            if result["ok"]:
                rows.extend(result["rows"])
                total_saved += 1
                print_success(result)
            else:
                total_skipped += 1
                print_failure(result)

    return rows, total_saved, total_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multi-step teacher examples using safety-aware "
            "impact beam search."
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
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=70)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--pf-alg", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=5)

    parser.add_argument(
        "--max-teacher-steps",
        type=int,
        default=4,
        help=(
            "Maximum number of teacher sequence steps to save as examples. "
            "Usually equal to --depth."
        ),
    )

    parser.add_argument(
        "--soft-policy-temperature",
        type=float,
        default=500.0,
        help=(
            "Temperature for converting final beam into soft root policy. "
            "Use 0 for deterministic one-hot root target."
        ),
    )

    parser.add_argument(
        "--use-soft-root-policy",
        action="store_true",
        help=(
            "Use soft final-beam policy for step 0. "
            "All later steps use one-hot targets from the best teacher sequence."
        ),
    )

    parser.add_argument(
        "--min-safety-improvement",
        type=float,
        default=0.0,
        help="Skip scenario if best final safety improvement is below this value.",
    )

    parser.add_argument(
        "--allow-hard-count-increase",
        action="store_true",
        help="Allow beam search to increase hard-overload count.",
    )

    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable power flow and action-space caches inside workers.",
    )

    parser.add_argument(
        "--power-flow-failure-penalty",
        type=float,
        default=1_000_000.0,
        help="Safety penalty used if replay action causes power-flow failure.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 for sequential mode.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional scenario limit for quick testing.",
    )

    parser.add_argument(
        "--min-continue-improvement-with-hard",
        type=float,
        default=250.0,
        help=(
            "Minimum safety-score improvement required to continue topology "
            "switching while hard overloads still exist."
        ),
    )

    parser.add_argument(
        "--min-continue-improvement-without-hard",
        type=float,
        default=500.0,
        help=(
            "Minimum safety-score improvement required to continue topology "
            "switching after hard overloads are removed."
        ),
    )

    parser.add_argument(
        "--max-loading-increase-limit",
        type=float,
        default=5.0,
        help=(
            "Reject teacher action if it increases max loading by more than "
            "this number of percentage points."
        ),
    )

    parser.add_argument(
        "--add-handoff-example",
        action="store_true",
        help=(
            "Add action 0 teacher example when the next topology action is not useful."
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

    print("=" * 100)
    print("Generating multi-step impact-beam teacher examples")
    print("=" * 100)
    print(f"Raw directory:        {raw_dir.resolve()}")
    print(f"Transitions:          {transitions_path.resolve()}")
    print(f"Output dir:           {output_dir}")
    print(f"States dir:           {states_dir}")
    print(f"Examples CSV:         {examples_path}")
    print(f"Scenarios:            {len(scenario_ids)}")
    print(f"Depth:                {args.depth}")
    print(f"Beam width:           {args.beam_width}")
    print(f"Candidate pool:       {args.candidate_pool}")
    print(f"Top-K actions:        {args.top_k}")
    print(f"Gamma:                {args.gamma}")
    print(f"PF algorithm:         {args.pf_alg}")
    print(f"Max teacher steps:    {args.max_teacher_steps}")
    print(f"Soft root policy:     {args.use_soft_root_policy}")
    print(f"Soft policy temp:     {args.soft_policy_temperature}")
    print(f"Min safety improve:   {args.min_safety_improvement}")
    print(f"Allow hard increase:  {args.allow_hard_count_increase}")
    print(f"Cache enabled:        {not args.disable_cache}")
    print(f"Num workers:          {args.num_workers}")
    print(f"Continue improvement hard:    {args.min_continue_improvement_with_hard}")
    print(f"Continue improvement no hard: {args.min_continue_improvement_without_hard}")
    print(f"Max loading increase limit:   {args.max_loading_increase_limit}")
    print(f"Add handoff examples:         {args.add_handoff_example}")

    tasks = build_tasks(
        scenario_ids=scenario_ids,
        raw_dir=raw_dir,
        states_dir=states_dir,
        args=args,
    )

    if int(args.num_workers) <= 1:
        rows, total_saved, total_skipped = run_sequential(tasks)
    else:
        rows, total_saved, total_skipped = run_parallel(
            tasks=tasks,
            num_workers=int(args.num_workers),
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
    print("Multi-step impact teacher generation summary")
    print("=" * 100)
    print(f"Saved scenarios: {total_saved}")
    print(f"Skipped:         {total_skipped}")
    print(f"Saved examples:  {len(examples_df)}")
    print(f"Examples CSV:    {examples_path}")
    print(f"States dir:      {states_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
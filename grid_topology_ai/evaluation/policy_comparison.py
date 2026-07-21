from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.evaluation.metrics import build_evaluation_metrics
from grid_topology_ai.search.root_policy import (
    constrain_policy,
    normalize_policy,
    require_action_in_policy_support,
    select_action_from_policy,
)


class PolicyMode(StrEnum):
    UNGATED = "ungated"
    CONSTRAINED = "constrained"


@dataclass(frozen=True, slots=True)
class RootPolicyDecision:
    mode: PolicyMode
    raw_policy: dict[int, float]
    policy: dict[int, float]
    action_id: int | None
    branch_id: int | None
    allowed_action_ids: tuple[int, ...]
    constraint_changed_policy: bool
    empty_constrained_support: bool


def evaluation_policy_modes(compare_constrained: bool) -> tuple[PolicyMode, ...]:
    if compare_constrained:
        return (PolicyMode.UNGATED, PolicyMode.CONSTRAINED)
    return (PolicyMode.UNGATED,)


def select_evaluation_root_policy(
    *,
    search_result: Any,
    mode: PolicyMode | str,
    continuation_analysis: Any | None = None,
) -> RootPolicyDecision:
    parsed_mode = PolicyMode(mode)
    raw_policy = normalize_policy(
        search_result.policy,
        context=f"{parsed_mode.value} MCTS root policy",
    )

    if parsed_mode is PolicyMode.UNGATED:
        policy = raw_policy
        allowed_action_ids = tuple(raw_policy)
    else:
        if continuation_analysis is None:
            raise ValueError(
                "constrained evaluation requires continuation analysis"
            )
        allowed = {
            int(action_id)
            for action_id in continuation_analysis.allowed_action_ids
        }
        # Stop legality is already enforced by MCTS stop_policy. Preserve a
        # searched stop action instead of treating the continuation heuristic
        # as a second, conflicting stop policy.
        if 0 in raw_policy:
            allowed.add(0)
        allowed_action_ids = tuple(sorted(allowed))
        policy = constrain_policy(
            raw_policy,
            allowed_action_ids,
            context="constrained MCTS root policy",
        )

    changed = not _policies_close(raw_policy, policy)
    if not policy:
        return RootPolicyDecision(
            mode=parsed_mode,
            raw_policy=raw_policy,
            policy={},
            action_id=None,
            branch_id=None,
            allowed_action_ids=allowed_action_ids,
            constraint_changed_policy=changed,
            empty_constrained_support=True,
        )

    action_id = select_action_from_policy(
        policy,
        temperature=0.0,
        rng=np.random.default_rng(0),
        context=f"{parsed_mode.value} evaluation policy",
    )
    require_action_in_policy_support(
        action_id,
        policy,
        context=f"{parsed_mode.value} evaluation policy",
    )

    if action_id == 0:
        branch_id = None
    else:
        action = search_result.root.actions_by_id.get(action_id)
        if action is None:
            raise RuntimeError(
                f"Action {action_id} is present in the {parsed_mode.value} "
                "policy but missing from root.actions_by_id."
            )
        branch_id = action.branch_id

    return RootPolicyDecision(
        mode=parsed_mode,
        raw_policy=raw_policy,
        policy=policy,
        action_id=int(action_id),
        branch_id=branch_id,
        allowed_action_ids=allowed_action_ids,
        constraint_changed_policy=changed,
        empty_constrained_support=False,
    )


def build_policy_comparison_metrics(
    *,
    df: pd.DataFrame,
    failed_results: list[dict[str, Any]],
    requested_scenarios: int,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    modes = tuple(
        PolicyMode(mode)
        for mode in task_config.get("evaluation_modes", [PolicyMode.UNGATED.value])
    )
    mode_metrics: dict[str, dict[str, Any]] = {}

    for mode in modes:
        subset = df[df["policy_mode"] == mode.value].copy()
        if subset.empty:
            continue
        mode_failures = [
            item
            for item in failed_results
            if item.get("policy_mode", mode.value) == mode.value
        ]
        mode_task = dict(task_config)
        mode_task["policy_mode"] = mode.value
        mode_task["use_continuation_gate"] = (
            mode is PolicyMode.CONSTRAINED
        )
        mode_metrics[mode.value] = build_evaluation_metrics(
            df=subset,
            failed_results=mode_failures,
            requested_scenarios=requested_scenarios,
            task_config=mode_task,
        )

    if not mode_metrics:
        raise RuntimeError("No evaluation mode produced metrics.")

    primary_mode = (
        PolicyMode.CONSTRAINED.value
        if PolicyMode.CONSTRAINED.value in mode_metrics
        else PolicyMode.UNGATED.value
    )
    metrics = dict(mode_metrics[primary_mode])
    metrics["evaluation_mode"] = (
        "comparison" if len(mode_metrics) > 1 else primary_mode
    )
    metrics["primary_policy_mode"] = primary_mode
    metrics["mode_metrics"] = mode_metrics

    if {
        PolicyMode.UNGATED.value,
        PolicyMode.CONSTRAINED.value,
    }.issubset(mode_metrics):
        metrics["comparison"] = _comparison_summary(
            df=df,
            mode_metrics=mode_metrics,
        )

    return metrics


def print_policy_comparison_summary(metrics: dict[str, Any]) -> None:
    comparison = metrics.get("comparison")
    if not isinstance(comparison, dict):
        return

    print("\n" + "=" * 100)
    print("Ungated vs constrained MCTS")
    print("=" * 100)
    print(f"Paired scenarios:                {comparison['paired_scenarios']}")
    print(
        "Action sequence changed:          "
        f"{comparison['action_sequence_changed_scenarios']}"
    )
    print(
        "Constraint changed policy:        "
        f"{comparison['policy_changed_scenarios']} scenarios / "
        f"{comparison['policy_changed_steps']} steps"
    )
    print(
        "Empty constrained support:        "
        f"{comparison['empty_constrained_support_scenarios']} scenarios / "
        f"{comparison['empty_constrained_support_count']} decisions"
    )
    print(
        "Solve-rate delta:                 "
        f"{_format_delta(comparison['solve_rate_delta'])}"
    )
    print(
        "Average return delta:             "
        f"{_format_delta(comparison['avg_discounted_return_delta'])}"
    )
    print(
        "Average safety-score delta:       "
        f"{_format_delta(comparison['avg_safety_score_delta'])}"
    )


def _comparison_summary(
    *,
    df: pd.DataFrame,
    mode_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ungated = df[df["policy_mode"] == PolicyMode.UNGATED.value]
    constrained = df[df["policy_mode"] == PolicyMode.CONSTRAINED.value]
    paired = ungated.merge(
        constrained,
        on="scenario_id",
        how="inner",
        suffixes=("_ungated", "_constrained"),
    )
    constrained_metrics = mode_metrics[PolicyMode.CONSTRAINED.value]
    ungated_metrics = mode_metrics[PolicyMode.UNGATED.value]

    return {
        "paired_scenarios": int(len(paired)),
        "action_sequence_changed_scenarios": int(
            (paired["actions_ungated"] != paired["actions_constrained"]).sum()
        ),
        "policy_changed_scenarios": int(
            constrained["constraint_changed_policy"].astype(bool).sum()
        ),
        "policy_changed_steps": int(
            constrained["constraint_changed_policy_steps"].sum()
        ),
        "empty_constrained_support_scenarios": int(
            constrained["constraint_exhausted"].astype(bool).sum()
        ),
        "empty_constrained_support_count": int(
            constrained["empty_constrained_support_count"].sum()
        ),
        "solve_rate_delta": _metric_delta(
            constrained_metrics,
            ungated_metrics,
            "solve_rate",
        ),
        "avg_discounted_return_delta": _metric_delta(
            constrained_metrics,
            ungated_metrics,
            "avg_discounted_return",
        ),
        "avg_safety_score_delta": _metric_delta(
            constrained_metrics,
            ungated_metrics,
            "avg_safety_score",
        ),
        "avg_final_loading_percent_delta": _metric_delta(
            constrained_metrics,
            ungated_metrics,
            "avg_final_loading_percent",
        ),
        "hard_overload_free_rate_delta": _metric_delta(
            constrained_metrics,
            ungated_metrics,
            "hard_overload_free_rate",
        ),
        "power_flow_failure_rate_delta": _metric_delta(
            constrained_metrics,
            ungated_metrics,
            "power_flow_failure_rate",
        ),
    }


def _metric_delta(
    constrained: dict[str, Any],
    ungated: dict[str, Any],
    name: str,
) -> float | None:
    constrained_value = constrained.get(name)
    ungated_value = ungated.get(name)
    if constrained_value is None or ungated_value is None:
        return None
    return float(constrained_value) - float(ungated_value)


def _policies_close(
    left: dict[int, float],
    right: dict[int, float],
) -> bool:
    if left.keys() != right.keys():
        return False
    return all(
        math.isclose(
            left[action_id],
            right[action_id],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for action_id in left
    )


def _format_delta(value: object) -> str:
    return "n/a" if value is None else f"{float(value):+.6f}"

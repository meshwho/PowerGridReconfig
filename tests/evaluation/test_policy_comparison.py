from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from grid_topology_ai.evaluation import policy_comparison
from grid_topology_ai.evaluation.policy_comparison import (
    PolicyMode,
    build_policy_comparison_metrics,
    evaluation_policy_modes,
    select_evaluation_root_policy,
)


class _Action:
    def __init__(self, branch_id: int | None) -> None:
        self.branch_id = branch_id


def _search_result(policy: dict[int, float]) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        root=SimpleNamespace(
            actions_by_id={
                1: _Action(11),
                2: _Action(22),
            }
        ),
    )


def _analysis(*allowed_action_ids: int) -> SimpleNamespace:
    return SimpleNamespace(allowed_action_ids=allowed_action_ids)


def test_comparison_mode_evaluates_ungated_and_constrained() -> None:
    assert evaluation_policy_modes(False) == (PolicyMode.UNGATED,)
    assert evaluation_policy_modes(True) == (
        PolicyMode.UNGATED,
        PolicyMode.CONSTRAINED,
    )


def test_ungated_policy_uses_normalized_root_visits() -> None:
    decision = select_evaluation_root_policy(
        search_result=_search_result({1: 7.0, 2: 3.0}),
        mode=PolicyMode.UNGATED,
    )

    assert decision.policy == pytest.approx({1: 0.7, 2: 0.3})
    assert decision.action_id == 1
    assert decision.branch_id == 11
    assert decision.constraint_changed_policy is False


def test_constrained_policy_filters_and_renormalizes_root_visits() -> None:
    decision = select_evaluation_root_policy(
        search_result=_search_result({1: 0.7, 2: 0.3}),
        mode=PolicyMode.CONSTRAINED,
        continuation_analysis=_analysis(2),
    )

    assert decision.raw_policy == pytest.approx({1: 0.7, 2: 0.3})
    assert decision.policy == {2: 1.0}
    assert decision.action_id == 2
    assert decision.branch_id == 22
    assert decision.constraint_changed_policy is True


def test_constrained_policy_preserves_searched_stop_action() -> None:
    decision = select_evaluation_root_policy(
        search_result=_search_result({0: 0.2, 1: 0.5, 2: 0.3}),
        mode=PolicyMode.CONSTRAINED,
        continuation_analysis=_analysis(2),
    )

    assert decision.policy == pytest.approx({0: 0.4, 2: 0.6})
    assert decision.action_id == 2
    assert decision.allowed_action_ids == (0, 2)


def test_constrained_policy_reports_empty_support_without_fallback() -> None:
    decision = select_evaluation_root_policy(
        search_result=_search_result({1: 0.7, 2: 0.3}),
        mode=PolicyMode.CONSTRAINED,
        continuation_analysis=_analysis(),
    )

    assert decision.policy == {}
    assert decision.action_id is None
    assert decision.empty_constrained_support is True


def test_comparison_metrics_report_mode_deltas_and_constraint_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_metrics(*, df, failed_results, requested_scenarios, task_config):
        return {
            "solve_rate": float(df["solved"].mean()),
            "avg_discounted_return": float(df["discounted_return"].mean()),
            "avg_safety_score": float(df["safety_score"].mean()),
            "avg_final_loading_percent": float(
                df["final_max_loading_percent"].mean()
            ),
            "hard_overload_free_rate": float(
                df["hard_overload_free"].mean()
            ),
            "power_flow_failure_rate": 0.0,
            "task_config": task_config,
        }

    monkeypatch.setattr(
        policy_comparison,
        "build_evaluation_metrics",
        fake_metrics,
    )
    df = pd.DataFrame(
        [
            {
                "scenario_id": 1,
                "policy_mode": "ungated",
                "actions": "[1]",
                "solved": False,
                "discounted_return": 1.0,
                "safety_score": 10.0,
                "final_max_loading_percent": 130.0,
                "hard_overload_free": False,
                "constraint_changed_policy": False,
                "constraint_changed_policy_steps": 0,
                "constraint_exhausted": False,
                "empty_constrained_support_count": 0,
            },
            {
                "scenario_id": 1,
                "policy_mode": "constrained",
                "actions": "[2]",
                "solved": True,
                "discounted_return": 3.0,
                "safety_score": 20.0,
                "final_max_loading_percent": 95.0,
                "hard_overload_free": True,
                "constraint_changed_policy": True,
                "constraint_changed_policy_steps": 1,
                "constraint_exhausted": False,
                "empty_constrained_support_count": 0,
            },
        ]
    )

    metrics = build_policy_comparison_metrics(
        df=df,
        failed_results=[],
        requested_scenarios=1,
        task_config={"evaluation_modes": ["ungated", "constrained"]},
    )

    assert metrics["primary_policy_mode"] == "constrained"
    assert set(metrics["mode_metrics"]) == {"ungated", "constrained"}
    comparison = metrics["comparison"]
    assert comparison["paired_scenarios"] == 1
    assert comparison["action_sequence_changed_scenarios"] == 1
    assert comparison["policy_changed_scenarios"] == 1
    assert comparison["solve_rate_delta"] == 1.0
    assert comparison["avg_discounted_return_delta"] == 2.0

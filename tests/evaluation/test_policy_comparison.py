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


def test_comparison_mode_evaluates_constrained_as_secondary() -> None:
    assert evaluation_policy_modes(False) == (PolicyMode.UNGATED,)
    assert evaluation_policy_modes(True) == (
        PolicyMode.CONSTRAINED,
        PolicyMode.UNGATED,
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


def test_better_constrained_result_does_not_replace_ungated_headline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_metrics(*, df, failed_results, requested_scenarios, task_config):
        secure_rate = float(df["solved"].mean())
        return {
            "solve_rate": secure_rate,
            "physically_secure_rate_requested": secure_rate,
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
        task_config={
            "evaluation_modes": ["constrained", "ungated"],
            "primary_policy_mode": "ungated",
        },
    )

    assert metrics["primary_policy_mode"] == "ungated"
    assert metrics["solve_rate"] == pytest.approx(0.0)
    assert metrics["physically_secure_rate_requested"] == pytest.approx(0.0)
    assert set(metrics["mode_metrics"]) == {"ungated", "constrained"}
    assert metrics["mode_metrics"]["constrained"]["solve_rate"] == pytest.approx(
        1.0
    )
    assert metrics["ungated_physically_secure_rate_requested"] == pytest.approx(
        0.0
    )
    assert metrics[
        "constrained_physically_secure_rate_requested"
    ] == pytest.approx(1.0)
    assert metrics["continuation_gate_gain"] == pytest.approx(1.0)

    comparison = metrics["comparison"]
    assert comparison["paired_scenarios"] == 1
    assert comparison["action_sequence_changed_scenarios"] == 1
    assert comparison["policy_changed_scenarios"] == 1
    assert comparison["solve_rate_delta"] == pytest.approx(1.0)
    assert comparison[
        "physically_secure_rate_requested_delta"
    ] == pytest.approx(1.0)
    assert comparison["avg_discounted_return_delta"] == pytest.approx(2.0)


def test_primary_policy_mode_must_produce_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        policy_comparison,
        "build_evaluation_metrics",
        lambda **kwargs: {"solve_rate": 1.0},
    )
    df = pd.DataFrame(
        [
            {
                "scenario_id": 1,
                "policy_mode": "ungated",
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="primary policy mode did not produce metrics",
    ):
        build_policy_comparison_metrics(
            df=df,
            failed_results=[],
            requested_scenarios=1,
            task_config={
                "evaluation_modes": ["ungated"],
                "primary_policy_mode": "constrained",
            },
        )

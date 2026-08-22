from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.evaluation import (
    EvaluationEpisodeTrace,
    build_evaluation_episode_row,
    build_evaluation_metrics,
)
from grid_topology_ai.physics.utility import state_security_penalty, state_utility
from grid_topology_ai.termination import TerminationReason


def _state(loading: float) -> GridFMState:
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[
        0,
        BRANCH_FEATURE_COLUMNS.index("br_status"),
    ] = 1.0
    branch_features[
        0,
        BRANCH_FEATURE_COLUMNS.index("loading_percent"),
    ] = float(loading)

    overloaded = int(loading > 100.0)
    hard = int(loading > 120.0)
    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=np.zeros((2, 1), dtype=np.float32),
        branch_features=branch_features,
        edge_index=np.array([[0], [1]], dtype=np.int64),
        branch_ids=np.array([10], dtype=np.int64),
        branch_status=np.array([1], dtype=np.int64),
        metrics={
            "power_flow_converged": True,
            "all_values_finite": True,
            "topology_connected": True,
            "max_loading_percent": float(loading),
            "num_overloaded_branches": overloaded,
            "num_hard_overloaded_branches": hard,
            "num_outaged_branches": 0,
            "total_thermal_overload_mva": max(float(loading) - 100.0, 0.0),
            "num_low_voltage_buses": 0,
            "num_high_voltage_buses": 0,
            "total_voltage_violation": 0.0,
            "num_generator_p_violations": 0,
            "total_generator_p_violation_mw": 0.0,
            "num_generator_q_violations": 0,
            "total_generator_q_violation_mvar": 0.0,
            "num_angle_difference_violations": 0,
            "total_angle_difference_violation_degrees": 0.0,
        },
        outaged_branch_ids=[],
    )


def _episode_row(
    *,
    initial: GridFMState,
    final: GridFMState,
    reason: TerminationReason,
    solved: bool = False,
    utility: float | None = None,
) -> dict[str, object]:
    if utility is None:
        utility = state_utility(final)
    env = SimpleNamespace(
        initial_state=initial,
        current_state=final,
        termination_reason=reason,
        done=True,
        solved=solved,
        terminal_outcome_evidence=SimpleNamespace(
            topology_utility=float(utility),
        ),
    )
    return build_evaluation_episode_row(
        scenario_id=1,
        policy_mode="ungated",
        env=env,
        trace=EvaluationEpisodeTrace(actions=[1]),
        physics_config=None,
    )


def test_evaluation_row_reports_canonical_topology_improvement() -> None:
    initial = _state(150.0)
    final = _state(110.0)

    row = _episode_row(
        initial=initial,
        final=final,
        reason=TerminationReason.MAX_STEPS_REACHED,
    )

    J0 = state_security_penalty(initial)
    Jfinal = state_security_penalty(final)
    assert row["J0"] == pytest.approx(J0)
    assert row["Jfinal"] == pytest.approx(Jfinal)
    assert row["delta_J"] == pytest.approx(J0 - Jfinal)
    assert row["relative_J_improvement"] == pytest.approx(
        (J0 - Jfinal) / J0
    )
    assert row["final_topology_utility"] == pytest.approx(
        state_utility(final)
    )


def test_solved_zero_penalty_state_reports_unit_utility() -> None:
    initial = _state(110.0)
    final = _state(90.0)

    row = _episode_row(
        initial=initial,
        final=final,
        reason=TerminationReason.SOLVED,
        solved=True,
    )

    assert row["Jfinal"] == pytest.approx(0.0)
    assert row["final_topology_utility"] == pytest.approx(1.0)
    assert row["delta_J"] == pytest.approx(row["J0"])
    assert row["relative_J_improvement"] == pytest.approx(1.0)


def test_power_flow_failure_does_not_report_last_valid_state_as_Jfinal() -> None:
    initial = _state(150.0)

    row = _episode_row(
        initial=initial,
        final=initial,
        reason=TerminationReason.POWER_FLOW_FAILED,
        utility=-1.0,
    )

    assert row["J0"] == pytest.approx(state_security_penalty(initial))
    assert np.isnan(float(row["Jfinal"]))
    assert np.isnan(float(row["delta_J"]))
    assert np.isnan(float(row["relative_J_improvement"]))
    assert row["final_topology_utility"] == pytest.approx(-1.0)


def test_evaluation_metrics_aggregate_topology_quality() -> None:
    improved = _episode_row(
        initial=_state(150.0),
        final=_state(110.0),
        reason=TerminationReason.MAX_STEPS_REACHED,
    )
    solved = _episode_row(
        initial=_state(110.0),
        final=_state(90.0),
        reason=TerminationReason.SOLVED,
        solved=True,
    )
    failed = _episode_row(
        initial=_state(140.0),
        final=_state(140.0),
        reason=TerminationReason.POWER_FLOW_FAILED,
        utility=-1.0,
    )

    frame = pd.DataFrame([improved, solved, failed])
    metrics = build_evaluation_metrics(
        frame,
        failed_results=[],
        requested_scenarios=3,
    )

    valid_delta = pd.to_numeric(frame["delta_J"], errors="coerce")
    assert metrics["topology_quality_count"] == 2
    assert metrics["topology_improved_count"] == 2
    assert metrics["topology_improved_rate"] == pytest.approx(1.0)
    assert metrics["avg_J0"] == pytest.approx(frame["J0"].mean())
    assert metrics["avg_Jfinal"] == pytest.approx(frame["Jfinal"].mean())
    assert metrics["avg_delta_J"] == pytest.approx(valid_delta.mean())
    assert metrics["avg_relative_J_improvement"] == pytest.approx(
        frame["relative_J_improvement"].mean()
    )
    assert metrics["avg_final_topology_utility"] == pytest.approx(
        frame["final_topology_utility"].mean()
    )


class _ResetBackend:
    def __init__(self, state: GridFMState) -> None:
        self.state = state

    def run_power_flow(self, **_kwargs):
        return SimpleNamespace(
            success=True,
            next_state=self.state,
            failure_kind=None,
            message="ok",
        )


def test_environment_preserves_canonical_initial_state_for_evaluation() -> None:
    initial = _state(150.0)
    env = TopologySwitchingEnv(
        adapter=object(),
        backend=_ResetBackend(initial),
        action_space=object(),
        reward_fn=object(),
    )

    state = env.reset(1)

    assert state is initial
    assert env.initial_state is initial
    assert env.clone().initial_state is initial

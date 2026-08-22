from __future__ import annotations

import numpy as np
import pytest

from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.physics.objective import RedispatchStatus, TerminalOutcomeEvidence
from grid_topology_ai.physics.objective import assess_physical_state
from grid_topology_ai.physics.utility import state_security_penalty, state_utility
from grid_topology_ai.value_targets import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
    terminal_utility_from_outcome,
)
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows


def _state(
    *,
    loading: float = 90.0,
    voltage_violation: float = 0.0,
    generator_p_violation: float = 0.0,
    generator_q_violation: float = 0.0,
    angle_violation: float = 0.0,
) -> GridFMState:
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0
    branch_features[
        0,
        BRANCH_FEATURE_COLUMNS.index("loading_percent"),
    ] = float(loading)

    overloaded = int(loading > 100.0)
    hard_overloaded = int(loading > 120.0)

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
            "num_hard_overloaded_branches": hard_overloaded,
            "total_thermal_overload_mva": max(float(loading) - 100.0, 0.0),
            "num_low_voltage_buses": int(voltage_violation > 0.0),
            "num_high_voltage_buses": 0,
            "total_voltage_violation": float(voltage_violation),
            "num_generator_p_violations": int(generator_p_violation > 0.0),
            "total_generator_p_violation_mw": float(generator_p_violation),
            "num_generator_q_violations": int(generator_q_violation > 0.0),
            "total_generator_q_violation_mvar": float(generator_q_violation),
            "num_angle_difference_violations": int(angle_violation > 0.0),
            "total_angle_difference_violation_degrees": float(angle_violation),
        },
        outaged_branch_ids=[],
    )


def _evidence(
    state: GridFMState,
    reason: TerminationReason,
    *,
    redispatch_status: RedispatchStatus,
    redispatch_assessment=None,
) -> TerminalOutcomeEvidence:
    assessment = assess_physical_state(state.metrics)
    return TerminalOutcomeEvidence(
        solved=reason is TerminationReason.SOLVED,
        termination_reason=reason,
        assessment=assessment,
        redispatch_status=redispatch_status,
        topology_utility=state_utility(state),
        redispatch_assessment=redispatch_assessment,
    )


def test_strictly_solved_topology_has_exact_unit_utility() -> None:
    state = _state()
    assessment = assess_physical_state(state.metrics)

    assert assessment.physically_secure is True
    assert state_security_penalty(state) == pytest.approx(0.0)
    assert state_utility(state) == pytest.approx(1.0)

    evidence = _evidence(
        state,
        TerminationReason.SOLVED,
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
    )
    assert terminal_utility_from_outcome(
        True,
        TerminationReason.SOLVED,
        evidence=evidence,
    )[0] == pytest.approx(1.0)


def test_better_unsolved_topology_receives_higher_utility() -> None:
    better = _state(loading=108.0)
    worse = _state(loading=140.0)

    better_utility = state_utility(better)
    worse_utility = state_utility(worse)
    assert -1.0 < worse_utility < better_utility < 1.0

    better_evidence = _evidence(
        better,
        TerminationReason.MAX_STEPS_REACHED,
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
    )
    worse_evidence = _evidence(
        worse,
        TerminationReason.MAX_STEPS_REACHED,
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
    )

    assert terminal_utility_from_outcome(
        False,
        TerminationReason.MAX_STEPS_REACHED,
        evidence=better_evidence,
    )[0] > terminal_utility_from_outcome(
        False,
        TerminationReason.MAX_STEPS_REACHED,
        evidence=worse_evidence,
    )[0]


def test_primary_topology_utility_is_independent_of_redispatch_result() -> None:
    topology = _state(loading=110.0)
    topology_utility = state_utility(topology)
    assessment = assess_physical_state(topology.metrics)
    redispatch_assessment = assess_physical_state(_state().metrics)

    handoff = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        assessment=assessment,
        redispatch_status=RedispatchStatus.REQUESTED,
        topology_utility=topology_utility,
    )
    validated = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.REDISPATCH_VALIDATED,
        assessment=assessment,
        redispatch_status=RedispatchStatus.VALIDATED,
        topology_utility=topology_utility,
        redispatch_assessment=redispatch_assessment,
    )

    handoff_value, _ = terminal_utility_from_outcome(
        False,
        handoff.termination_reason,
        evidence=handoff,
    )
    validated_value, _ = terminal_utility_from_outcome(
        False,
        validated.termination_reason,
        evidence=validated,
    )

    assert handoff_value == pytest.approx(topology_utility)
    assert validated_value == pytest.approx(topology_utility)


def test_multistep_rows_all_receive_final_topology_utility() -> None:
    final_state = _state(loading=108.0)
    evidence = _evidence(
        final_state,
        TerminationReason.MAX_STEPS_REACHED,
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
    )
    evidence_json = evidence.to_json()

    rows = [
        {
            "run_id": "run-1",
            "iteration": 1,
            "episode_id": "episode-1",
            "scenario_id": 1,
            "step": step,
            "step_reward": step_reward,
            "solved": False,
            "done": True,
            "termination_reason": TerminationReason.MAX_STEPS_REACHED.value,
            "terminal_outcome_evidence_json": evidence_json,
        }
        for step, step_reward in enumerate((-25.0, 60.0, 15.0))
    ]

    add_outcome_value_targets_to_rows(
        rows,
        gamma=TERMINAL_UTILITY_GAMMA,
    )

    expected = state_utility(final_state)
    assert [row["outcome_steps_to_terminal"] for row in rows] == [3, 2, 1]
    assert all(
        row["outcome_value_target"] == pytest.approx(expected)
        for row in rows
    )
    assert all(
        row["outcome_value_target_mode"] == VALUE_TARGET_MODE
        for row in rows
    )
    assert all(
        "outcome_value_target_contract_version" not in row
        for row in rows
    )
    assert rows[0]["step_reward"] < 0.0


@pytest.mark.parametrize(
    "state",
    [
        _state(loading=110.0),
        _state(voltage_violation=0.02),
        _state(generator_p_violation=2.0),
        _state(generator_q_violation=2.0),
        _state(angle_violation=5.0),
    ],
)
def test_any_strict_physical_violation_prevents_unit_utility(
    state: GridFMState,
) -> None:
    assert state_security_penalty(state) > 0.0
    assert state_utility(state) < 1.0

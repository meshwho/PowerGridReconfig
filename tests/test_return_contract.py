from __future__ import annotations

import numpy as np
import pytest

from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.value_targets import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
    heuristic_terminal_utility_estimate,
    require_bounded_utility,
    require_discount_factor,
    topology_utility_from_evidence,
)
from grid_topology_ai.physics.objective import RedispatchStatus
from grid_topology_ai.physics.utility import require_reward_discount_factor
from grid_topology_ai.termination import (
    TeacherOutcome,
    TerminationReason,
    classify_teacher_outcome,
)
from tests.outcome_evidence_helpers import terminal_evidence


def _state(*, loading: float, overloaded: int, hard: int) -> GridFMState:
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0
    branch_features[
        0,
        BRANCH_FEATURE_COLUMNS.index("loading_percent"),
    ] = loading
    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=np.zeros((1, 1), dtype=np.float32),
        branch_features=branch_features,
        edge_index=np.zeros((2, 1), dtype=np.int64),
        branch_ids=np.array([1], dtype=np.int64),
        branch_status=np.array([1.0], dtype=np.float32),
        metrics={
            "max_loading_percent": float(loading),
            "num_overloaded_branches": int(overloaded),
            "num_hard_overloaded_branches": int(hard),
            "total_voltage_violation": 0.0,
        },
        outaged_branch_ids=[],
    )


def test_teacher_outcome_is_separate_from_diagnostic_reason() -> None:
    assert classify_teacher_outcome(
        topology_solved=True,
        redispatch_validated=False,
    ) is TeacherOutcome.SOLVED

    for reason in (
        TerminationReason.HANDOFF_TO_REDISPATCH,
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        TerminationReason.MAX_STEPS_REACHED,
    ):
        evidence = terminal_evidence(reason, topology_utility=-0.4)
        outcome = classify_teacher_outcome(
            topology_solved=evidence.solved,
            redispatch_validated=(
                evidence.redispatch_status is RedispatchStatus.VALIDATED
            ),
        )
        assert outcome is TeacherOutcome.MAX_STEPS_REACHED
        assert topology_utility_from_evidence(evidence) == pytest.approx(-0.4)

    power_flow_failure = terminal_evidence(TerminationReason.POWER_FLOW_FAILED)
    assert classify_teacher_outcome(
        topology_solved=False,
        redispatch_validated=False,
    ) is TeacherOutcome.MAX_STEPS_REACHED
    assert topology_utility_from_evidence(power_flow_failure) == pytest.approx(-1.0)

    validated = terminal_evidence(
        TerminationReason.REDISPATCH_VALIDATED,
        topology_utility=0.25,
        redispatch_status=RedispatchStatus.VALIDATED,
    )
    assert classify_teacher_outcome(
        topology_solved=validated.solved,
        redispatch_validated=True,
    ) is TeacherOutcome.REDISPATCH
    assert topology_utility_from_evidence(validated) == pytest.approx(0.25)


def test_policy_value_semantics_are_undiscounted() -> None:
    assert TERMINAL_UTILITY_GAMMA == 1.0
    assert VALUE_TARGET_MODE == "final_topology_state_utility"
    assert require_discount_factor(0.5) == 1.0
    assert require_reward_discount_factor(0.5) == 0.5


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_discount_factor_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError):
        require_discount_factor(value)
    with pytest.raises(ValueError):
        require_reward_discount_factor(value)


def test_bounded_utility_rejects_mixed_return_scale() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        require_bounded_utility(50.0, context="neural value")


def test_heuristic_utility_is_bounded_and_monotonic() -> None:
    secure = heuristic_terminal_utility_estimate(
        _state(loading=90.0, overloaded=0, hard=0)
    )
    overloaded = heuristic_terminal_utility_estimate(
        _state(loading=110.0, overloaded=1, hard=0)
    )
    hard = heuristic_terminal_utility_estimate(
        _state(loading=140.0, overloaded=1, hard=1)
    )

    assert secure == 1.0
    assert -1.0 <= hard < overloaded < secure <= 1.0

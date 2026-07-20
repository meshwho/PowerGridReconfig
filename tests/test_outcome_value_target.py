import math

import pytest
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.value_targets import (
    add_outcome_value_targets_to_rows,
    terminal_value_from_outcome,
)


def _current(rows):
    for row in rows:
        row["physical_objective_schema_version"] = (
            PHYSICAL_OBJECTIVE_SCHEMA_VERSION
        )
    return rows


def test_outcome_value_target_solved_episode():
    rows = [
        {
            "scenario_id": 1,
            "step": 0,
            "solved": True,
            "done": True,
            "termination_reason": TerminationReason.SOLVED.value,
        },
        {
            "scenario_id": 1,
            "step": 1,
            "solved": True,
            "done": True,
            "termination_reason": TerminationReason.SOLVED.value,
        },
        {
            "scenario_id": 1,
            "step": 2,
            "solved": True,
            "done": True,
            "termination_reason": TerminationReason.SOLVED.value,
        },
    ]

    add_outcome_value_targets_to_rows(_current(rows), gamma=0.99)

    assert rows[0]["outcome_class"] == "solved"
    assert math.isclose(rows[0]["outcome_value_target"], 0.99**3)
    assert math.isclose(rows[1]["outcome_value_target"], 0.99**2)
    assert math.isclose(rows[2]["outcome_value_target"], 0.99)


def test_outcome_value_target_handoff_episode():
    rows = [
        {
            "scenario_id": 2,
            "step": 0,
            "solved": False,
            "done": True,
            "termination_reason": TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value,
        },
        {
            "scenario_id": 2,
            "step": 1,
            "solved": False,
            "done": True,
            "termination_reason": TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value,
        },
    ]

    add_outcome_value_targets_to_rows(_current(rows), gamma=0.95)

    assert rows[0]["outcome_class"] == "handoff_to_redispatch"
    assert rows[0]["outcome_value_target"] == 0.0
    assert rows[1]["outcome_value_target"] == 0.0


def test_outcome_value_target_max_steps_episode():
    rows = [
        {
            "scenario_id": 3,
            "step": 0,
            "solved": False,
            "done": True,
            "termination_reason": TerminationReason.MAX_STEPS_REACHED.value,
        },
        {
            "scenario_id": 3,
            "step": 1,
            "solved": False,
            "done": True,
            "termination_reason": TerminationReason.MAX_STEPS_REACHED.value,
        },
    ]

    add_outcome_value_targets_to_rows(_current(rows), gamma=0.95)

    assert rows[0]["outcome_class"] == "max_steps_reached"
    assert math.isclose(rows[0]["outcome_value_target"], -(0.95**2))
    assert math.isclose(rows[1]["outcome_value_target"], -0.95)


def test_outcome_value_target_teacher_depth_limit_is_negative():
    rows = [
        {
            "scenario_id": 4,
            "step": 0,
            "solved": False,
            "done": True,
            "termination_reason": TerminationReason.TEACHER_DEPTH_LIMIT.value,
        },
    ]

    add_outcome_value_targets_to_rows(_current(rows), gamma=0.99)

    assert rows[0]["outcome_class"] == "teacher_depth_limit"
    assert rows[0]["outcome_steps_to_terminal"] == 1
    assert rows[0]["outcome_value_target_mode"] == "alphazero_discounted"
    assert rows[0]["outcome_value_target"] == pytest.approx(-0.99)

def test_terminal_value_from_outcome_public_helper():
    assert terminal_value_from_outcome(
        solved=True,
        termination_reason="solved",
    ) == (1.0, "solved")

    assert terminal_value_from_outcome(
        solved=False,
        termination_reason="handoff_to_redispatch",
    ) == (0.0, "handoff_to_redispatch")


def test_contradictory_solved_and_reason_are_rejected():
    with pytest.raises(ValueError, match="Contradictory outcome"):
        terminal_value_from_outcome(
            solved=False,
            termination_reason="solved",
        )


def test_unknown_reason_is_rejected():
    with pytest.raises(ValueError, match="Unknown termination_reason"):
        terminal_value_from_outcome(
            solved=False,
            termination_reason="reason_17",
        )

    assert terminal_value_from_outcome(
        solved=False,
        termination_reason="max_steps_reached",
    ) == (-1.0, "max_steps_reached")


def test_outcome_targets_reject_unfinished_episode_without_mutating_rows():
    rows = _current(
        [
            {
                "scenario_id": 5,
                "step": 0,
                "solved": False,
                "done": False,
                "termination_reason": TerminationReason.MAX_STEPS_REACHED.value,
            }
        ]
    )

    with pytest.raises(ValueError, match=r"group \(5,\).*not done"):
        add_outcome_value_targets_to_rows(rows, gamma=0.95)

    assert "outcome_value_target" not in rows[0]


def test_outcome_targets_reject_mixed_episode_outcomes():
    rows = _current(
        [
            {
                "scenario_id": 6,
                "step": 0,
                "solved": False,
                "done": True,
                "termination_reason": TerminationReason.MAX_STEPS_REACHED.value,
            },
            {
                "scenario_id": 6,
                "step": 1,
                "solved": True,
                "done": True,
                "termination_reason": TerminationReason.SOLVED.value,
            },
        ]
    )

    with pytest.raises(ValueError, match=r"group \(6,\).*differ"):
        add_outcome_value_targets_to_rows(rows, gamma=0.95)


@pytest.mark.parametrize("gamma", [float("nan"), float("inf"), True])
def test_outcome_targets_reject_non_finite_or_boolean_gamma(gamma):
    with pytest.raises(ValueError, match="finite number"):
        add_outcome_value_targets_to_rows([], gamma=gamma)

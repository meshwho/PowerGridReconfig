import math
import pytest
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows


def test_outcome_value_target_solved_episode():
    rows = [
        {
            "scenario_id": 1,
            "step": 0,
            "solved": False,
            "done": False,
            "termination_reason": None,
        },
        {
            "scenario_id": 1,
            "step": 1,
            "solved": False,
            "done": False,
            "termination_reason": None,
        },
        {
            "scenario_id": 1,
            "step": 2,
            "solved": True,
            "done": True,
            "termination_reason": "solved",
        },
    ]

    add_outcome_value_targets_to_rows(rows, gamma=0.99)

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
            "done": False,
            "termination_reason": None,
        },
        {
            "scenario_id": 2,
            "step": 1,
            "solved": False,
            "done": True,
            "termination_reason": "handoff_to_redispatch_teacher",
        },
    ]

    add_outcome_value_targets_to_rows(rows, gamma=0.95)

    assert rows[0]["outcome_class"] == "handoff_to_redispatch_teacher"
    assert rows[0]["outcome_value_target"] == 0.0
    assert rows[1]["outcome_value_target"] == 0.0


def test_outcome_value_target_max_steps_episode():
    rows = [
        {
            "scenario_id": 3,
            "step": 0,
            "solved": False,
            "done": False,
            "termination_reason": None,
        },
        {
            "scenario_id": 3,
            "step": 1,
            "solved": False,
            "done": True,
            "termination_reason": "max_steps_reached",
        },
    ]

    add_outcome_value_targets_to_rows(rows, gamma=0.95)

    assert rows[0]["outcome_class"] == "max_steps_reached"
    assert math.isclose(rows[0]["outcome_value_target"], -(0.95**2))
    assert math.isclose(rows[1]["outcome_value_target"], -0.95)


def test_outcome_value_target_teacher_depth_limit_is_negative():
    rows = [
        {
            "scenario_id": 4,
            "step": 0,
            "solved": False,
            "done": False,
            "termination_reason": "teacher_depth_limit",
        },
    ]

    add_outcome_value_targets_to_rows(rows, gamma=0.99)

    assert rows[0]["outcome_class"] == "teacher_depth_limit"
    assert rows[0]["outcome_steps_to_terminal"] == 1
    assert rows[0]["outcome_value_target_mode"] == "alphazero_discounted"
    assert rows[0]["outcome_value_target"] == pytest.approx(-0.99)

from grid_topology_ai.value_targets import terminal_value_from_outcome


def test_terminal_value_from_outcome_public_helper():
    assert terminal_value_from_outcome(
        solved=True,
        termination_reason="solved",
    ) == (1.0, "solved")

    assert terminal_value_from_outcome(
        solved=False,
        termination_reason="handoff_to_redispatch",
    ) == (0.0, "handoff_to_redispatch")

    assert terminal_value_from_outcome(
        solved=False,
        termination_reason="max_steps_reached",
    ) == (-1.0, "max_steps_reached")
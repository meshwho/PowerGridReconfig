from __future__ import annotations

import numpy as np
import pytest

from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.return_contract import (
    VALUE_TARGET_MODE,
    discounted_terminal_utility,
    heuristic_terminal_utility_estimate,
    require_bounded_utility,
    require_discount_factor,
    terminal_utility_from_outcome,
)
from grid_topology_ai.termination import TerminationReason


def _state(*, loading: float, overloaded: int, hard: int) -> GridFMState:
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = loading
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


def test_terminal_utility_contract_distinguishes_safe_and_unsafe_handoffs() -> None:
    assert terminal_utility_from_outcome(True, TerminationReason.SOLVED) == (
        1.0,
        "solved",
    )
    assert terminal_utility_from_outcome(
        False,
        TerminationReason.HANDOFF_TO_REDISPATCH,
    ) == (0.0, "handoff_to_redispatch")
    assert terminal_utility_from_outcome(
        False,
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
    ) == (0.0, "handoff_to_redispatch")
    assert terminal_utility_from_outcome(
        False,
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
    ) == (-1.0, "handoff_to_redispatch_with_hard_overload")
    assert terminal_utility_from_outcome(
        False,
        TerminationReason.POWER_FLOW_FAILED,
    ) == (-1.0, "power_flow_failed")


def test_discounted_terminal_utility_counts_exact_transitions() -> None:
    assert discounted_terminal_utility(
        1.0,
        steps_to_terminal=0,
        gamma=0.5,
    ) == 1.0
    assert discounted_terminal_utility(
        1.0,
        steps_to_terminal=3,
        gamma=0.5,
    ) == 0.125
    assert discounted_terminal_utility(
        -1.0,
        steps_to_terminal=2,
        gamma=0.9,
    ) == pytest.approx(-0.81)
    assert VALUE_TARGET_MODE == "alphazero_discounted"


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_discount_factor_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError):
        require_discount_factor(value)


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

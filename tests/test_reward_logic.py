from __future__ import annotations

import numpy as np
import pytest

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.grid_utility import state_potential
from grid_topology_ai.reward import GridFMReward


def _state(
    scenario_id: int = 1,
    loadings: tuple[float, ...] = (130.0,),
    vm_values: tuple[float, ...] = (1.0, 1.0),
    total_voltage_violation: float = 0.0,
) -> GridFMState:
    num_buses = len(vm_values)
    num_branches = len(loadings)
    bus_features = np.zeros(
        (num_buses, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    bus_features[:, BUS_FEATURE_COLUMNS.index("Vm")] = np.asarray(
        vm_values,
        dtype=np.float32,
    )
    branch_features = np.zeros(
        (num_branches, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = loadings
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0
    num_overloaded = sum(float(value) > 100.0 for value in loadings)
    num_hard = sum(float(value) > 120.0 for value in loadings)
    edge_index = (
        np.array([[0], [1]], dtype=np.int64)
        if num_branches == 1
        else np.vstack(
            [
                np.zeros(num_branches, dtype=np.int64),
                np.ones(num_branches, dtype=np.int64),
            ]
        )
    )
    return GridFMState(
        scenario_id=scenario_id,
        load_scenario_idx=0.0,
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        branch_ids=np.arange(num_branches, dtype=np.int64),
        branch_status=np.ones(num_branches, dtype=np.int64),
        metrics={
            "power_flow_converged": True,
            "all_values_finite": True,
            "topology_connected": True,
            "max_loading_percent": float(max(loadings, default=0.0)),
            "num_overloaded_branches": int(num_overloaded),
            "num_hard_overloaded_branches": int(num_hard),
            "total_thermal_overload_mva": float(
                sum(max(float(value) - 100.0, 0.0) for value in loadings)
            ),
            "num_low_voltage_buses": 0,
            "num_high_voltage_buses": int(total_voltage_violation > 0.0),
            "total_voltage_violation": float(total_voltage_violation),
            "num_generator_p_violations": 0,
            "total_generator_p_violation_mw": 0.0,
            "num_generator_q_violations": 0,
            "total_generator_q_violation_mvar": 0.0,
            "num_angle_difference_violations": 0,
            "total_angle_difference_violation_degrees": 0.0,
        },
        outaged_branch_ids=[],
    )


def test_reward_is_exact_potential_shaping() -> None:
    reward_fn = GridFMReward(discount_factor=0.95)
    before = _state(loadings=(130.0,))
    after = _state(loadings=(110.0,))

    result = reward_fn.compute(before, after, True, True)

    expected = 0.95 * state_potential(after) - state_potential(before)
    assert result.reward == pytest.approx(expected)
    assert result.potential_shaping == pytest.approx(expected)
    assert result.improvement == pytest.approx(
        result.before_penalty - result.after_penalty
    )
    assert result.reward_role == "diagnostic_potential_shaping"
    assert result.success is True
    assert result.done is False


def test_worse_grid_has_negative_potential_shaping() -> None:
    result = GridFMReward(discount_factor=0.95).compute(
        _state(loadings=(110.0,)),
        _state(loadings=(130.0,)),
        True,
        True,
    )

    assert result.reward < 0.0
    assert result.after_penalty > result.before_penalty


def test_solved_state_has_no_solved_bonus() -> None:
    before = _state(loadings=(130.0,))
    after = _state(loadings=(90.0,))
    result = GridFMReward(discount_factor=0.95).compute(
        before,
        after,
        True,
        True,
    )

    assert result.reward == pytest.approx(
        0.95 * state_potential(after) - state_potential(before)
    )
    assert result.done is True
    assert result.switching_penalty == 0.0


def test_power_flow_failure_adds_no_second_terminal_penalty() -> None:
    result = GridFMReward(discount_factor=0.95).compute(
        _state(loadings=(130.0,)),
        None,
        True,
        False,
    )

    assert result.reward == 0.0
    assert result.potential_shaping == 0.0
    assert result.after_potential is None
    assert result.success is False
    assert result.done is True
    assert "no non-potential failure reward" in result.message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"switching_penalty": 1.0},
        {"non_convergence_penalty": 1.0},
        {"solved_bonus": 1.0},
    ],
)
def test_non_potential_reward_terms_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="only potential-based shaping"):
        GridFMReward(**kwargs)


def test_reward_config_identifies_diagnostic_contract() -> None:
    config = GridFMReward(
        discount_factor=0.91,
        overload_limit_percent=101.0,
        hard_overload_limit_percent=125.0,
        total_overload_weight=3.0,
        hard_overload_weight=6.0,
        num_overloaded_weight=11.0,
        num_hard_overloaded_weight=31.0,
        voltage_violation_weight=600.0,
    ).config_dict()

    assert config["reward_contract"] == "potential_shaping_v1"
    assert config["reward_role"] == "diagnostic_only"
    assert config["discount_factor"] == pytest.approx(0.91)
    assert config["overload_limit_percent"] == pytest.approx(101.0)
    assert config["hard_overload_limit_percent"] == pytest.approx(125.0)
    assert config["total_overload_weight"] == pytest.approx(3.0)
    assert config["hard_overload_weight"] == pytest.approx(6.0)
    assert config["num_overloaded_weight"] == pytest.approx(11.0)
    assert config["num_hard_overloaded_weight"] == pytest.approx(31.0)
    assert config["voltage_violation_weight"] == pytest.approx(600.0)

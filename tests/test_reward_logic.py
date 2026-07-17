import numpy as np
import pytest

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.reward import GridFMReward


def _state(
    scenario_id=1,
    loadings=(130.0,),
    vm_values=(1.0, 1.0),
    total_voltage_violation=0.0,
):
    """
    Build a minimal GridFMState compatible with GridFMReward.

    Reward uses:
    - branch_features[:, loading_percent]
    - branch_features[:, br_status]
    - state.metrics
    """

    num_buses = len(vm_values)
    num_branches = len(loadings)

    bus_features = np.zeros(
        (num_buses, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )

    vm_idx = BUS_FEATURE_COLUMNS.index("Vm")
    bus_features[:, vm_idx] = np.asarray(vm_values, dtype=np.float32)

    branch_features = np.zeros(
        (num_branches, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )

    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
    status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")

    branch_features[:, loading_idx] = np.asarray(loadings, dtype=np.float32)
    branch_features[:, status_idx] = 1.0

    num_overloaded = int(sum(float(x) > 100.0 for x in loadings))
    num_hard_overloaded = int(sum(float(x) > 120.0 for x in loadings))

    if num_branches == 1:
        edge_index = np.array([[0], [1]], dtype=np.int64)
    else:
        edge_index = np.vstack(
            [
                np.zeros(num_branches, dtype=np.int64),
                np.ones(num_branches, dtype=np.int64),
            ]
        )

    return GridFMState(
        scenario_id=int(scenario_id),
        load_scenario_idx=0.0,
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        branch_ids=np.arange(num_branches, dtype=np.int64),
        branch_status=np.ones(num_branches, dtype=np.int64),
        metrics={
            "max_loading_percent": float(max(loadings) if loadings else 0.0),
            "num_overloaded_branches": num_overloaded,
            "num_hard_overloaded_branches": num_hard_overloaded,
            "total_voltage_violation": float(total_voltage_violation),
        },
        outaged_branch_ids=[],
    )


def test_reward_is_positive_when_grid_security_improves():
    reward_fn = GridFMReward(
        switching_penalty=1.0,
        solved_bonus=50.0,
    )

    before = _state(loadings=(130.0,))
    after = _state(loadings=(110.0,))

    result = reward_fn.compute(
        before_state=before,
        after_state=after,
        action_is_switching=True,
        power_flow_success=True,
    )

    assert np.isfinite(result.reward)
    assert result.reward > 0.0
    assert result.improvement > 0.0
    assert result.after_penalty < result.before_penalty
    assert result.success is True
    assert result.done is False


def test_reward_is_negative_when_grid_security_gets_worse():
    reward_fn = GridFMReward(
        switching_penalty=1.0,
        solved_bonus=50.0,
    )

    before = _state(loadings=(110.0,))
    after = _state(loadings=(130.0,))

    result = reward_fn.compute(
        before_state=before,
        after_state=after,
        action_is_switching=True,
        power_flow_success=True,
    )

    assert np.isfinite(result.reward)
    assert result.reward < 0.0
    assert result.improvement < 0.0
    assert result.after_penalty > result.before_penalty
    assert result.success is True
    assert result.done is False


def test_reward_adds_solved_bonus_and_marks_done():
    reward_fn = GridFMReward(
        switching_penalty=1.0,
        solved_bonus=50.0,
    )

    before = _state(loadings=(130.0,))
    after = _state(loadings=(90.0,))

    result = reward_fn.compute(
        before_state=before,
        after_state=after,
        action_is_switching=True,
        power_flow_success=True,
    )

    assert np.isfinite(result.reward)
    assert result.reward > 50.0
    assert result.done is True
    assert result.success is True
    assert result.after_num_overloaded == 0
    assert result.after_num_hard_overloaded == 0


def test_reward_penalizes_power_flow_failure():
    reward_fn = GridFMReward(
        non_convergence_penalty=1000.0,
    )

    before = _state(loadings=(130.0,))

    result = reward_fn.compute(
        before_state=before,
        after_state=None,
        action_is_switching=True,
        power_flow_success=False,
    )

    assert result.reward == pytest.approx(-1000.0)
    assert result.success is False
    assert result.done is True
    assert result.message == "Power flow failed after action."


def test_switching_penalty_reduces_reward():
    before = _state(loadings=(130.0,))
    after = _state(loadings=(110.0,))

    no_switching_penalty = GridFMReward(
        switching_penalty=0.0,
        solved_bonus=0.0,
    )

    with_switching_penalty = GridFMReward(
        switching_penalty=5.0,
        solved_bonus=0.0,
    )

    reward_without_penalty = no_switching_penalty.compute(
        before_state=before,
        after_state=after,
        action_is_switching=True,
        power_flow_success=True,
    )

    reward_with_penalty = with_switching_penalty.compute(
        before_state=before,
        after_state=after,
        action_is_switching=True,
        power_flow_success=True,
    )

    assert reward_with_penalty.reward == pytest.approx(
        reward_without_penalty.reward - 5.0
    )


def test_voltage_violation_increases_state_penalty():
    reward_fn = GridFMReward()

    before = _state(
        loadings=(101.0,),
        total_voltage_violation=0.00,
    )

    after = _state(
        loadings=(101.0,),
        total_voltage_violation=0.05,
    )

    result = reward_fn.compute(
        before_state=before,
        after_state=after,
        action_is_switching=False,
        power_flow_success=True,
    )

    assert result.after_voltage_penalty > result.before_voltage_penalty
    assert result.after_penalty > result.before_penalty
    assert result.improvement < 0.0
    assert result.reward < 0.0
    assert result.done is False

def test_reward_weights_are_configurable():
    before = _state(
        loadings=(100.0,),
        total_voltage_violation=0.0,
    )

    after = _state(
        loadings=(100.0,),
        total_voltage_violation=0.05,
    )

    default_reward = GridFMReward(
        solved_bonus=0.0,
        voltage_violation_weight=500.0,
    )

    weaker_voltage_penalty_reward = GridFMReward(
        solved_bonus=0.0,
        voltage_violation_weight=100.0,
    )

    default_result = default_reward.compute(
        before_state=before,
        after_state=after,
        action_is_switching=False,
        power_flow_success=True,
    )

    weaker_result = weaker_voltage_penalty_reward.compute(
        before_state=before,
        after_state=after,
        action_is_switching=False,
        power_flow_success=True,
    )

    assert default_result.after_penalty == pytest.approx(25.0)
    assert weaker_result.after_penalty == pytest.approx(5.0)
    assert weaker_result.reward > default_result.reward

def test_reward_config_dict_contains_all_weights():
    reward_fn = GridFMReward(
        overload_limit_percent=101.0,
        hard_overload_limit_percent=125.0,
        switching_penalty=2.0,
        non_convergence_penalty=900.0,
        solved_bonus=40.0,
        total_overload_weight=3.0,
        hard_overload_weight=6.0,
        num_overloaded_weight=11.0,
        num_hard_overloaded_weight=31.0,
        voltage_violation_weight=600.0,
    )

    config = reward_fn.config_dict()

    assert config["overload_limit_percent"] == pytest.approx(101.0)
    assert config["hard_overload_limit_percent"] == pytest.approx(125.0)
    assert config["switching_penalty"] == pytest.approx(2.0)
    assert config["non_convergence_penalty"] == pytest.approx(900.0)
    assert config["solved_bonus"] == pytest.approx(40.0)
    assert config["total_overload_weight"] == pytest.approx(3.0)
    assert config["hard_overload_weight"] == pytest.approx(6.0)
    assert config["num_overloaded_weight"] == pytest.approx(11.0)
    assert config["num_hard_overloaded_weight"] == pytest.approx(31.0)
    assert config["voltage_violation_weight"] == pytest.approx(600.0)

def test_done_remains_true_for_thermal_solved_state():
    reward_fn = GridFMReward(switching_penalty=1.0, solved_bonus=50.0)
    result = reward_fn.compute(_state(loadings=(130.0,)), _state(loadings=(99.0,)), True, True)
    assert result.done is True


def test_done_remains_false_for_soft_overload_state():
    reward_fn = GridFMReward(switching_penalty=1.0, solved_bonus=50.0)
    result = reward_fn.compute(_state(loadings=(130.0,)), _state(loadings=(110.0,)), True, True)
    assert result.done is False


def test_voltage_violation_without_thermal_overload_is_not_done():
    reward_fn = GridFMReward(switching_penalty=1.0, solved_bonus=50.0)
    result = reward_fn.compute(
        _state(loadings=(130.0,)),
        _state(loadings=(90.0,), total_voltage_violation=0.5),
        True,
        True,
    )
    assert result.done is False
    assert result.reward == pytest.approx(-101.0)


def test_existing_reward_fixture_numerical_value_is_unchanged():
    reward_fn = GridFMReward(switching_penalty=1.0, solved_bonus=50.0)
    result = reward_fn.compute(
        _state(loadings=(130.0,)),
        _state(loadings=(90.0,)),
        True,
        True,
    )
    assert result.reward == pytest.approx(199.0)

import json
import math

import pytest

from grid_topology_ai.physical_objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
    assess_physical_state,
    classify_stop_outcome,
    physical_objective_contract,
    stop_allowed_for_policy,
)


def _metrics(**overrides):
    metrics = {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": 99.0,
        "num_overloaded_branches": 0,
        "num_hard_overloaded_branches": 0,
        "total_thermal_overload_mva": 0.0,
        "num_low_voltage_buses": 0,
        "num_high_voltage_buses": 0,
        "total_voltage_violation": 0.0,
        "num_generator_p_violations": 0,
        "total_generator_p_violation_mw": 0.0,
        "num_generator_q_violations": 0,
        "total_generator_q_violation_mvar": 0.0,
        "num_angle_difference_violations": 0,
        "total_angle_difference_violation_degrees": 0.0,
    }
    metrics.update(overrides)
    if (
        float(metrics["total_voltage_violation"]) > 0.0
        and "num_low_voltage_buses" not in overrides
        and "num_high_voltage_buses" not in overrides
    ):
        metrics["num_high_voltage_buses"] = 1
    return metrics


def test_contract_metadata_is_deterministic_and_json_safe():
    first = physical_objective_contract()
    second = physical_objective_contract()
    assert first == second
    assert first is not second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["schema_version"] == PHYSICAL_OBJECTIVE_SCHEMA_VERSION


def test_thermal_solved_and_voltage_feasible_is_physically_secure():
    a = assess_physical_state(_metrics())
    assert a.thermal_solved and a.voltage_feasible and a.physically_secure


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("power_flow_converged", False),
        ("all_values_finite", False),
        ("topology_connected", False),
        ("num_overloaded_branches", 1),
        ("num_low_voltage_buses", 1),
        ("num_generator_p_violations", 1),
        ("num_generator_q_violations", 1),
        ("num_angle_difference_violations", 1),
    ],
)
def test_each_physical_component_fails_closed(field, value):
    assessment = assess_physical_state(_metrics(**{field: value}))
    assert assessment.physically_secure is False


def test_thermal_solved_with_voltage_violation_is_not_physically_secure():
    a = assess_physical_state(_metrics(total_voltage_violation=1e-3))
    assert a.thermal_solved is True
    assert a.physically_secure is False


def test_soft_overload_is_safe_for_handoff_but_not_solved():
    a = assess_physical_state(_metrics(max_loading_percent=110.0, num_overloaded_branches=1))
    assert a.thermal_solved is False
    assert a.hard_overload_free is True


def test_hard_overload_is_not_hard_overload_free():
    a = assess_physical_state(_metrics(max_loading_percent=130.0, num_overloaded_branches=1, num_hard_overloaded_branches=1))
    assert a.hard_overload_free is False


def test_stop_policy_never():
    assert stop_allowed_for_policy(assess_physical_state(_metrics()), stop_policy="never") is False


def test_stop_policy_always():
    assert stop_allowed_for_policy(assess_physical_state(_metrics(num_overloaded_branches=1)), stop_policy="always") is True


def test_stop_policy_solved_only():
    assert stop_allowed_for_policy(assess_physical_state(_metrics()), stop_policy="solved_only") is True
    assert stop_allowed_for_policy(assess_physical_state(_metrics(num_overloaded_branches=1)), stop_policy="solved_only") is False


def test_stop_policy_no_hard_overloads():
    assert stop_allowed_for_policy(assess_physical_state(_metrics(num_overloaded_branches=1)), stop_policy="no_hard_overloads") is True
    assert stop_allowed_for_policy(assess_physical_state(_metrics(num_overloaded_branches=1, num_hard_overloaded_branches=1)), stop_policy="no_hard_overloads") is False


def test_include_stop_action_false_overrides_policy():
    assert stop_allowed_for_policy(assess_physical_state(_metrics()), stop_policy="always", include_stop_action=False) is False


def test_unknown_stop_policy_is_rejected():
    with pytest.raises(ValueError):
        stop_allowed_for_policy(assess_physical_state(_metrics()), stop_policy="bad")


def test_stop_outcome_solved():
    assert classify_stop_outcome(assess_physical_state(_metrics()), allow_handoff_with_hard_overloads=False).termination_reason == "solved"


def test_stop_outcome_safe_handoff():
    outcome = classify_stop_outcome(assess_physical_state(_metrics(num_overloaded_branches=1)), allow_handoff_with_hard_overloads=False)
    assert outcome.solved is False
    assert outcome.termination_reason == "handoff_to_redispatch"


def test_stop_outcome_allowed_unsafe_handoff():
    outcome = classify_stop_outcome(assess_physical_state(_metrics(num_overloaded_branches=1, num_hard_overloaded_branches=1)), allow_handoff_with_hard_overloads=True)
    assert outcome.termination_reason == "handoff_to_redispatch_with_hard_overload"


def test_stop_outcome_rejects_unsafe_stop():
    outcome = classify_stop_outcome(assess_physical_state(_metrics(num_overloaded_branches=1, num_hard_overloaded_branches=1)), allow_handoff_with_hard_overloads=False)
    assert outcome.termination_reason == "unsafe_stop_with_hard_overload"


def test_missing_metric_is_rejected():
    metrics = _metrics()
    del metrics["max_loading_percent"]
    with pytest.raises(KeyError):
        assess_physical_state(metrics)


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError):
        assess_physical_state(_metrics(num_overloaded_branches=-1))


def test_hard_count_cannot_exceed_overloaded_count():
    with pytest.raises(ValueError):
        assess_physical_state(_metrics(num_hard_overloaded_branches=1))


def test_non_finite_metric_is_rejected():
    with pytest.raises(ValueError):
        assess_physical_state(_metrics(max_loading_percent=math.inf))


def test_bool_is_not_accepted_as_numeric():
    with pytest.raises(TypeError):
        assess_physical_state(_metrics(max_loading_percent=True))


def test_integer_valued_float_count_is_accepted():
    a = assess_physical_state(
        _metrics(
            max_loading_percent=110.0,
            num_overloaded_branches=1.0,
            num_hard_overloaded_branches=0.0,
        )
    )
    assert a.num_overloaded_branches == 1
    assert a.num_hard_overloaded_branches == 0


def test_fractional_count_is_rejected():
    with pytest.raises(ValueError, match="integer-valued"):
        assess_physical_state(_metrics(num_overloaded_branches=1.5))


def test_non_finite_count_is_rejected():
    for value in (math.inf, math.nan):
        with pytest.raises(ValueError, match="finite"):
            assess_physical_state(_metrics(num_overloaded_branches=value))


def test_bool_count_is_rejected():
    with pytest.raises(TypeError, match="integer-valued"):
        assess_physical_state(_metrics(num_overloaded_branches=True))


def test_numpy_integer_count_is_accepted():
    import numpy as np

    a = assess_physical_state(
        _metrics(
            max_loading_percent=110.0,
            num_overloaded_branches=np.int64(1),
            num_hard_overloaded_branches=np.int64(0),
        )
    )
    assert a.num_overloaded_branches == 1
    assert a.num_hard_overloaded_branches == 0

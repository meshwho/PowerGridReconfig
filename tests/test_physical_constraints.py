from __future__ import annotations

import numpy as np
import pytest
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_STATUS,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    T_BUS,
)
from pypower.idx_bus import BUS_I, VA, VM, VMAX, VMIN
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
)

from grid_topology_ai.physical_constraints import (
    PhysicalNetworkArrays,
    calculate_physical_metrics,
)
from grid_topology_ai.physical_objective import assess_physical_state


def _arrays() -> PhysicalNetworkArrays:
    bus = np.zeros((2, 13), dtype=float)
    bus[:, BUS_I] = [10, 20]
    bus[:, VM] = [1.0, 1.0]
    bus[:, VA] = [0.0, -2.0]
    bus[:, VMIN] = 0.95
    bus[:, VMAX] = 1.05

    branch = np.zeros((1, QT + 1), dtype=float)
    branch[0, F_BUS] = 10
    branch[0, T_BUS] = 20
    branch[0, RATE_A] = 100.0
    branch[0, BR_STATUS] = 1.0
    branch[0, ANGMIN] = -30.0
    branch[0, ANGMAX] = 30.0
    branch[0, PF] = 50.0
    branch[0, QF] = 0.0
    branch[0, PT] = -50.0
    branch[0, QT] = 0.0

    gen = np.zeros((1, 21), dtype=float)
    gen[0, GEN_BUS] = 10
    gen[0, GEN_STATUS] = 1.0
    gen[0, PG] = 50.0
    gen[0, PMIN] = 0.0
    gen[0, PMAX] = 100.0
    gen[0, QG] = 0.0
    gen[0, QMIN] = -50.0
    gen[0, QMAX] = 50.0
    return PhysicalNetworkArrays(bus=bus, branch=branch, gen=gen)


def _assessment(
    arrays: PhysicalNetworkArrays | None = None,
    *,
    converged: bool = True,
):
    return assess_physical_state(
        calculate_physical_metrics(
            _arrays() if arrays is None else arrays,
            power_flow_converged=converged,
        )
    )


def test_fully_feasible_pypower_arrays_are_physically_secure() -> None:
    assessment = _assessment()
    assert assessment.physically_secure is True
    assert assessment.thermal_solved is True
    assert assessment.generator_p_feasible is True
    assert assessment.generator_q_feasible is True
    assert assessment.angle_difference_feasible is True


def test_power_flow_failure_is_not_secure() -> None:
    assert _assessment(converged=False).physically_secure is False


def test_non_finite_result_is_detected_before_feature_sanitization() -> None:
    arrays = _arrays()
    arrays.bus[0, VM] = np.nan
    assessment = _assessment(arrays)
    assert assessment.all_values_finite is False
    assert assessment.physically_secure is False


def test_disconnected_topology_is_not_secure() -> None:
    arrays = _arrays()
    arrays.branch[0, BR_STATUS] = 0.0
    assessment = _assessment(arrays)
    assert assessment.topology_connected is False
    assert assessment.physically_secure is False


def test_thermal_limit_violation_is_not_secure() -> None:
    arrays = _arrays()
    arrays.branch[0, PF] = 101.0
    assessment = _assessment(arrays)
    assert assessment.thermal_solved is False
    assert assessment.physically_secure is False


@pytest.mark.parametrize(("vm", "field"), [(0.94, "num_low_voltage_buses"), (1.06, "num_high_voltage_buses")])
def test_voltage_minimum_and_maximum_violations(vm: float, field: str) -> None:
    arrays = _arrays()
    arrays.bus[0, VM] = vm
    assessment = _assessment(arrays)
    assert getattr(assessment, field) == 1
    assert assessment.physically_secure is False


@pytest.mark.parametrize(("column", "value"), [(PG, -1.0), (PG, 101.0)])
def test_generator_p_minimum_and_maximum_violations(column: int, value: float) -> None:
    arrays = _arrays()
    arrays.gen[0, column] = value
    assessment = _assessment(arrays)
    assert assessment.generator_p_feasible is False
    assert assessment.physically_secure is False


@pytest.mark.parametrize("value", [-51.0, 51.0])
def test_generator_q_minimum_and_maximum_violations(value: float) -> None:
    arrays = _arrays()
    arrays.gen[0, QG] = value
    assessment = _assessment(arrays)
    assert assessment.generator_q_feasible is False
    assert assessment.physically_secure is False


@pytest.mark.parametrize("difference", [-31.0, 31.0])
def test_branch_angle_minimum_and_maximum_violations(difference: float) -> None:
    arrays = _arrays()
    arrays.bus[0, VA] = difference
    arrays.bus[1, VA] = 0.0
    assessment = _assessment(arrays)
    assert assessment.angle_difference_feasible is False
    assert assessment.physically_secure is False


def test_exact_limits_are_feasible_with_named_tolerances() -> None:
    arrays = _arrays()
    arrays.branch[0, PF] = arrays.branch[0, RATE_A]
    arrays.bus[0, VM] = arrays.bus[0, VMIN]
    arrays.bus[1, VM] = arrays.bus[1, VMAX]
    arrays.gen[0, PG] = arrays.gen[0, PMAX]
    arrays.gen[0, QG] = arrays.gen[0, QMIN]
    arrays.bus[0, VA] = arrays.branch[0, ANGMAX]
    arrays.bus[1, VA] = 0.0
    assert _assessment(arrays).physically_secure is True


def test_disabled_elements_do_not_create_false_limit_violations() -> None:
    arrays = _arrays()
    arrays.branch[0, BR_STATUS] = 0.0
    arrays.branch[0, PF] = np.nan
    arrays.gen[0, GEN_STATUS] = 0.0
    arrays.gen[0, PG] = np.nan
    assessment = _assessment(arrays)
    assert assessment.thermal_solved is True
    assert assessment.generator_p_feasible is True
    assert assessment.angle_difference_feasible is True


def test_rate_a_zero_is_unconstrained_per_matpower_semantics() -> None:
    arrays = _arrays()
    arrays.branch[0, RATE_A] = 0.0
    arrays.branch[0, PF] = 10_000.0
    assessment = _assessment(arrays)
    assert assessment.thermal_solved is True


def test_angle_check_maps_bus_ids_instead_of_using_them_as_positions() -> None:
    arrays = _arrays()
    arrays.bus[:] = arrays.bus[::-1]
    arrays.bus[0, VA] = -5.0  # bus id 20
    arrays.bus[1, VA] = 5.0   # bus id 10
    assessment = _assessment(arrays)
    assert assessment.angle_difference_feasible is True


def test_active_generator_with_unknown_bus_fails_closed() -> None:
    arrays = _arrays()
    arrays.gen[0, GEN_BUS] = 999
    assessment = _assessment(arrays)
    assert assessment.generator_p_feasible is False
    assert assessment.generator_q_feasible is False
    assert assessment.physically_secure is False

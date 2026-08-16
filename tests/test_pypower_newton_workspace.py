from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from pypower.api import case9, case118, ppoption, runpf as stock_runpf
from pypower.bustypes import bustypes
from pypower.dSbus_dV import dSbus_dV as stock_dSbus_dV
from pypower.ext2int import ext2int
from pypower.idx_brch import BR_STATUS, SHIFT, TAP
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, QG, QMAX, QMIN, VG
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from pypower.newtonpf import newtonpf as stock_newtonpf

from grid_topology_ai import pypower_compat as compat
from grid_topology_ai.pypower_newton_workspace import (
    PowerDerivativeWorkspace,
    newton_power_flow,
)


def _options(*, qlim: int = 0) -> dict[str, object]:
    return ppoption(
        PF_ALG=1,
        PF_MAX_IT=30,
        VERBOSE=0,
        OUT_ALL=0,
        ENFORCE_Q_LIMS=qlim,
    )


def _newton_problem(ppc: dict):
    internal = ext2int(deepcopy(ppc))
    base_mva = float(internal["baseMVA"])
    bus = np.asarray(internal["bus"], dtype=float)
    gen = np.asarray(internal["gen"], dtype=float)
    branch = np.asarray(internal["branch"], dtype=float)

    ref, pv, pq = bustypes(bus, gen)
    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)

    voltage = bus[:, VM] * np.exp(1j * np.pi / 180.0 * bus[:, VA])
    voltage_controlled = np.ones(voltage.shape)
    voltage_controlled[pq] = 0
    controlled = np.flatnonzero(voltage_controlled[gbus])
    if len(controlled):
        controlled_buses = gbus[controlled]
        voltage[controlled_buses] = (
            gen[on[controlled], VG]
            / np.abs(voltage[controlled_buses])
            * voltage[controlled_buses]
        )

    ybus, _yf, _yt = makeYbus(base_mva, bus, branch)
    sbus = makeSbus(base_mva, bus, gen)
    return ybus, sbus, voltage, ref, pv, pq


def _perturbed_voltage(voltage: np.ndarray) -> np.ndarray:
    count = len(voltage)
    magnitude = 1.0 + np.linspace(-0.015, 0.015, count)
    angle = np.linspace(-0.01, 0.01, count)
    return voltage * magnitude * np.exp(1j * angle)


def _assert_same_derivatives(ybus, voltage) -> None:
    expected_vm, expected_va = stock_dSbus_dV(ybus, voltage)
    workspace = PowerDerivativeWorkspace.from_ybus(ybus)
    actual_vm, actual_va = workspace.derivatives(voltage)

    np.testing.assert_allclose(
        actual_vm.toarray(),
        expected_vm.toarray(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual_va.toarray(),
        expected_va.toarray(),
        rtol=1e-12,
        atol=1e-12,
    )


def _force_upper_limit(
    ppc: dict,
    baseline: dict,
    gen_index: int,
    *,
    margin: float,
) -> None:
    q_limit = float(baseline["gen"][gen_index, QG]) - float(margin)
    ppc["gen"][gen_index, QMAX] = q_limit
    if ppc["gen"][gen_index, QMIN] >= q_limit:
        ppc["gen"][gen_index, QMIN] = q_limit - 1000.0


def _stock_q_limit_reference(
    ppc: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qlim: int,
) -> dict:
    def stock_iteration(working, solver_options, prepared_network):
        del prepared_network
        result, success = stock_runpf(
            deepcopy(working),
            solver_options,
            "",
            "",
        )
        return result, bool(success), None

    with monkeypatch.context() as context:
        context.setattr(compat, "_solve_q_limit_iteration", stock_iteration)
        result, success = compat.runpf(
            deepcopy(ppc),
            _options(qlim=qlim),
        )

    assert bool(success)
    return result


@pytest.mark.parametrize("case_factory", [case9, case118])
def test_sparse_power_derivatives_match_pypower(case_factory) -> None:
    ybus, _sbus, voltage, _ref, _pv, _pq = _newton_problem(case_factory())
    _assert_same_derivatives(ybus, _perturbed_voltage(voltage))


def test_sparse_power_derivatives_match_with_tap_shift_and_outage() -> None:
    ppc = case118()
    ppc["branch"][0, TAP] = 1.05
    ppc["branch"][1, SHIFT] = 2.0
    ppc["branch"][2, BR_STATUS] = 0.0

    ybus, _sbus, voltage, _ref, _pv, _pq = _newton_problem(ppc)
    _assert_same_derivatives(ybus, _perturbed_voltage(voltage))


@pytest.mark.parametrize("case_factory", [case9, case118])
def test_newton_workspace_matches_stock_iterations_and_voltage(case_factory) -> None:
    ybus, sbus, voltage, ref, pv, pq = _newton_problem(case_factory())
    options = _options()

    expected_voltage, expected_success, expected_iterations = stock_newtonpf(
        ybus,
        sbus,
        voltage.copy(),
        ref,
        pv,
        pq,
        options,
    )
    workspace = PowerDerivativeWorkspace.from_ybus(ybus)
    actual_voltage, actual_success, actual_iterations = newton_power_flow(
        ybus,
        sbus,
        voltage.copy(),
        ref,
        pv,
        pq,
        options,
        derivative_workspace=workspace,
    )

    assert bool(actual_success) == bool(expected_success)
    assert actual_iterations == expected_iterations
    np.testing.assert_allclose(
        actual_voltage,
        expected_voltage,
        rtol=1e-11,
        atol=1e-11,
    )


def test_multiple_q_limit_resolves_match_stock_newton_sequence(monkeypatch) -> None:
    ppc = case9()
    baseline, success = stock_runpf(
        deepcopy(ppc),
        _options(),
    )
    assert bool(success)

    _force_upper_limit(ppc, baseline, 1, margin=6.0)
    _force_upper_limit(ppc, baseline, 2, margin=3.0)
    expected = _stock_q_limit_reference(ppc, monkeypatch, qlim=2)

    actual, success = compat.runpf(
        deepcopy(ppc),
        _options(qlim=2),
    )

    assert bool(success)
    np.testing.assert_allclose(actual["bus"], expected["bus"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["gen"], expected["gen"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["branch"], expected["branch"], rtol=1e-10, atol=1e-10)

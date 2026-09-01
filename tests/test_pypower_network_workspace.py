from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from pypower.api import case9, case118, ppoption, runpf as stock_runpf
from pypower.idx_brch import BR_STATUS, SHIFT, TAP
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, PQ, REF, VA
from pypower.idx_gen import GEN_BUS, GEN_STATUS, QG, QMAX, QMIN

from grid_topology_ai.power_flow import solver as compat
from grid_topology_ai.power_flow import workspace
from grid_topology_ai.power_flow.workspace import solve_newton_power_flow


def _options(*, qlim: int) -> dict[str, object]:
    return ppoption(
        PF_ALG=1,
        PF_MAX_IT=30,
        VERBOSE=0,
        OUT_ALL=0,
        ENFORCE_Q_LIMS=qlim,
    )


def _stock_plain(ppc: dict) -> dict:
    result, success = stock_runpf(
        deepcopy(ppc),
        _options(qlim=0),
    )
    assert bool(success)
    return result


def _force_upper_limit(
    ppc: dict,
    baseline: dict,
    gen_index: int,
    *,
    margin: float = 1.0,
) -> float:
    q_limit = float(baseline["gen"][gen_index, QG]) - float(margin)
    ppc["gen"][gen_index, QMAX] = q_limit
    if ppc["gen"][gen_index, QMIN] >= q_limit:
        ppc["gen"][gen_index, QMIN] = q_limit - 1000.0
    return q_limit


def _bus_row(ppc: dict, bus_id: int) -> int:
    rows = np.flatnonzero(
        np.rint(ppc["bus"][:, BUS_I]).astype(np.int64) == int(bus_id)
    )
    assert len(rows) == 1
    return int(rows[0])


def _assert_same_solution(actual: dict, expected: dict) -> None:
    np.testing.assert_allclose(actual["bus"], expected["bus"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["gen"], expected["gen"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["branch"], expected["branch"], rtol=1e-10, atol=1e-10)


def _stock_q_limit_reference(ppc: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
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
            _options(qlim=1),
        )
    assert bool(success)
    return result


def test_prepared_newton_matches_stock_case118() -> None:
    ppc = case118()
    expected = _stock_plain(ppc)

    actual, success, prepared = solve_newton_power_flow(
        deepcopy(ppc),
        _options(qlim=0),
    )

    assert bool(success)
    assert prepared is not None
    _assert_same_solution(actual, expected)


def test_prepared_newton_matches_stock_with_tap_shift_and_outage() -> None:
    ppc = case118()
    ppc["branch"][0, TAP] = 1.05
    ppc["branch"][1, SHIFT] = 2.0
    ppc["branch"][2, BR_STATUS] = 0.0
    expected = _stock_plain(ppc)

    actual, success, _prepared = solve_newton_power_flow(
        deepcopy(ppc),
        _options(qlim=0),
    )

    assert bool(success)
    _assert_same_solution(actual, expected)


def test_prepared_network_rejects_changed_physical_network() -> None:
    ppc = case9()
    _first, success, prepared = solve_newton_power_flow(
        deepcopy(ppc),
        _options(qlim=0),
    )
    assert bool(success)

    changed = deepcopy(ppc)
    changed["branch"][0, TAP] = 1.05

    with pytest.raises(ValueError, match="branch data changed"):
        solve_newton_power_flow(
            changed,
            _options(qlim=0),
            prepared_network=prepared,
        )


def test_q_limit_runs_reuse_admittance_for_same_network(monkeypatch) -> None:
    first_case = case9()
    second_case = deepcopy(first_case)
    second_case["bus"][4, PD] += 1.0

    original = workspace.makeYbus
    calls = 0

    def counted_make_ybus(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(workspace, "makeYbus", counted_make_ybus)
    compat.clear_prepared_network_cache()

    first, first_success = compat.runpf(
        deepcopy(first_case),
        _options(qlim=1),
    )
    cached, cached_success = compat.runpf(
        deepcopy(second_case),
        _options(qlim=1),
    )

    assert bool(first_success)
    assert bool(cached_success)
    assert calls == 1
    assert compat.get_prepared_network_cache_info()["entries"] == 1

    compat.clear_prepared_network_cache()
    uncached, uncached_success = compat.runpf(
        deepcopy(second_case),
        _options(qlim=1),
    )

    assert bool(uncached_success)
    assert calls == 2
    _assert_same_solution(cached, uncached)
    assert not np.array_equal(first["bus"], cached["bus"])


def test_prepared_network_cache_rebuilds_for_network_change(monkeypatch) -> None:
    first_case = case9()
    changed_case = deepcopy(first_case)
    changed_case["branch"][0, TAP] = 1.05

    original = workspace.makeYbus
    calls = 0

    def counted_make_ybus(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(workspace, "makeYbus", counted_make_ybus)
    compat.clear_prepared_network_cache()

    _first, first_success = compat.runpf(
        deepcopy(first_case),
        _options(qlim=1),
    )
    _changed, changed_success = compat.runpf(
        deepcopy(changed_case),
        _options(qlim=1),
    )

    assert bool(first_success)
    assert bool(changed_success)
    assert calls == 2
    assert compat.get_prepared_network_cache_info()["entries"] == 2


def test_q_limit_sequence_builds_admittance_once(monkeypatch) -> None:
    ppc = case9()
    baseline = _stock_plain(ppc)
    _force_upper_limit(ppc, baseline, 1, margin=6.0)
    _force_upper_limit(ppc, baseline, 2, margin=3.0)

    original = workspace.makeYbus
    calls = 0

    def counted_make_ybus(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(workspace, "makeYbus", counted_make_ybus)
    compat.clear_prepared_network_cache()
    compat.reset_power_flow_workload_counters()

    result, success = compat.runpf(
        deepcopy(ppc),
        _options(qlim=2),
    )

    assert bool(success)
    assert np.isfinite(result["bus"]).all()
    assert int(compat.get_power_flow_workload_counters()["q_limit_resolves"]) >= 2
    assert calls == 1


def test_q_limited_pv_conversion_matches_stock_sequence(monkeypatch) -> None:
    ppc = case9()
    baseline = _stock_plain(ppc)
    q_limit = _force_upper_limit(ppc, baseline, 1)
    expected = _stock_q_limit_reference(ppc, monkeypatch)

    actual, success = compat.runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert bool(success)
    _assert_same_solution(actual, expected)
    assert actual["gen"][1, GEN_STATUS] == 1.0
    assert actual["gen"][1, QG] == q_limit
    gen_bus = int(actual["gen"][1, GEN_BUS])
    assert actual["bus"][_bus_row(actual, gen_bus), BUS_TYPE] == PQ


def test_q_limited_reference_replacement_matches_stock_sequence(monkeypatch) -> None:
    ppc = case9()
    baseline = _stock_plain(ppc)
    q_limit = _force_upper_limit(ppc, baseline, 0)
    original_ref_bus = int(ppc["gen"][0, GEN_BUS])
    original_ref_row = _bus_row(ppc, original_ref_bus)
    original_ref_angle = float(ppc["bus"][original_ref_row, VA])
    expected = _stock_q_limit_reference(ppc, monkeypatch)

    actual, success = compat.runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert bool(success)
    _assert_same_solution(actual, expected)
    assert actual["gen"][0, QG] == q_limit
    result_ref_row = _bus_row(actual, original_ref_bus)
    assert actual["bus"][result_ref_row, BUS_TYPE] == PQ
    assert np.count_nonzero(actual["bus"][:, BUS_TYPE] == REF) == 1
    assert np.isclose(
        actual["bus"][result_ref_row, VA],
        original_ref_angle,
        rtol=0.0,
        atol=1e-10,
    )

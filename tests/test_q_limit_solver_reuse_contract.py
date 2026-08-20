from __future__ import annotations

from copy import deepcopy

import numpy as np
from pypower.api import case9, ppoption, runpf as stock_runpf
from pypower.idx_gen import QG, QMAX, QMIN

from grid_topology_ai.power_flow import solver as compat
from grid_topology_ai.power_flow.solver import (
    get_power_flow_workload_counters,
    reset_power_flow_workload_counters,
    runpf,
)


def _options(*, qlim: int, pf_alg: int = 1) -> dict[str, object]:
    return ppoption(
        PF_ALG=pf_alg,
        PF_MAX_IT=30,
        PF_MAX_IT_FD=30,
        VERBOSE=0,
        OUT_ALL=0,
        ENFORCE_Q_LIMS=qlim,
    )


def _plain_solution(ppc: dict, *, pf_alg: int = 1) -> dict:
    result, success = stock_runpf(
        deepcopy(ppc),
        _options(qlim=0, pf_alg=pf_alg),
    )
    assert bool(success)
    return result


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


def test_newton_q_limit_path_matches_stock_when_no_limit_binds() -> None:
    ppc = case9()
    expected = _plain_solution(ppc)

    actual, success = runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert bool(success)
    np.testing.assert_allclose(actual["bus"], expected["bus"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["gen"], expected["gen"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["branch"], expected["branch"], rtol=1e-10, atol=1e-10)


def test_newton_q_limit_path_records_one_resolve() -> None:
    ppc = case9()
    baseline = _plain_solution(ppc)
    _force_upper_limit(ppc, baseline, 1, margin=1.0)
    reset_power_flow_workload_counters()

    result, success = runpf(
        deepcopy(ppc),
        _options(qlim=2),
    )

    assert bool(success)
    assert np.isfinite(result["bus"]).all()
    counters = get_power_flow_workload_counters()
    assert counters["q_limit_sequences"] == 1
    assert counters["q_limit_resolves"] == 1
    assert counters["stock_runpf_calls"] == 2
    assert counters["q_limit_resolve_histogram"] == {1: 1}


def test_newton_q_limit_path_records_multiple_resolves() -> None:
    ppc = case9()
    baseline = _plain_solution(ppc)
    _force_upper_limit(ppc, baseline, 1, margin=6.0)
    _force_upper_limit(ppc, baseline, 2, margin=3.0)
    reset_power_flow_workload_counters()

    result, success = runpf(
        deepcopy(ppc),
        _options(qlim=2),
    )

    assert bool(success)
    assert np.isfinite(result["bus"]).all()
    counters = get_power_flow_workload_counters()
    resolves = int(counters["q_limit_resolves"])
    assert resolves >= 2
    assert counters["stock_runpf_calls"] == resolves + 1
    assert counters["q_limit_resolve_histogram"] == {resolves: 1}


def test_q_limit_sequence_reuses_bus_row_mapping(monkeypatch) -> None:
    ppc = case9()
    baseline = _plain_solution(ppc)
    _force_upper_limit(ppc, baseline, 1, margin=1.0)
    original = compat._bus_rows
    calls = 0

    def counted(bus, gen):
        nonlocal calls
        calls += 1
        return original(bus, gen)

    monkeypatch.setattr(compat, "_bus_rows", counted)

    _, success = runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert bool(success)
    assert calls == 1


def test_q_limit_resolve_uses_minimal_solver_case(monkeypatch) -> None:
    ppc = case9()
    baseline = _plain_solution(ppc, pf_alg=3)
    _force_upper_limit(ppc, baseline, 1, margin=1.0)
    original = compat._runpf
    seen_keys: list[set[str]] = []

    def recorded(casedata, *args, **kwargs):
        if isinstance(casedata, dict):
            seen_keys.append(set(casedata))
        return original(casedata, *args, **kwargs)

    monkeypatch.setattr(compat, "_runpf", recorded)

    _, success = runpf(
        deepcopy(ppc),
        _options(qlim=1, pf_alg=3),
    )

    assert bool(success)
    assert len(seen_keys) >= 2
    assert "order" not in seen_keys[1]
    assert seen_keys[1] <= {"version", "baseMVA", "bus", "gen", "branch"}

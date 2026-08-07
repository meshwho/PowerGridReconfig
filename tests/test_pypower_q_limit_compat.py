from __future__ import annotations

from copy import deepcopy

import numpy as np
from pypower.api import case9, ppoption, runpf as stock_runpf
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VA, PQ, REF
from pypower.idx_gen import GEN_BUS, GEN_STATUS, QG, QMAX, QMIN

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.pypower_compat import (
    _largest_violation_only,
    runpf,
)


def _options(*, pf_alg: int = 3, qlim: int = 1) -> dict[str, object]:
    return ppoption(
        PF_ALG=pf_alg,
        PF_MAX_IT=30,
        PF_MAX_IT_FD=30,
        VERBOSE=0,
        OUT_ALL=0,
        ENFORCE_Q_LIMS=qlim,
    )


def _baseline(ppc: dict, *, pf_alg: int = 3) -> dict:
    result, success = stock_runpf(
        deepcopy(ppc),
        _options(pf_alg=pf_alg, qlim=0),
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
    q_limit = float(baseline["gen"][gen_index, QG]) - margin
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


def _remap_bus_ids(ppc: dict) -> None:
    old_ids = ppc["bus"][:, BUS_I].astype(np.int64)
    mapping = {
        int(old_id): int(old_id) * 10 + 7
        for old_id in old_ids
    }

    for row, old_id in enumerate(old_ids):
        ppc["bus"][row, BUS_I] = mapping[int(old_id)]

    for column in (F_BUS, T_BUS):
        ppc["branch"][:, column] = [
            mapping[int(bus_id)]
            for bus_id in ppc["branch"][:, column]
        ]

    ppc["gen"][:, GEN_BUS] = [
        mapping[int(bus_id)]
        for bus_id in ppc["gen"][:, GEN_BUS]
    ]


def test_q_limit_mode_matches_plain_solve_when_no_limit_binds() -> None:
    ppc = case9()
    plain = _baseline(ppc)

    result, success = runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert bool(success)
    np.testing.assert_allclose(result["bus"], plain["bus"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result["gen"], plain["gen"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result["branch"], plain["branch"], rtol=1e-10, atol=1e-10)


def test_q_limit_enforcement_converts_pv_bus_and_restores_injection() -> None:
    ppc = case9()
    baseline = _baseline(ppc)
    q_limit = _force_upper_limit(ppc, baseline, 1)
    original_load = ppc["bus"][:, [PD, QD]].copy()

    result, success = runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert bool(success)
    assert result["gen"][1, GEN_STATUS] == 1.0
    assert result["gen"][1, QG] == q_limit
    np.testing.assert_allclose(
        result["bus"][:, [PD, QD]],
        original_load,
        rtol=0.0,
        atol=1e-12,
    )

    gen_bus = int(result["gen"][1, GEN_BUS])
    assert result["bus"][_bus_row(result, gen_bus), BUS_TYPE] == PQ


def test_q_limit_enforcement_can_move_the_reference_bus() -> None:
    ppc = case9()
    baseline = _baseline(ppc, pf_alg=1)
    q_limit = _force_upper_limit(ppc, baseline, 0)

    original_ref_bus = int(ppc["gen"][0, GEN_BUS])
    original_ref_row = _bus_row(ppc, original_ref_bus)
    original_ref_angle = float(ppc["bus"][original_ref_row, VA])

    result, success = runpf(
        deepcopy(ppc),
        _options(pf_alg=1, qlim=1),
    )

    assert bool(success)
    assert result["gen"][0, GEN_STATUS] == 1.0
    assert result["gen"][0, QG] == q_limit

    result_ref_row = _bus_row(result, original_ref_bus)
    assert result["bus"][result_ref_row, BUS_TYPE] == PQ
    assert np.count_nonzero(result["bus"][:, BUS_TYPE] == REF) == 1
    assert np.isclose(
        result["bus"][result_ref_row, VA],
        original_ref_angle,
        rtol=0.0,
        atol=1e-10,
    )


def test_q_limit_infeasibility_returns_failure_instead_of_crashing() -> None:
    ppc = case9()
    baseline = _baseline(ppc)

    active = np.flatnonzero(ppc["gen"][:, GEN_STATUS] > 0)
    for gen_index in active:
        _force_upper_limit(
            ppc,
            baseline,
            int(gen_index),
        )

    result, success = runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert not bool(success)
    assert result["success"] is False
    assert np.isfinite(result["bus"]).all()
    assert np.isfinite(result["branch"]).all()
    assert np.isfinite(result["gen"]).all()


def test_q_limit_enforcement_supports_nonconsecutive_bus_ids() -> None:
    ppc = case9()
    _remap_bus_ids(ppc)
    baseline = _baseline(ppc)
    q_limit = _force_upper_limit(ppc, baseline, 1)

    result, success = runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert bool(success)
    assert result["gen"][1, GEN_STATUS] == 1.0
    assert result["gen"][1, QG] == q_limit
    assert set(result["bus"][:, BUS_I].astype(int)) == set(
        ppc["bus"][:, BUS_I].astype(int)
    )


def test_q_limit_mode_two_selects_largest_violation() -> None:
    gen = np.zeros((3, QMIN + 1), dtype=float)
    gen[:, QMAX] = [10.0, 10.0, 10.0]
    gen[:, QMIN] = [-10.0, -10.0, -10.0]
    gen[:, QG] = [12.0, 15.0, -14.0]

    upper, lower = _largest_violation_only(
        gen,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
    )

    assert upper.tolist() == [1]
    assert lower.tolist() == []


def test_backend_accepts_q_limited_result_as_physically_valid() -> None:
    ppc = case9()
    baseline = _baseline(ppc)
    q_limit = _force_upper_limit(ppc, baseline, 1)

    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=PhysicsConfig(
            pf_alg=3,
            max_iterations=30,
        ),
    )

    result, metrics = backend._solve_ppc(
        deepcopy(ppc),
        context="q-limit regression",
    )

    assert result["gen"][1, QG] == q_limit
    assert result["gen"][1, GEN_STATUS] == 1.0
    assert metrics["num_generator_q_violations"] == 0

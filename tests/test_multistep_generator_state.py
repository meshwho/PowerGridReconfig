from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.cache import exact_power_flow_fingerprint
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


def _adapter() -> SimpleNamespace:
    buses = []
    for bus_id, bus_type in ((10, "REF"), (20, "PV"), (30, "PQ")):
        row = {name: 0.0 for name in BUS_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": 1,
                "load_scenario_idx": 2.0,
                "bus": bus_id,
                "Pd": 10.0 if bus_id == 30 else 0.0,
                "Qd": 3.0 if bus_id == 30 else 0.0,
                "Vm": 1.0,
                "Va": 0.0,
                "vn_kv": 110.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
                bus_type: 1.0,
            }
        )
        buses.append(row)

    branches = []
    for idx, from_bus, to_bus in (
        (7, 10, 20),
        (9, 20, 30),
    ):
        row = {name: 0.0 for name in BRANCH_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": 1,
                "load_scenario_idx": 2.0,
                "idx": idx,
                "from_bus": from_bus,
                "to_bus": to_bus,
                "r": 0.01,
                "x": 0.1,
                "b": 0.0,
                "tap": 0.0,
                "shift": 0.0,
                "rate_a": 100.0,
                "br_status": 1.0,
                "ang_min": -30.0,
                "ang_max": 30.0,
            }
        )
        branches.append(row)

    generators = [
        {
            "scenario": 1,
            "idx": 0,
            "bus": 10,
            "p_mw": 40.0,
            "q_mvar": 5.0,
            "min_p_mw": 0.0,
            "max_p_mw": 100.0,
            "min_q_mvar": -50.0,
            "max_q_mvar": 50.0,
            "in_service": 1.0,
        },
        {
            "scenario": 1,
            "idx": 1,
            "bus": 20,
            "p_mw": 30.0,
            "q_mvar": 10.0,
            "min_p_mw": 0.0,
            "max_p_mw": 60.0,
            "min_q_mvar": -20.0,
            "max_q_mvar": 40.0,
            "in_service": 1.0,
        },
        {
            "scenario": 1,
            "idx": 2,
            "bus": 20,
            "p_mw": 10.0,
            "q_mvar": -5.0,
            "min_p_mw": 0.0,
            "max_p_mw": 30.0,
            "min_q_mvar": -20.0,
            "max_q_mvar": 20.0,
            "in_service": 1.0,
        },
    ]

    return SimpleNamespace(
        bus_df=pd.DataFrame(buses),
        branch_df=pd.DataFrame(branches),
        gen_df=pd.DataFrame(generators),
    )


def _result(ppc: dict) -> dict:
    result = copy.deepcopy(ppc)
    result["branch"] = np.pad(
        result["branch"],
        ((0, 0), (0, 4)),
    )
    return result


def _solved_state(
    backend: GridFMPowerFlowBackend,
    *,
    pg: list[float],
    qg: list[float],
):
    ppc, frames = backend._build_ppc(1, None)
    result = _result(ppc)
    result["gen"][:, PG] = pg
    result["gen"][:, QG] = qg
    result["branch"][0, [PF, QF, PT, QT]] = [35.0, 5.0, -34.0, -4.0]
    result["branch"][1, [PF, QF, PT, QT]] = [20.0, 3.0, -19.0, -2.0]
    return backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=result,
        original_frames=frames,
        physical_metrics={"marker": 1.0},
    )


def test_second_topology_step_uses_previous_generator_operating_point() -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())
    state0 = _solved_state(
        backend,
        pg=[42.0, 28.0, 12.0],
        qg=[6.0, 9.0, -4.0],
    )

    ppc1, frames1 = backend._build_ppc_from_state(
        state0,
        switched_off_branch_id=9,
    )

    np.testing.assert_allclose(ppc1["gen"][:, PG], [42.0, 28.0, 12.0])
    np.testing.assert_allclose(ppc1["gen"][:, QG], [6.0, 9.0, -4.0])

    result1 = _result(ppc1)
    result1["gen"][:, PG] = [44.0, 32.0, 8.0]
    result1["gen"][:, QG] = [7.0, 40.0, -5.0]
    result1["gen"][2, GEN_STATUS] = 0.0
    result1["branch"][0, [PF, QF, PT, QT]] = [45.0, 8.0, -44.0, -7.0]
    result1["branch"][1, [PF, QF, PT, QT]] = [0.0, 0.0, 0.0, 0.0]

    state1 = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=result1,
        previous_state=state0,
        original_frames=frames1,
        physical_metrics={"marker": 2.0},
    )

    ppc2, _frames2 = backend._build_ppc_from_state(
        state1,
        switched_off_branch_id=7,
    )

    np.testing.assert_allclose(ppc2["gen"][:, PG], [44.0, 32.0, 8.0])
    np.testing.assert_allclose(ppc2["gen"][:, QG], [7.0, 40.0, -5.0])
    np.testing.assert_array_equal(ppc2["gen"][:, GEN_STATUS], [1.0, 1.0, 0.0])

    original = backend.adapter.gen_df.sort_values("idx")
    assert not np.allclose(
        ppc2["gen"][:, QG],
        original["q_mvar"].to_numpy(dtype=float),
    )


def test_power_flow_cache_distinguishes_generator_allocations() -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())

    # The two generators on bus 20 have the same aggregate Qg in both states,
    # but different individual reactive-power operating points.
    first = _solved_state(
        backend,
        pg=[42.0, 28.0, 12.0],
        qg=[6.0, 15.0, 5.0],
    )
    second = _solved_state(
        backend,
        pg=[42.0, 28.0, 12.0],
        qg=[6.0, 5.0, 15.0],
    )

    qg_col = BUS_FEATURE_COLUMNS.index("Qg")
    np.testing.assert_allclose(
        first.bus_features[:, qg_col],
        second.bus_features[:, qg_col],
    )

    first_ppc, _ = backend._build_ppc_from_state(first)
    second_ppc, _ = backend._build_ppc_from_state(second)
    first_problem = backend._problem_from_ppc(first_ppc)
    second_problem = backend._problem_from_ppc(second_ppc)
    physics_fingerprint = backend.physics_config.fingerprint()

    assert exact_power_flow_fingerprint(
        first_problem,
        physics_fingerprint=physics_fingerprint,
    ) != exact_power_flow_fingerprint(
        second_problem,
        physics_fingerprint=physics_fingerprint,
    )

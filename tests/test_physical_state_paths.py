from types import SimpleNamespace

import numpy as np
import pandas as pd
from pypower.idx_brch import PF, PT, QF, QT

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)
from grid_topology_ai.physical_objective import assess_physical_state
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


def _adapter() -> SimpleNamespace:
    buses = []
    for bus_id in (10, 20):
        row = {name: 0.0 for name in BUS_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": 1,
                "load_scenario_idx": 0.0,
                "bus": bus_id,
                "Vm": 1.0,
                "Va": 0.0,
                "PQ": 1.0,
                "vn_kv": 110.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
            }
        )
        buses.append(row)

    branch = {name: 0.0 for name in BRANCH_FEATURE_COLUMNS}
    branch.update(
        {
            "scenario": 1,
            "load_scenario_idx": 0.0,
            "idx": 7,
            "from_bus": 10,
            "to_bus": 20,
            "r": 0.01,
            "x": 0.1,
            "rate_a": 100.0,
            "br_status": 1.0,
            "tap": 0.0,
            "shift": 0.0,
            "ang_min": -30.0,
            "ang_max": 30.0,
        }
    )
    gen = {
        "scenario": 1,
        "idx": 1,
        "bus": 10,
        "p_mw": 50.0,
        "q_mvar": 0.0,
        "min_p_mw": 0.0,
        "max_p_mw": 100.0,
        "min_q_mvar": -50.0,
        "max_q_mvar": 50.0,
        "in_service": 1.0,
    }
    return SimpleNamespace(
        bus_df=pd.DataFrame(buses),
        branch_df=pd.DataFrame([branch]),
        gen_df=pd.DataFrame([gen]),
    )


def test_slow_fast_and_cache_paths_preserve_identical_assessment() -> None:
    backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        enable_cache=True,
    )
    ppc, frames = backend._build_ppc(1, None)
    result = {
        "bus": ppc["bus"].copy(),
        "branch": np.pad(ppc["branch"].copy(), ((0, 0), (0, 4))),
        "gen": ppc["gen"].copy(),
    }
    result["branch"][0, PF] = 50.0
    result["branch"][0, QF] = 0.0
    result["branch"][0, PT] = -50.0
    result["branch"][0, QT] = 0.0

    slow = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=result,
        original_frames=frames,
    )
    fast = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=result,
        previous_state=slow,
        original_frames=frames,
    )

    cache_key = backend._make_cache_key_from_state(slow, None)
    backend._cache[cache_key] = fast
    cached = backend.run_power_flow_from_state(slow, None)

    assert cached.success is True
    assert cached.next_state is fast
    assert assess_physical_state(slow.metrics) == assess_physical_state(fast.metrics)
    assert assess_physical_state(cached.next_state.metrics) == assess_physical_state(
        fast.metrics
    )

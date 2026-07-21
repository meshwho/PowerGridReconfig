from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pypower.idx_brch import PF, PT, QF, QT, RATE_A

from grid_topology_ai.config.physics import (
    PhysicsConfig,
    ZeroRateAPolicy,
)
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMAdapter,
    UNRATED_LOADING_PERCENT,
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


def _completed_result(ppc: dict) -> dict:
    result = copy.deepcopy(ppc)
    result["branch"] = np.pad(
        result["branch"],
        ((0, 0), (0, 4)),
    )
    result["branch"][0, PF] = 50.0
    result["branch"][0, QF] = 0.0
    result["branch"][0, PT] = -50.0
    result["branch"][0, QT] = 0.0
    return result


def test_adapter_encodes_active_unrated_branch_with_finite_sentinel() -> None:
    frame = pd.DataFrame(
        {
            "pf": [500.0, 0.0],
            "qf": [0.0, 0.0],
            "pt": [-500.0, 0.0],
            "qt": [0.0, 0.0],
            "rate_a": [0.0, 0.0],
            "br_status": [1.0, 0.0],
        }
    )

    result = GridFMAdapter._add_branch_loading(
        frame,
        physics_config=PhysicsConfig(
            zero_rate_a_policy=ZeroRateAPolicy.UNLIMITED,
        ),
    )

    assert result.loc[0, "loading_percent"] == pytest.approx(
        UNRATED_LOADING_PERCENT
    )
    assert result.loc[0, "loading_percent"] != 0.0
    assert result.loc[1, "loading_percent"] == pytest.approx(0.0)


def test_slow_and_fast_paths_preserve_unrated_branch_semantics() -> None:
    backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        physics_config=PhysicsConfig(
            zero_rate_a_policy=ZeroRateAPolicy.UNLIMITED,
        ),
    )
    ppc, frames = backend._build_ppc(1, None)

    rated_result = _completed_result(ppc)
    previous = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=rated_result,
        original_frames=frames,
    )

    unrated_result = _completed_result(ppc)
    unrated_result["branch"][0, RATE_A] = 0.0
    unrated_result["branch"][0, PF] = 500.0
    unrated_result["branch"][0, PT] = -500.0

    slow = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=unrated_result,
        original_frames=frames,
    )
    fast = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=unrated_result,
        previous_state=previous,
        original_frames=frames,
    )

    loading_column = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    for state in (slow, fast):
        assert state.branch_features[0, loading_column] == pytest.approx(
            UNRATED_LOADING_PERCENT
        )
        assert state.branch_features[0, loading_column] != 0.0
        assert state.metrics["num_unrated_active_branches"] == 1
        assert state.metrics["mean_loading_percent"] == pytest.approx(0.0)
        assert assess_physical_state(state.metrics).thermal_solved is True

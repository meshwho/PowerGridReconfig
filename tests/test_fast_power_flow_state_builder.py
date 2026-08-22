from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pypower.idx_brch import BR_STATUS, PF, PT, QF, QT, RATE_A
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.config import PhysicsConfig, ZeroRateAPolicy
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend


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
    for idx, from_bus, to_bus, rate_a in (
        (7, 10, 20, 100.0),
        (9, 20, 30, 80.0),
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
                "rate_a": rate_a,
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


def _initial_state(backend: GridFMPowerFlowBackend):
    ppc, frames = backend._build_ppc(1, None)
    result = _result(ppc)
    result["bus"][:, VM] = [1.01, 1.00, 0.99]
    result["bus"][:, VA] = [0.0, -1.0, -2.0]
    result["gen"][:, PG] = [42.0, 28.0, 12.0]
    result["gen"][:, QG] = [6.0, 9.0, -4.0]
    result["branch"][0, [PF, QF, PT, QT]] = [35.0, 5.0, -34.0, -4.0]
    result["branch"][1, [PF, QF, PT, QT]] = [20.0, 3.0, -19.0, -2.0]
    return backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=result,
        original_frames=frames,
        physical_metrics={"marker": 1.0},
    )


def _transition(
    backend: GridFMPowerFlowBackend,
    previous,
    branch_id: int,
):
    ppc, frames = backend._build_ppc_from_state(
        previous,
        switched_off_branch_id=branch_id,
    )
    result = _result(ppc)

    result["bus"][:, VM] = [1.02, 0.985, 0.97]
    result["bus"][:, VA] = [0.0, -1.5, -3.0]
    result["gen"][:, PG] = [44.0, 32.0, 8.0]
    result["gen"][:, QG] = [7.0, 40.0, -5.0]
    result["branch"][0, [PF, QF, PT, QT]] = [45.0, 8.0, -44.0, -7.0]
    result["branch"][1, [PF, QF, PT, QT]] = [0.0, 0.0, 0.0, 0.0]

    canonical = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=result,
        original_frames=frames,
        physical_metrics={"marker": 2.0},
    )
    fast = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=result,
        previous_state=previous,
        original_frames=frames,
        physical_metrics={"marker": 2.0},
    )
    return canonical, fast, result, frames


def _assert_equivalent(canonical, fast) -> None:
    np.testing.assert_allclose(
        fast.bus_features,
        canonical.bus_features,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        fast.branch_features,
        canonical.branch_features,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(fast.branch_status, canonical.branch_status)
    np.testing.assert_array_equal(fast.edge_index, canonical.edge_index)
    np.testing.assert_array_equal(fast.branch_ids, canonical.branch_ids)
    np.testing.assert_array_equal(fast.bus_ids, canonical.bus_ids)
    assert fast.outaged_branch_ids == canonical.outaged_branch_ids
    assert fast.metrics.keys() == canonical.metrics.keys()

    for key in fast.metrics:
        fast_value = fast.metrics[key]
        canonical_value = canonical.metrics[key]
        if isinstance(fast_value, (float, np.floating)):
            assert np.isclose(
                float(fast_value),
                float(canonical_value),
                rtol=0.0,
                atol=1e-6,
            ), key
        else:
            assert fast_value == canonical_value, key


def test_fast_builder_matches_canonical_after_topology_change() -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())
    previous = _initial_state(backend)
    canonical, fast, _result_ppc, _frames = _transition(
        backend,
        previous,
        branch_id=9,
    )

    _assert_equivalent(canonical, fast)
    assert fast.outaged_branch_ids == [9]


def test_fast_builder_refreshes_generator_features_at_q_limit() -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())
    previous = _initial_state(backend)
    canonical, fast, _result_ppc, _frames = _transition(
        backend,
        previous,
        branch_id=9,
    )
    _assert_equivalent(canonical, fast)

    row = int(np.flatnonzero(fast.bus_ids == 20)[0])
    qg = BUS_FEATURE_COLUMNS.index("Qg")
    online = BUS_FEATURE_COLUMNS.index("gen_online_count")
    q_up = BUS_FEATURE_COLUMNS.index("gen_q_up_margin_mvar")
    min_q_up = BUS_FEATURE_COLUMNS.index("gen_min_q_up_margin_mvar")
    q_violations = BUS_FEATURE_COLUMNS.index("gen_q_limit_violation_count")

    assert fast.bus_features[row, qg] == 35.0
    assert fast.bus_features[row, online] == 2.0
    assert fast.bus_features[row, q_up] == 25.0
    assert fast.bus_features[row, min_q_up] == 0.0
    assert fast.bus_features[row, q_violations] == 0.0


def test_fast_builder_refreshes_generator_status_dependent_features() -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())
    previous = _initial_state(backend)
    ppc, frames = backend._build_ppc_from_state(
        previous,
        switched_off_branch_id=9,
    )
    result = _result(ppc)
    result["gen"][:, PG] = [44.0, 32.0, 1000.0]
    result["gen"][:, QG] = [7.0, 12.0, 1000.0]
    result["gen"][2, GEN_STATUS] = 0.0
    result["branch"][0, [PF, QF, PT, QT]] = [40.0, 5.0, -39.0, -4.0]
    result["branch"][1, [PF, QF, PT, QT]] = [0.0, 0.0, 0.0, 0.0]

    canonical = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=result,
        original_frames=frames,
        physical_metrics={"marker": 3.0},
    )
    fast = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=result,
        previous_state=previous,
        original_frames=frames,
        physical_metrics={"marker": 3.0},
    )
    _assert_equivalent(canonical, fast)

    row = int(np.flatnonzero(fast.bus_ids == 20)[0])
    pg = BUS_FEATURE_COLUMNS.index("Pg")
    online = BUS_FEATURE_COLUMNS.index("gen_online_count")
    assert fast.bus_features[row, pg] == 32.0
    assert fast.bus_features[row, online] == 1.0


def test_fast_builder_preserves_active_unrated_branch_semantics() -> None:
    physics = PhysicsConfig(
        zero_rate_a_policy=ZeroRateAPolicy.UNLIMITED,
    )
    adapter = _adapter()
    adapter.branch_df.loc[
        adapter.branch_df["idx"] == 7,
        "rate_a",
    ] = 0.0
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=physics,
    )
    previous = _initial_state(backend)

    ppc, frames = backend._build_ppc_from_state(
        previous,
        switched_off_branch_id=9,
    )
    result = _result(ppc)
    result["branch"][0, [PF, QF, PT, QT]] = [500.0, 0.0, -500.0, 0.0]
    result["branch"][0, RATE_A] = 0.0
    result["branch"][1, [PF, QF, PT, QT]] = [0.0, 0.0, 0.0, 0.0]

    canonical = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=result,
        original_frames=frames,
        physical_metrics={"marker": 4.0},
    )
    fast = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=result,
        previous_state=previous,
        original_frames=frames,
        physical_metrics={"marker": 4.0},
    )
    _assert_equivalent(canonical, fast)

    unlimited = BRANCH_FEATURE_COLUMNS.index("unlimited_rating")
    loading = BRANCH_FEATURE_COLUMNS.index("loading_percent")
    assert fast.branch_features[0, unlimited] == 1.0
    assert fast.branch_features[0, loading] == 0.0
    assert fast.metrics["mean_loading_percent"] == 0.0


def test_fast_builder_does_not_delegate_to_canonical_builder(monkeypatch) -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())
    previous = _initial_state(backend)
    ppc, frames = backend._build_ppc_from_state(
        previous,
        switched_off_branch_id=9,
    )
    result = _result(ppc)
    result["branch"][0, [PF, QF, PT, QT]] = [30.0, 2.0, -29.0, -1.0]
    result["branch"][1, [PF, QF, PT, QT]] = [0.0, 0.0, 0.0, 0.0]

    def fail_canonical(**_kwargs):
        raise AssertionError("canonical builder was called")

    monkeypatch.setattr(
        backend,
        "_build_canonical_state",
        fail_canonical,
    )

    fast = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=result,
        previous_state=previous,
        original_frames=frames,
        physical_metrics={"marker": 5.0},
    )
    assert fast.outaged_branch_ids == [9]


def test_fast_builder_matches_canonical_on_second_topology_step() -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())
    initial = _initial_state(backend)
    _canonical_one, state_one, _result_one, _frames_one = _transition(
        backend,
        initial,
        branch_id=9,
    )

    ppc, frames = backend._build_ppc_from_state(
        state_one,
        switched_off_branch_id=7,
    )
    result = _result(ppc)
    result["bus"][:, VM] = [1.01, 0.98, 0.965]
    result["gen"][:, PG] = [45.0, 30.0, 9.0]
    result["gen"][:, QG] = [8.0, 35.0, -4.0]
    result["branch"][:, BR_STATUS] = [0.0, 0.0]
    result["branch"][:, [PF, QF, PT, QT]] = 0.0

    canonical = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=result,
        original_frames=frames,
        physical_metrics={"marker": 6.0},
    )
    fast = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=result,
        previous_state=state_one,
        original_frames=frames,
        physical_metrics={"marker": 6.0},
    )

    _assert_equivalent(canonical, fast)
    assert fast.outaged_branch_ids == [7, 9]

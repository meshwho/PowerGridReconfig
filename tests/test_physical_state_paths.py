from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pypower.idx_brch import PF, PT, QF, QT, RATE_A
from pypower.idx_bus import VM

import grid_topology_ai.pypower_backend as backend_module
from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    ZeroRateAPolicy,
)
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMAdapter,
)
from grid_topology_ai.physical_objective import assess_physical_state
from grid_topology_ai.power_flow_errors import InvalidPhysicalState
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.state_builder import GridFMStateBuilder


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


def _real_adapter(
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
) -> GridFMAdapter:
    source = _adapter()
    adapter = object.__new__(GridFMAdapter)
    adapter.bus_df = source.bus_df.copy()
    adapter.branch_df = source.branch_df.copy()
    adapter.gen_df = source.gen_df.copy()
    adapter.physics_config = physics_config
    return adapter


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


def test_initial_slow_and_fast_states_share_one_representation() -> None:
    physics_config = PhysicsConfig(
        zero_rate_a_policy=ZeroRateAPolicy.UNLIMITED,
    )
    adapter = _real_adapter(physics_config)
    adapter.branch_df.loc[0, ["pf", "qf", "pt", "qt"]] = [
        500.0,
        0.0,
        -500.0,
        0.0,
    ]
    adapter.branch_df.loc[0, "rate_a"] = 0.0

    initial = adapter.build_state(1)
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=physics_config,
    )
    ppc, frames = backend._build_ppc(1, None)
    result = _completed_result(ppc)
    result["branch"][0, PF] = 500.0
    result["branch"][0, PT] = -500.0
    result["branch"][0, RATE_A] = 0.0

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

    for state in (initial, slow, fast):
        assert state.bus_features.shape[1] == len(BUS_FEATURE_COLUMNS)
        assert state.branch_features.shape[1] == len(BRANCH_FEATURE_COLUMNS)
        np.testing.assert_array_equal(state.bus_ids, [10, 20])
        np.testing.assert_array_equal(state.edge_index, [[0], [1]])
        np.testing.assert_array_equal(state.branch_ids, [7])
        np.testing.assert_array_equal(state.branch_status, [1.0])

    np.testing.assert_array_equal(initial.bus_features, slow.bus_features)
    np.testing.assert_array_equal(initial.branch_features, slow.branch_features)
    np.testing.assert_array_equal(slow.bus_features, fast.bus_features)
    np.testing.assert_array_equal(slow.branch_features, fast.branch_features)

    unlimited_index = BRANCH_FEATURE_COLUMNS.index("unlimited_rating")
    p_up_index = BUS_FEATURE_COLUMNS.index("gen_p_up_margin_mw")
    assert initial.branch_features[0, unlimited_index] == 1.0
    assert initial.bus_features[0, p_up_index] == 50.0


def test_canonical_builder_calculates_metrics_once_per_state() -> None:
    calls: list[str] = []

    def frame_metrics(**_kwargs: object) -> dict[str, object]:
        calls.append("frames")
        return {"metric_source": "frames"}

    def result_metrics(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append("result")
        return {"metric_source": "result"}

    source = _adapter()
    builder = GridFMStateBuilder(
        frame_metrics_calculator=frame_metrics,
        result_metrics_calculator=result_metrics,
    )
    initial = builder.build_from_frames(
        scenario_id=1,
        bus_df=source.bus_df,
        branch_df=source.branch_df,
        gen_df=source.gen_df,
        power_flow_converged=False,
    )
    assert calls == ["frames"]
    assert initial.metrics["metric_source"] == "frames"

    backend = GridFMPowerFlowBackend(adapter=source)
    ppc, frames = backend._build_ppc(1, None)
    solved = builder.build_from_pypower_result(
        scenario_id=1,
        result_ppc=_completed_result(ppc),
        original_frames=frames,
    )

    assert calls == ["frames", "result"]
    assert solved.metrics["metric_source"] == "result"


def test_fast_builder_rejects_float32_overflow_in_active_branch_features() -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())
    ppc, frames = backend._build_ppc(1, None)
    result = _completed_result(ppc)
    # Finite float64 flow cannot be represented in the float32 feature tensor.
    result["branch"][0, PF] = 1e39
    result["branch"][0, PT] = -1e39
    previous = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=_completed_result(ppc),
        original_frames=frames,
    )
    with np.testing.assert_raises(InvalidPhysicalState):
        backend._build_state_from_pypower_result_fast(
            scenario_id=1,
            result_ppc=result,
            previous_state=previous,
            original_frames=frames,
        )


def test_slow_and_fast_builders_reject_rate_a_underflow() -> None:
    backend = GridFMPowerFlowBackend(adapter=_adapter())
    ppc, frames = backend._build_ppc(1, None)

    valid_result = _completed_result(ppc)
    previous = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=valid_result,
        original_frames=frames,
    )

    damaged_result = _completed_result(ppc)
    damaged_result["branch"][0, PF] = 1e20
    damaged_result["branch"][0, PT] = -1e20
    damaged_result["branch"][0, RATE_A] = 1e-300

    damaged_frames = {
        name: frame.copy()
        for name, frame in frames.items()
    }
    damaged_frames["branch"].loc[0, "rate_a"] = 1e-300

    with np.testing.assert_raises(InvalidPhysicalState):
        backend._build_state_from_pypower_result(
            scenario_id=1,
            result_ppc=damaged_result,
            original_frames=damaged_frames,
        )

    with np.testing.assert_raises(InvalidPhysicalState):
        backend._build_state_from_pypower_result_fast(
            scenario_id=1,
            result_ppc=damaged_result,
            previous_state=previous,
            original_frames=damaged_frames,
        )


def test_initial_power_flow_rejects_non_finite_result(
    monkeypatch,
) -> None:
    backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        enable_cache=True,
    )

    def fake_runpf(ppc, _options):
        result = _completed_result(ppc)
        result["bus"][0, VM] = np.nan
        return result, True

    monkeypatch.setattr(
        backend_module,
        "runpf",
        fake_runpf,
    )

    result = backend.run_power_flow(1, None)

    assert result.success is False
    assert result.next_state is None
    assert "non-finite" in result.message


def test_state_power_flow_rejects_non_finite_result(
    monkeypatch,
) -> None:
    backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        enable_cache=True,
    )

    ppc, frames = backend._build_ppc(1, None)
    valid_result = _completed_result(ppc)

    state = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=valid_result,
        original_frames=frames,
    )

    def fake_runpf(next_ppc, _options):
        result = _completed_result(next_ppc)
        result["branch"][0, PF] = np.nan
        return result, True

    monkeypatch.setattr(
        backend_module,
        "runpf",
        fake_runpf,
    )

    result = backend.run_power_flow_from_state(
        state,
        switched_off_branch_id=None,
    )

    assert result.success is False
    assert result.next_state is None
    assert backend.cache_info()["size"] == 0

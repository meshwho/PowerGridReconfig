from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import PHYSICS_CONFIG_CONTRACT_VERSION
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.power_flow.problem import build_scenario_power_flow_template
from grid_topology_ai.runtime.numpy_scenario import (
    NumPyMemoryMappedGridFMAdapter,
    NumPyMemoryMappedGridFMPowerFlowBackend,
)
from grid_topology_ai.runtime import build_memory_mapped_teacher_context
from grid_topology_ai.runtime.scenario_store import ensure_runtime_scenario_store
from grid_topology_ai.topology_actions import GridFMAction


def _write_dataset(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "scenario": 1,
                "load_scenario_idx": 3.0,
                "bus": 1,
                "Pd": 0.0,
                "Qd": 0.0,
                "Pg": 50.0,
                "Qg": 8.0,
                "Vm": 1.02,
                "Va": 0.0,
                "PQ": 0.0,
                "PV": 0.0,
                "REF": 1.0,
                "vn_kv": 110.0,
                "GS": 0.0,
                "BS": 0.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
            },
            {
                "scenario": 1,
                "load_scenario_idx": 3.0,
                "bus": 2,
                "Pd": 45.0,
                "Qd": 12.0,
                "Pg": 0.0,
                "Qg": 0.0,
                "Vm": 0.99,
                "Va": -1.0,
                "PQ": 1.0,
                "PV": 0.0,
                "REF": 0.0,
                "vn_kv": 110.0,
                "GS": 0.0,
                "BS": 0.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
            },
        ]
    ).to_parquet(raw_dir / "bus_data.parquet", index=False)

    pd.DataFrame(
        [
            {
                "scenario": 1,
                "load_scenario_idx": 3.0,
                "idx": 10,
                "from_bus": 1,
                "to_bus": 2,
                "pf": 25.0,
                "qf": 3.0,
                "pt": -24.8,
                "qt": -2.9,
                "r": 0.01,
                "x": 0.08,
                "b": 0.02,
                "tap": 0.0,
                "shift": 0.0,
                "rate_a": 100.0,
                "br_status": 1.0,
                "ang_min": -360.0,
                "ang_max": 360.0,
            },
            {
                "scenario": 1,
                "load_scenario_idx": 3.0,
                "idx": 11,
                "from_bus": 1,
                "to_bus": 2,
                "pf": 20.0,
                "qf": 2.5,
                "pt": -19.9,
                "qt": -2.4,
                "r": 0.012,
                "x": 0.10,
                "b": 0.015,
                "tap": 1.05,
                "shift": 0.0,
                "rate_a": 0.0,
                "br_status": 1.0,
                "ang_min": -360.0,
                "ang_max": 360.0,
            },
        ]
    ).to_parquet(raw_dir / "branch_data.parquet", index=False)

    pd.DataFrame(
        [
            {
                "scenario": 1,
                "idx": 100,
                "bus": 1,
                "p_mw": 50.0,
                "q_mvar": 8.0,
                "min_p_mw": 0.0,
                "max_p_mw": 120.0,
                "min_q_mvar": -50.0,
                "max_q_mvar": 50.0,
                "in_service": 1.0,
            },
            {
                "scenario": 1,
                "idx": 101,
                "bus": 1,
                "p_mw": 0.0,
                "q_mvar": 0.0,
                "min_p_mw": 0.0,
                "max_p_mw": 80.0,
                "min_q_mvar": -30.0,
                "max_q_mvar": 30.0,
                "in_service": 0.0,
            },
        ]
    ).to_parquet(raw_dir / "gen_data.parquet", index=False)


def _assert_state_equal(left, right) -> None:
    assert int(left.scenario_id) == int(right.scenario_id)
    assert float(left.load_scenario_idx) == float(right.load_scenario_idx)
    np.testing.assert_array_equal(left.bus_features, right.bus_features)
    np.testing.assert_array_equal(left.branch_features, right.branch_features)
    np.testing.assert_array_equal(left.edge_index, right.edge_index)
    np.testing.assert_array_equal(left.branch_ids, right.branch_ids)
    np.testing.assert_array_equal(left.branch_status, right.branch_status)
    np.testing.assert_array_equal(left.bus_ids, right.bus_ids)

    left_metrics = dict(left.metrics)
    right_metrics = dict(right.metrics)
    left_mean = float(left_metrics.pop("mean_loading_percent"))
    right_mean = float(right_metrics.pop("mean_loading_percent"))
    assert left_metrics == right_metrics
    np.testing.assert_allclose(right_mean, left_mean, rtol=0.0, atol=1e-6)

    assert list(left.outaged_branch_ids) == list(right.outaged_branch_ids)


def test_numpy_runtime_matches_canonical_state_and_pf_template(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_dataset(raw_dir)
    physics = PhysicsConfig()
    canonical = GridFMAdapter(raw_dir, physics_config=physics)
    runtime = NumPyMemoryMappedGridFMAdapter(
        ensure_runtime_scenario_store(raw_dir),
        physics_config=physics,
    )

    def reject_dataframe_materialization(*args, **kwargs):
        raise AssertionError("NumPy teacher hot path materialized a DataFrame")

    monkeypatch.setattr(
        runtime.store,
        "scenario_frames",
        reject_dataframe_materialization,
    )

    _assert_state_equal(canonical.build_state(1), runtime.build_state(1))

    expected = build_scenario_power_flow_template(
        scenario_id=1,
        bus_df=canonical.bus_df[canonical.bus_df["scenario"] == 1],
        branch_df=canonical.branch_df[canonical.branch_df["scenario"] == 1],
        gen_df=canonical.gen_df[canonical.gen_df["scenario"] == 1],
        base_mva=physics.base_mva,
    )
    actual = runtime.scenario_power_flow_template(1)

    np.testing.assert_array_equal(actual.bus_ids, expected.bus_ids)
    np.testing.assert_array_equal(actual.branch_ids, expected.branch_ids)
    np.testing.assert_array_equal(actual.generator_ids, expected.generator_ids)
    np.testing.assert_array_equal(actual.bus, expected.bus)
    np.testing.assert_array_equal(actual.branch, expected.branch)
    np.testing.assert_array_equal(actual.gen, expected.gen)

    backend = NumPyMemoryMappedGridFMPowerFlowBackend(
        adapter=runtime,  # type: ignore[arg-type]
        physics_config=physics,
        enable_cache=True,
    )
    backend_template, frames = backend._scenario_problem_resources(1)
    assert backend_template is actual
    assert not isinstance(frames["bus"], pd.DataFrame)
    assert not isinstance(frames["branch"], pd.DataFrame)
    assert not isinstance(frames["gen"], pd.DataFrame)


def _teacher_context(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    _write_dataset(raw_dir)
    physics = PhysicsConfig()
    return build_memory_mapped_teacher_context(
        runtime_store_dir=ensure_runtime_scenario_store(raw_dir),
        states_dir=tmp_path / "states",
        task_config={
            "physics_config_contract_version": PHYSICS_CONFIG_CONTRACT_VERSION,
            "physics_config": physics.to_dict(),
            "physics_config_fingerprint": physics.fingerprint(),
            "disable_cache": False,
        },
        scenario_ids=[1],
    )


def test_production_teacher_builder_selects_numpy_mmap_runtime(tmp_path: Path) -> None:
    context = _teacher_context(tmp_path)

    assert isinstance(context["adapter"], NumPyMemoryMappedGridFMAdapter)
    assert isinstance(context["backend"], NumPyMemoryMappedGridFMPowerFlowBackend)


def test_numpy_teacher_backend_reuses_identical_pf_from_l1(tmp_path: Path) -> None:
    context = _teacher_context(tmp_path)
    backend = context["backend"]
    state = context["adapter"].build_state(1)
    action = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
    )

    first = backend.run_power_flow_from_state(state, action=action)
    second = backend.run_power_flow_from_state(state, action=action)

    assert first.success and second.success
    assert backend.cache_info()["misses"] == 1
    assert backend.cache_info()["hits"] == 1
    np.testing.assert_array_equal(
        first.next_state.bus_features,
        second.next_state.bus_features,
    )
    np.testing.assert_array_equal(
        first.next_state.branch_features,
        second.next_state.branch_features,
    )

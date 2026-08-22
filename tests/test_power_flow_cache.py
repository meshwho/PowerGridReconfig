from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pypower.idx_bus import VM

from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, BUS_FEATURE_COLUMNS
from grid_topology_ai.power_flow.backend import (
    GridFMPowerFlowBackend,
    _GeneratorOperatingPointState,
)
from grid_topology_ai.topology_actions import GridFMAction


def _adapter() -> SimpleNamespace:
    buses = []
    branches = []
    generators = []
    for scenario_id in (1, 2):
        for bus_id, bus_type in ((10, "REF"), (20, "PQ")):
            row = {name: 0.0 for name in BUS_FEATURE_COLUMNS}
            row.update(
                {
                    "scenario": scenario_id,
                    "bus": bus_id,
                    "Pd": 35.0 if bus_id == 20 else 0.0,
                    "Qd": 8.0 if bus_id == 20 else 0.0,
                    "Vm": 1.0,
                    "Va": 0.0,
                    "vn_kv": 110.0,
                    "min_vm_pu": 0.95,
                    "max_vm_pu": 1.05,
                    bus_type: 1.0,
                }
            )
            buses.append(row)

        branch = {name: 0.0 for name in BRANCH_FEATURE_COLUMNS}
        branch.update(
            {
                "scenario": scenario_id,
                "idx": 7,
                "from_bus": 10,
                "to_bus": 20,
                "r": 0.01,
                "x": 0.1,
                "b": 0.01,
                "tap": 0.0,
                "shift": 0.0,
                "rate_a": 100.0,
                "br_status": 1.0,
                "ang_min": -30.0,
                "ang_max": 30.0,
            }
        )
        branches.append(branch)
        generators.append(
            {
                "scenario": scenario_id,
                "idx": 0,
                "bus": 10,
                "p_mw": 40.0,
                "q_mvar": 5.0,
                "min_p_mw": 0.0,
                "max_p_mw": 100.0,
                "min_q_mvar": -50.0,
                "max_q_mvar": 50.0,
                "in_service": 1.0,
            }
        )

    return SimpleNamespace(
        bus_df=pd.DataFrame(buses),
        branch_df=pd.DataFrame(branches),
        gen_df=pd.DataFrame(generators),
    )


def _state(
    adapter: SimpleNamespace,
    *,
    scenario_id: int = 1,
    generator_p_mw: float = 40.0,
) -> _GeneratorOperatingPointState:
    bus = adapter.bus_df[
        adapter.bus_df["scenario"] == scenario_id
    ].sort_values("bus").reset_index(drop=True)
    branch = adapter.branch_df[
        adapter.branch_df["scenario"] == scenario_id
    ].sort_values("idx").reset_index(drop=True)
    return _GeneratorOperatingPointState(
        scenario_id=scenario_id,
        load_scenario_idx=0.0,
        bus_features=bus[BUS_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        branch_features=branch[BRANCH_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        branch_ids=branch["idx"].to_numpy(dtype=np.int64),
        branch_status=branch["br_status"].to_numpy(dtype=np.float32),
        metrics={},
        outaged_branch_ids=[],
        bus_ids=bus["bus"].to_numpy(dtype=np.int64),
        generator_ids=np.asarray([0], dtype=np.int64),
        generator_p_mw=np.asarray([generator_p_mw], dtype=np.float64),
        generator_q_mvar=np.asarray([5.0], dtype=np.float64),
        generator_status=np.asarray([1.0], dtype=np.float64),
    )


def _switch_off() -> GridFMAction:
    return GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=7,
        branch_pos=0,
    )


def _install_deterministic_solver(backend, monkeypatch):
    solve_inputs: list[dict[str, np.ndarray]] = []
    state_inputs: list[dict[str, np.ndarray]] = []

    def solve(ppc, *, context):
        del context
        result = {
            "version": "2",
            "baseMVA": float(ppc["baseMVA"]),
            "bus": np.asarray(ppc["bus"], dtype=np.float64).copy(),
            "branch": np.pad(
                np.asarray(ppc["branch"], dtype=np.float64),
                ((0, 0), (0, 4)),
            ),
            "gen": np.asarray(ppc["gen"], dtype=np.float64).copy(),
        }
        result["bus"][:, VM] = [1.01, 0.99]
        solve_inputs.append(
            {
                "bus": np.asarray(ppc["bus"]).copy(),
                "branch": np.asarray(ppc["branch"]).copy(),
                "gen": np.asarray(ppc["gen"]).copy(),
            }
        )
        return result, {"marker": 1.0}

    def build_state(*, state, result_ppc, frames, metrics):
        del frames, metrics
        state_inputs.append(
            {
                "bus": np.asarray(result_ppc["bus"]).copy(),
                "branch": np.asarray(result_ppc["branch"]).copy(),
                "gen": np.asarray(result_ppc["gen"]).copy(),
            }
        )
        branch_features = state.branch_features.copy()
        branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = 0.0
        return replace(
            state,
            branch_features=branch_features,
            branch_status=np.asarray([0.0], dtype=np.float32),
            outaged_branch_ids=[7],
        )

    monkeypatch.setattr(backend, "_solve_ppc", solve)
    monkeypatch.setattr(backend, "_state_from_solved_ppc", build_state)
    return solve_inputs, state_inputs


def test_exact_cache_hit_skips_solver_but_uses_same_state_build_path(
    monkeypatch,
) -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        enable_cache=True,
        exact_cache_max_bytes=1024 * 1024,
    )
    solve_inputs, state_inputs = _install_deterministic_solver(
        backend,
        monkeypatch,
    )
    state = _state(adapter)

    first = backend.run_power_flow_from_state(state, action=_switch_off())
    second = backend.run_power_flow_from_state(state, action=_switch_off())

    assert first.success and second.success
    assert len(solve_inputs) == 1
    assert len(state_inputs) == 2
    np.testing.assert_array_equal(state_inputs[0]["bus"], state_inputs[1]["bus"])
    np.testing.assert_array_equal(
        state_inputs[0]["branch"], state_inputs[1]["branch"]
    )
    np.testing.assert_array_equal(state_inputs[0]["gen"], state_inputs[1]["gen"])
    np.testing.assert_array_equal(
        first.next_state.branch_status,
        second.next_state.branch_status,
    )

    info = backend.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 1
    assert info["bytes"] <= info["max_bytes"]


def test_disabling_cache_does_not_reuse_stale_l1_result(monkeypatch) -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(adapter=adapter, enable_cache=True)
    solve_inputs, state_inputs = _install_deterministic_solver(
        backend,
        monkeypatch,
    )
    state = _state(adapter)

    first = backend.run_power_flow_from_state(state, action=_switch_off())
    backend.enable_cache = False
    second = backend.run_power_flow_from_state(state, action=_switch_off())

    assert first.success and second.success
    assert len(solve_inputs) == 2
    assert len(state_inputs) == 2
    np.testing.assert_array_equal(solve_inputs[0]["bus"], solve_inputs[1]["bus"])
    np.testing.assert_array_equal(
        solve_inputs[0]["branch"], solve_inputs[1]["branch"]
    )
    np.testing.assert_array_equal(solve_inputs[0]["gen"], solve_inputs[1]["gen"])
    assert backend.cache_info()["hits"] == 0
    assert backend.cache_info()["misses"] == 1


def test_tiny_generator_change_is_an_exact_cache_miss(monkeypatch) -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        enable_cache=True,
        exact_cache_max_bytes=1024 * 1024,
    )
    solve_inputs, _state_inputs = _install_deterministic_solver(
        backend,
        monkeypatch,
    )

    first = _state(adapter, generator_p_mw=40.0)
    changed = _state(adapter, generator_p_mw=40.0 + 1e-7)
    backend.run_power_flow_from_state(first, action=_switch_off())
    backend.run_power_flow_from_state(changed, action=_switch_off())

    assert len(solve_inputs) == 2
    assert backend.cache_info()["hits"] == 0
    assert backend.cache_info()["misses"] == 2


def test_identical_physical_problem_reuses_result_across_scenario_ids(
    monkeypatch,
) -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        enable_cache=True,
        exact_cache_max_bytes=1024 * 1024,
    )
    solve_inputs, state_inputs = _install_deterministic_solver(
        backend,
        monkeypatch,
    )

    first = backend.run_power_flow_from_state(
        _state(adapter, scenario_id=1),
        action=_switch_off(),
    )
    second = backend.run_power_flow_from_state(
        _state(adapter, scenario_id=2),
        action=_switch_off(),
    )

    assert first.success and second.success
    assert first.scenario_id == 1
    assert second.scenario_id == 2
    assert len(solve_inputs) == 1
    assert len(state_inputs) == 2
    assert backend.cache_info()["hits"] == 1

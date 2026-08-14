from __future__ import annotations

from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)
from grid_topology_ai.pypower_backend import (
    GridFMPowerFlowBackend,
    _GeneratorOperatingPointState,
)
from grid_topology_ai.topology_actions import GridFMAction


_BUS = {name: index for index, name in enumerate(BUS_FEATURE_COLUMNS)}
_BRANCH = {name: index for index, name in enumerate(BRANCH_FEATURE_COLUMNS)}


def _metrics() -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": 80.0,
        "num_overloaded_branches": 0,
        "num_hard_overloaded_branches": 0,
        "total_thermal_overload_mva": 0.0,
        "num_low_voltage_buses": 0,
        "num_high_voltage_buses": 0,
        "total_voltage_violation": 0.0,
        "num_generator_p_violations": 0,
        "total_generator_p_violation_mw": 0.0,
        "num_generator_q_violations": 0,
        "total_generator_q_violation_mvar": 0.0,
        "num_angle_difference_violations": 0,
        "total_angle_difference_violation_degrees": 0.0,
    }


def _state(
    scenario_id: int,
    *,
    load_scenario_idx: float,
    pd_bus_2: float = 25.0,
    branch_active: bool = True,
) -> _GeneratorOperatingPointState:
    bus_features = np.zeros(
        (2, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    bus_features[:, _BUS["Pd"]] = [45.0, pd_bus_2]
    bus_features[:, _BUS["Qd"]] = [12.0, 7.0]
    bus_features[:, _BUS["Vm"]] = [1.01, 0.99]
    bus_features[:, _BUS["Va"]] = [0.0, -2.0]
    bus_features[:, _BUS["PQ"]] = [0.0, 1.0]
    bus_features[:, _BUS["PV"]] = [0.0, 0.0]
    bus_features[:, _BUS["REF"]] = [1.0, 0.0]
    bus_features[:, _BUS["vn_kv"]] = [230.0, 230.0]
    bus_features[:, _BUS["GS"]] = [0.0, 0.0]
    bus_features[:, _BUS["BS"]] = [0.0, 0.0]
    bus_features[:, _BUS["min_vm_pu"]] = [0.90, 0.90]
    bus_features[:, _BUS["max_vm_pu"]] = [1.10, 1.10]

    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, _BRANCH["r"]] = 0.01
    branch_features[:, _BRANCH["x"]] = 0.10
    branch_features[:, _BRANCH["b"]] = 0.02
    branch_features[:, _BRANCH["tap"]] = 0.0
    branch_features[:, _BRANCH["shift"]] = 0.0
    branch_features[:, _BRANCH["rate_a"]] = 100.0
    branch_features[:, _BRANCH["br_status"]] = float(branch_active)

    return _GeneratorOperatingPointState(
        scenario_id=int(scenario_id),
        load_scenario_idx=float(load_scenario_idx),
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=np.array([[0], [1]], dtype=np.int64),
        branch_ids=np.array([10], dtype=np.int64),
        branch_status=np.array([int(branch_active)], dtype=np.int64),
        metrics=_metrics(),
        outaged_branch_ids=[] if branch_active else [10],
        bus_ids=np.array([100, 200], dtype=np.int64),
        generator_ids=np.array([1], dtype=np.int64),
        generator_p_mw=np.array([70.0], dtype=np.float64),
        generator_q_mvar=np.array([8.0], dtype=np.float64),
        generator_status=np.array([1.0], dtype=np.float64),
    )


def _adapter() -> SimpleNamespace:
    bus_rows = []
    branch_rows = []
    gen_rows = []

    for scenario_id in (1, 2):
        bus_rows.extend(
            [
                {
                    "scenario": scenario_id,
                    "bus": 100,
                    "Vm": 1.01,
                    "min_vm_pu": 0.90,
                    "max_vm_pu": 1.10,
                },
                {
                    "scenario": scenario_id,
                    "bus": 200,
                    "Vm": 0.99,
                    "min_vm_pu": 0.90,
                    "max_vm_pu": 1.10,
                },
            ]
        )
        branch_rows.append(
            {
                "scenario": scenario_id,
                "idx": 10,
                "from_bus": 100,
                "to_bus": 200,
                "ang_min": -360.0,
                "ang_max": 360.0,
            }
        )
        gen_rows.append(
            {
                "scenario": scenario_id,
                "idx": 1,
                "bus": 100,
                "p_mw": 70.0,
                "q_mvar": 8.0,
                "max_q_mvar": 50.0,
                "min_q_mvar": -50.0,
                "in_service": 1.0,
                "max_p_mw": 120.0,
                "min_p_mw": 0.0,
            }
        )

    return SimpleNamespace(
        bus_df=pd.DataFrame(bus_rows),
        branch_df=pd.DataFrame(branch_rows),
        gen_df=pd.DataFrame(gen_rows),
    )


def _switch_off() -> GridFMAction:
    return GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
    )


def _install_fake_solver(backend: GridFMPowerFlowBackend) -> list[int]:
    calls: list[int] = []

    def fake_solve(self, ppc, *, context):
        del self, context
        calls.append(1)
        return ppc, _metrics()

    def fake_build(
        self,
        scenario_id,
        result_ppc,
        previous_state,
        original_frames,
        physical_metrics=None,
    ):
        del self, result_ppc, original_frames, physical_metrics
        solved = _state(
            int(scenario_id),
            load_scenario_idx=float(previous_state.load_scenario_idx),
            pd_bus_2=float(previous_state.bus_features[1, _BUS["Pd"]]),
            branch_active=False,
        )
        return replace(
            solved,
            metrics=_metrics(),
        )

    backend._solve_ppc = MethodType(fake_solve, backend)
    backend._build_state_from_pypower_result_fast = MethodType(fake_build, backend)
    return calls


def test_same_physical_problem_reuses_global_exact_result_across_scenarios(
    tmp_path,
) -> None:
    backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        persistent_cache_root=tmp_path,
    )
    calls = _install_fake_solver(backend)

    first = backend.run_power_flow_from_state(
        _state(1, load_scenario_idx=10.0),
        action=_switch_off(),
    )
    second = backend.run_power_flow_from_state(
        _state(2, load_scenario_idx=20.0),
        action=_switch_off(),
    )

    assert first.success is True
    assert second.success is True
    assert len(calls) == 1
    assert second.message == "Power flow converged. [global exact cache hit]"
    assert second.next_state is not None
    assert second.next_state.scenario_id == 2
    assert second.next_state.load_scenario_idx == 20.0
    np.testing.assert_array_equal(second.next_state.branch_ids, [10])
    np.testing.assert_array_equal(second.next_state.branch_status, [0])
    np.testing.assert_array_equal(second.next_state.bus_ids, [100, 200])
    np.testing.assert_array_equal(second.next_state.generator_ids, [1])
    assert backend.global_exact_cache_hits == 1
    assert backend.exact_cache_hits == 1

    backend.close()


def test_persistent_exact_result_survives_backend_restart(tmp_path) -> None:
    first_backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        persistent_cache_root=tmp_path,
    )
    first_calls = _install_fake_solver(first_backend)
    first_backend.run_power_flow_from_state(
        _state(1, load_scenario_idx=10.0),
        action=_switch_off(),
    )
    assert len(first_calls) == 1
    assert first_backend.persistent_exact_cache_writes == 1
    first_backend.close()

    second_backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        persistent_cache_root=tmp_path,
    )
    second_calls = _install_fake_solver(second_backend)
    result = second_backend.run_power_flow_from_state(
        _state(2, load_scenario_idx=20.0),
        action=_switch_off(),
    )

    assert result.success is True
    assert len(second_calls) == 0
    assert result.message == "Power flow converged. [persistent exact cache hit]"
    assert result.next_state is not None
    assert result.next_state.scenario_id == 2
    assert result.next_state.load_scenario_idx == 20.0
    assert second_backend.persistent_exact_cache_hits == 1
    assert second_backend.global_exact_cache_hits == 1

    second_backend.close()


def test_different_injection_does_not_reuse_global_exact_result(tmp_path) -> None:
    backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        persistent_cache_root=tmp_path,
    )
    calls = _install_fake_solver(backend)

    backend.run_power_flow_from_state(
        _state(1, load_scenario_idx=10.0),
        action=_switch_off(),
    )
    result = backend.run_power_flow_from_state(
        _state(2, load_scenario_idx=20.0, pd_bus_2=26.0),
        action=_switch_off(),
    )

    assert result.success is True
    assert len(calls) == 2
    assert backend.global_exact_cache_hits == 0

    backend.close()


def test_clear_cache_keeps_persistent_exact_results(tmp_path) -> None:
    backend = GridFMPowerFlowBackend(
        adapter=_adapter(),
        persistent_cache_root=tmp_path,
    )
    calls = _install_fake_solver(backend)

    backend.run_power_flow_from_state(
        _state(1, load_scenario_idx=10.0),
        action=_switch_off(),
    )
    backend.clear_cache()

    result = backend.run_power_flow_from_state(
        _state(2, load_scenario_idx=20.0),
        action=_switch_off(),
    )

    assert result.success is True
    assert len(calls) == 1
    assert result.message == "Power flow converged. [persistent exact cache hit]"
    assert backend.persistent_exact_cache_hits == 1

    backend.close()

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.power_flow_errors import PowerFlowFailureKind
from grid_topology_ai.pypower_backend import (
    GridFMPowerFlowBackend,
    GridFMPowerFlowResult,
)
from grid_topology_ai.termination import TerminationReason


def _state(scenario_id: int = 1) -> GridFMState:
    return GridFMState(
        scenario_id=scenario_id,
        load_scenario_idx=0.0,
        bus_features=np.zeros((2, 1), dtype=np.float32),
        branch_features=np.zeros((1, 1), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        branch_ids=np.array([10], dtype=np.int64),
        branch_status=np.array([1.0], dtype=np.float32),
        metrics={
            "power_flow_converged": True,
            "all_values_finite": True,
            "topology_connected": True,
            "max_loading_percent": 110.0,
            "num_overloaded_branches": 1,
            "num_hard_overloaded_branches": 0,
            "total_thermal_overload_mva": 1.0,
            "num_low_voltage_buses": 0,
            "num_high_voltage_buses": 0,
            "total_voltage_violation": 0.0,
            "num_generator_p_violations": 0,
            "total_generator_p_violation_mw": 0.0,
            "num_generator_q_violations": 0,
            "total_generator_q_violation_mvar": 0.0,
            "num_angle_difference_violations": 0,
            "total_angle_difference_violation_degrees": 0.0,
        },
        outaged_branch_ids=[],
    )


class _RawStateForbiddenAdapter:
    def __init__(self) -> None:
        self.build_state_calls = 0

    def build_state(self, scenario_id: int) -> GridFMState:
        self.build_state_calls += 1
        raise AssertionError(
            f"Raw adapter state requested for scenario {scenario_id}."
        )


class _ResetBackend:
    def __init__(self, result: GridFMPowerFlowResult) -> None:
        self.result = result
        self.calls: list[tuple[int, int | None]] = []

    def run_power_flow(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None = None,
    ) -> GridFMPowerFlowResult:
        self.calls.append((int(scenario_id), switched_off_branch_id))
        return self.result


class _RecordingStateBuilder:
    def __init__(self, state: GridFMState) -> None:
        self.state = state
        self.calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> GridFMState:
        self.calls.append(kwargs)
        return self.state


def _environment(
    adapter: _RawStateForbiddenAdapter,
    backend: _ResetBackend,
) -> TopologySwitchingEnv:
    return TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=object(),
        reward_fn=object(),
    )


def test_reset_uses_only_the_canonical_backend_state() -> None:
    canonical_state = _state(scenario_id=7)
    adapter = _RawStateForbiddenAdapter()
    backend = _ResetBackend(
        GridFMPowerFlowResult(
            success=True,
            scenario_id=7,
            switched_off_branch_id=None,
            next_state=canonical_state,
            raw_result=None,
            message="canonical state",
        )
    )
    env = _environment(adapter, backend)

    reset_state = env.reset(7)

    assert reset_state is canonical_state
    assert env.current_state is canonical_state
    assert adapter.build_state_calls == 0
    assert backend.calls == [(7, None)]


def test_reset_failure_never_publishes_a_raw_adapter_state() -> None:
    adapter = _RawStateForbiddenAdapter()
    backend = _ResetBackend(
        GridFMPowerFlowResult(
            success=False,
            scenario_id=9,
            switched_off_branch_id=None,
            next_state=None,
            raw_result=None,
            message="solver did not converge",
            failure_kind=PowerFlowFailureKind.NOT_CONVERGED,
        )
    )
    env = _environment(adapter, backend)

    with pytest.raises(RuntimeError, match="failure_kind=not_converged"):
        env.reset(9)

    assert adapter.build_state_calls == 0
    assert env.current_state is None
    assert env.done is True
    assert env.solved is False
    assert env.termination_reason is TerminationReason.POWER_FLOW_FAILED


def test_initial_and_subsequent_paths_share_one_state_builder() -> None:
    adapter = SimpleNamespace(physics_config=DEFAULT_PHYSICS_CONFIG)
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    expected_state = _state()
    recorder = _RecordingStateBuilder(expected_state)
    backend._state_builder = recorder

    initial_result = {"path": "initial"}
    subsequent_result = {"path": "subsequent"}
    frames = {"source": "same"}

    initial_state = backend._build_state_from_pypower_result(
        scenario_id=1,
        result_ppc=initial_result,
        original_frames=frames,
        physical_metrics={"step": 0},
    )
    subsequent_state = backend._build_state_from_pypower_result_fast(
        scenario_id=1,
        result_ppc=subsequent_result,
        previous_state=expected_state,
        original_frames=frames,
        physical_metrics={"step": 1},
    )

    assert initial_state is expected_state
    assert subsequent_state is expected_state
    assert [call["result_ppc"] for call in recorder.calls] == [
        initial_result,
        subsequent_result,
    ]
    assert all(call["original_frames"] is frames for call in recorder.calls)


def test_backend_rejects_mismatched_physics_fingerprints() -> None:
    adapter = SimpleNamespace(physics_config=DEFAULT_PHYSICS_CONFIG)
    incompatible_config = replace(DEFAULT_PHYSICS_CONFIG, pf_alg=1)

    with pytest.raises(ValueError, match="same PhysicsConfig fingerprint"):
        GridFMPowerFlowBackend(
            adapter=adapter,
            physics_config=incompatible_config,
        )

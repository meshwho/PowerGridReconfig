from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from grid_topology_ai._pypower_backend_core import (
    GridFMPowerFlowBackend as CoreGridFMPowerFlowBackend,
)
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.topology_actions import GridFMAction


def _state() -> GridFMState:
    bus_features = np.zeros(
        (2, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    bus_features[:, BUS_FEATURE_COLUMNS.index("Pd")] = [45.0, 25.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Qd")] = [12.0, 7.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Pg")] = [70.0, 0.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Qg")] = [8.0, 0.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Vm")] = [1.01, 0.99]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Va")] = [0.0, -2.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("min_vm_pu")] = 0.95
    bus_features[:, BUS_FEATURE_COLUMNS.index("max_vm_pu")] = 1.05

    branch_features = np.zeros(
        (2, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("pf")] = [35.0, 18.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("qf")] = [6.0, 3.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("pt")] = [-34.5, -17.8]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("qt")] = [-5.5, -2.8]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = [
        36.0,
        19.0,
    ]

    return GridFMState(
        scenario_id=7,
        load_scenario_idx=2.0,
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=np.array(
            [[0, 1], [1, 0]],
            dtype=np.int64,
        ),
        branch_ids=np.array([10, 20], dtype=np.int64),
        branch_status=np.ones(2, dtype=np.float32),
        metrics={},
        outaged_branch_ids=[],
        bus_ids=np.array([100, 200], dtype=np.int64),
    )


def _switch_off(branch_id: int, branch_pos: int) -> GridFMAction:
    return GridFMAction(
        action_id=1 + branch_pos,
        action_type="switch_off_branch",
        branch_id=branch_id,
        branch_pos=branch_pos,
    )


def _changed_feature(
    state: GridFMState,
    feature_group: str,
    feature_name: str,
) -> GridFMState:
    if feature_group == "bus":
        bus_features = state.bus_features.copy()
        bus_features[0, BUS_FEATURE_COLUMNS.index(feature_name)] += 1.0
        return replace(state, bus_features=bus_features)

    branch_features = state.branch_features.copy()
    branch_features[0, BRANCH_FEATURE_COLUMNS.index(feature_name)] += 1.0
    return replace(state, branch_features=branch_features)


def _core_backend() -> CoreGridFMPowerFlowBackend:
    return CoreGridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        enable_cache=False,
    )


def test_public_and_core_backends_share_cache_key_contract():
    state = _state()
    action = _switch_off(20, 1)
    core = _core_backend()
    public = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        enable_cache=False,
    )

    assert public._make_cache_key_from_state(
        state,
        action=action,
    ) == core._make_cache_key_from_state(
        state,
        action=action,
    )


@pytest.mark.parametrize(
    ("feature_group", "feature_name"),
    [
        ("bus", "Pd"),
        ("bus", "Pg"),
        ("bus", "Vm"),
        ("branch", "pf"),
    ],
)
def test_physical_feature_changes_power_flow_cache_key(
    feature_group: str,
    feature_name: str,
):
    state = _state()
    changed_state = _changed_feature(
        state,
        feature_group,
        feature_name,
    )
    backend = _core_backend()
    action = _switch_off(20, 1)

    assert backend._make_cache_key_from_state(
        state,
        action=action,
    ) != backend._make_cache_key_from_state(
        changed_state,
        action=action,
    )


def test_action_changes_power_flow_cache_key():
    state = _state()
    backend = _core_backend()

    first_key = backend._make_cache_key_from_state(
        state,
        action=_switch_off(10, 0),
    )
    second_key = backend._make_cache_key_from_state(
        state,
        action=_switch_off(20, 1),
    )

    assert first_key != second_key


def test_legacy_branch_id_matches_equivalent_action_key():
    state = _state()
    backend = _core_backend()

    action_key = backend._make_cache_key_from_state(
        state,
        action=_switch_off(20, 1),
    )
    legacy_key = backend._make_cache_key_from_state(
        state,
        switched_off_branch_id=20,
    )

    assert action_key == legacy_key


def test_physics_config_changes_power_flow_cache_key():
    state = _state()
    backend = _core_backend()
    action = _switch_off(20, 1)

    original_key = backend._make_cache_key_from_state(
        state,
        action=action,
    )
    backend.physics_config = replace(
        backend.physics_config,
        max_iterations=31,
    )

    assert original_key != backend._make_cache_key_from_state(
        state,
        action=action,
    )


def test_run_power_flow_cache_isolated_by_source_state():
    state = _state()
    changed_state = _changed_feature(state, "bus", "Pd")
    action = _switch_off(20, 1)

    backend = object.__new__(CoreGridFMPowerFlowBackend)
    backend.physics_config = PhysicsConfig()
    backend.enable_cache = True
    backend.store_raw_result = False
    backend._cache = {}
    backend.cache_hits = 0
    backend.cache_misses = 0

    solve_calls: list[dict[str, object]] = []

    backend._require_usable_next_state = lambda state: None
    backend._build_ppc_from_state = lambda **kwargs: ({}, {})

    def solve_ppc(
        ppc: dict[str, object],
        *,
        context: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        solve_calls.append(ppc)
        return {}, {}

    backend._solve_ppc = solve_ppc
    backend._build_state_from_pypower_result_fast = (
        lambda **kwargs: kwargs["previous_state"]
    )

    first = backend.run_power_flow_from_state(
        state,
        action=action,
    )
    repeated = backend.run_power_flow_from_state(
        state,
        action=action,
    )
    changed = backend.run_power_flow_from_state(
        changed_state,
        action=action,
    )

    assert first.success
    assert repeated.success
    assert changed.success
    assert len(solve_calls) == 2
    assert backend.cache_hits == 1
    assert backend.cache_misses == 2
    assert len(backend._cache) == 2

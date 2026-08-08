from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


def _switch_on(branch_id: int, branch_pos: int) -> GridFMAction:
    return GridFMAction(
        action_id=1 + branch_pos,
        action_type="switch_on_branch",
        branch_id=branch_id,
        branch_pos=branch_pos,
    )


def _with_topology(
    state: GridFMState,
    branch_status: tuple[int, int],
) -> GridFMState:
    status = np.asarray(branch_status, dtype=np.float32)
    branch_features = state.branch_features.copy()
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = status
    return replace(
        state,
        branch_features=branch_features,
        branch_status=status,
        outaged_branch_ids=[
            int(branch_id)
            for branch_id, active in zip(state.branch_ids, status)
            if active <= 0.0
        ],
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
    assert public._make_topology_cache_key_from_state(
        state,
        action=action,
    ) == core._make_topology_cache_key_from_state(
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


def test_topology_cache_key_reuses_equivalent_open_order():
    backend = _core_backend()

    after_10 = _with_topology(_state(), (0, 1))
    after_10 = _changed_feature(after_10, "bus", "Vm")
    after_10 = _changed_feature(after_10, "branch", "pf")

    after_20 = _with_topology(_state(), (1, 0))
    after_20 = _changed_feature(after_20, "bus", "Va")
    after_20 = _changed_feature(after_20, "branch", "qf")

    first_key = backend._make_topology_cache_key_from_state(
        after_10,
        action=_switch_off(20, 1),
    )
    second_key = backend._make_topology_cache_key_from_state(
        after_20,
        action=_switch_off(10, 0),
    )

    assert first_key == second_key
    assert first_key[-1] == ((10, 0), (20, 0))


def test_topology_cache_key_reuses_equivalent_close_order():
    backend = _core_backend()

    branch_10_open = _with_topology(_state(), (0, 1))
    branch_20_open = _with_topology(_state(), (1, 0))

    first_key = backend._make_topology_cache_key_from_state(
        branch_10_open,
        action=_switch_on(10, 0),
    )
    second_key = backend._make_topology_cache_key_from_state(
        branch_20_open,
        action=_switch_on(20, 1),
    )

    assert first_key == second_key
    assert first_key[-1] == ((10, 1), (20, 1))


@pytest.mark.parametrize(
    ("feature_group", "feature_name"),
    [
        ("bus", "Vm"),
        ("bus", "Va"),
        ("bus", "Pg"),
        ("bus", "Qg"),
        ("branch", "pf"),
        ("branch", "qf"),
        ("branch", "pt"),
        ("branch", "qt"),
        ("branch", "loading_percent"),
    ],
)
def test_solved_outputs_do_not_split_topology_cache(
    feature_group: str,
    feature_name: str,
):
    state = _state()
    changed_state = _changed_feature(state, feature_group, feature_name)
    backend = _core_backend()
    action = _switch_off(20, 1)

    assert backend._make_topology_cache_key_from_state(
        state,
        action=action,
    ) == backend._make_topology_cache_key_from_state(
        changed_state,
        action=action,
    )


@pytest.mark.parametrize(
    ("feature_group", "feature_name"),
    [
        ("bus", "Pd"),
        ("bus", "Qd"),
        ("branch", "r"),
        ("branch", "x"),
        ("branch", "tap"),
        ("branch", "rate_a"),
    ],
)
def test_power_flow_inputs_split_topology_cache(
    feature_group: str,
    feature_name: str,
):
    state = _state()
    changed_state = _changed_feature(state, feature_group, feature_name)
    backend = _core_backend()
    action = _switch_off(20, 1)

    assert backend._make_topology_cache_key_from_state(
        state,
        action=action,
    ) != backend._make_topology_cache_key_from_state(
        changed_state,
        action=action,
    )


def test_scenario_and_physics_contract_split_topology_cache():
    state = _state()
    action = _switch_off(20, 1)
    backend = _core_backend()

    original_key = backend._make_topology_cache_key_from_state(
        state,
        action=action,
    )
    other_scenario_key = backend._make_topology_cache_key_from_state(
        replace(state, scenario_id=8),
        action=action,
    )

    backend.physics_config = replace(
        backend.physics_config,
        max_iterations=31,
    )
    other_physics_key = backend._make_topology_cache_key_from_state(
        state,
        action=action,
    )

    assert original_key != other_scenario_key
    assert original_key != other_physics_key


def test_run_power_flow_cache_isolated_by_source_state():
    state = _state()
    changed_state = _changed_feature(state, "bus", "Pd")
    action = _switch_off(20, 1)

    backend = object.__new__(CoreGridFMPowerFlowBackend)
    backend.adapter = object()
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


def test_equivalent_topologies_reuse_one_power_flow_solve():
    after_10 = _with_topology(_state(), (0, 1))
    after_10 = _changed_feature(after_10, "bus", "Vm")
    after_20 = _with_topology(_state(), (1, 0))
    after_20 = _changed_feature(after_20, "branch", "pf")
    both_open = _with_topology(_state(), (0, 0))

    backend = object.__new__(CoreGridFMPowerFlowBackend)
    backend.adapter = object()
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
    backend._build_state_from_pypower_result_fast = lambda **kwargs: both_open

    first = backend.run_power_flow_from_state(
        after_10,
        action=_switch_off(20, 1),
    )
    second = backend.run_power_flow_from_state(
        after_20,
        action=_switch_off(10, 0),
    )

    assert first.success
    assert second.success
    assert second.next_state is both_open
    assert len(solve_calls) == 1
    assert backend.cache_hits == 1
    assert backend.cache_misses == 1
    assert len(backend._cache) == 1


def test_generator_voltage_control_uses_scenario_setpoint_from_state():
    backend = object.__new__(CoreGridFMPowerFlowBackend)
    backend.physics_config = PhysicsConfig()
    backend.adapter = SimpleNamespace(
        bus_df=pd.DataFrame(
            {
                "scenario": [7, 7],
                "bus": [100, 200],
                "Vm": [1.03, 0.98],
            }
        ),
        gen_df=pd.DataFrame(
            {
                "scenario": [7],
                "idx": [0],
                "bus": [100],
            }
        ),
    )

    current_bus_df = pd.DataFrame(
        {
            "bus": [100, 200],
            "Vm": [1.15, 0.91],
        }
    )
    current_branch_df = pd.DataFrame(
        {
            "idx": [10, 20],
            "br_status": [1.0, 1.0],
        }
    )

    seen: dict[str, list[float]] = {}
    backend._state_to_bus_df = lambda state: current_bus_df.copy()
    backend._state_to_branch_df = lambda state: current_branch_df.copy()
    backend._build_bus_matrix = lambda frame: np.zeros((len(frame), 13))
    backend._build_branch_matrix = lambda frame: np.zeros((len(frame), 13))

    def build_gen_matrix(gen_df, bus_df):
        seen["vm"] = bus_df["Vm"].astype(float).tolist()
        return np.zeros((len(gen_df), 21))

    backend._build_gen_matrix = build_gen_matrix
    backend._build_ppc_from_state(_state())

    assert seen["vm"] == [1.03, 0.98]

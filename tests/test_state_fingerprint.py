from __future__ import annotations

from dataclasses import replace

import numpy as np

import grid_topology_ai.state.fingerprint as state_fingerprint
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.state.schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)


def test_fingerprint_version_is_stable():
    assert state_fingerprint._FINGERPRINT_VERSION == b"physical-state-v1"


def _state() -> GridFMState:
    bus_features = np.zeros(
        (3, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    bus_features[:, BUS_FEATURE_COLUMNS.index("Pd")] = [40.0, 30.0, 20.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Qd")] = [12.0, 8.0, 5.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Pg")] = [70.0, 20.0, 0.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Qg")] = [10.0, 5.0, 0.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Vm")] = [1.01, 0.99, 1.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Va")] = [0.0, -2.5, 1.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("min_vm_pu")] = 0.95
    bus_features[:, BUS_FEATURE_COLUMNS.index("max_vm_pu")] = 1.05
    bus_features[
        :,
        BUS_FEATURE_COLUMNS.index("gen_p_up_margin_mw"),
    ] = [30.0, 10.0, 0.0]

    branch_features = np.zeros(
        (2, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("pf")] = [42.0, 0.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("qf")] = [8.0, 0.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("pt")] = [-41.5, 0.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("qt")] = [-7.5, 0.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("r")] = [0.01, 0.02]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("x")] = [0.1, 0.15]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("rate_a")] = [100.0, 90.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = [1.0, 0.0]
    branch_features[
        :,
        BRANCH_FEATURE_COLUMNS.index("loading_percent"),
    ] = [42.75, 0.0]

    return GridFMState(
        scenario_id=7,
        load_scenario_idx=3.0,
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=np.array(
            [[0, 1], [1, 2]],
            dtype=np.int64,
        ),
        branch_ids=np.array([10, 20], dtype=np.int64),
        branch_status=np.array([1.0, 0.0], dtype=np.float32),
        metrics={
            "power_flow_converged": True,
            "max_loading_percent": 42.75,
        },
        outaged_branch_ids=[20],
        bus_ids=np.array([100, 200, 300], dtype=np.int64),
    )


def _copy_state(state: GridFMState) -> GridFMState:
    bus_ids = None
    if state.bus_ids is not None:
        bus_ids = state.bus_ids.copy()

    return replace(
        state,
        bus_features=state.bus_features.copy(),
        branch_features=state.branch_features.copy(),
        edge_index=state.edge_index.copy(),
        branch_ids=state.branch_ids.copy(),
        branch_status=state.branch_status.copy(),
        metrics=dict(state.metrics),
        outaged_branch_ids=list(state.outaged_branch_ids),
        bus_ids=bus_ids,
    )


def test_equal_states_have_equal_fingerprints():
    state = _state()

    fingerprint = state_fingerprint.physical_state_fingerprint(state)

    assert fingerprint == state_fingerprint.physical_state_fingerprint(
        _copy_state(state)
    )
    assert len(fingerprint) == 64
    int(fingerprint, 16)


def test_dtype_and_memory_layout_do_not_change_fingerprint():
    state = _state()

    equivalent_state = replace(
        state,
        bus_features=np.asfortranarray(
            state.bus_features.astype(np.float64)
        ),
        branch_features=np.asfortranarray(
            state.branch_features.astype(np.float64)
        ),
        edge_index=np.asfortranarray(
            state.edge_index.astype(np.int32)
        ),
        branch_ids=state.branch_ids.astype(np.int32),
        branch_status=state.branch_status.astype(np.float64),
        bus_ids=state.bus_ids.astype(np.int32),
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) == state_fingerprint.physical_state_fingerprint(
        equivalent_state
    )


def test_load_change_changes_fingerprint():
    state = _state()
    bus_features = state.bus_features.copy()
    pd_index = BUS_FEATURE_COLUMNS.index("Pd")
    bus_features[0, pd_index] += 5.0

    changed_state = replace(
        state,
        bus_features=bus_features,
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) != state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_generation_change_changes_fingerprint():
    state = _state()
    bus_features = state.bus_features.copy()
    pg_index = BUS_FEATURE_COLUMNS.index("Pg")
    bus_features[1, pg_index] += 4.0

    changed_state = replace(
        state,
        bus_features=bus_features,
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) != state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_generator_margin_change_changes_fingerprint():
    state = _state()
    bus_features = state.bus_features.copy()
    margin_index = BUS_FEATURE_COLUMNS.index(
        "gen_p_up_margin_mw"
    )
    bus_features[0, margin_index] -= 2.0

    changed_state = replace(
        state,
        bus_features=bus_features,
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) != state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_voltage_change_changes_fingerprint():
    state = _state()
    bus_features = state.bus_features.copy()
    vm_index = BUS_FEATURE_COLUMNS.index("Vm")
    bus_features[2, vm_index] = 0.97

    changed_state = replace(
        state,
        bus_features=bus_features,
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) != state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_branch_flow_change_changes_fingerprint():
    state = _state()
    branch_features = state.branch_features.copy()
    pf_index = BRANCH_FEATURE_COLUMNS.index("pf")
    branch_features[0, pf_index] += 3.0

    changed_state = replace(
        state,
        branch_features=branch_features,
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) != state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_branch_status_change_changes_fingerprint():
    state = _state()
    branch_features = state.branch_features.copy()
    status_index = BRANCH_FEATURE_COLUMNS.index("br_status")
    branch_features[1, status_index] = 1.0

    changed_state = replace(
        state,
        branch_features=branch_features,
        branch_status=np.array([1.0, 1.0], dtype=np.float32),
        outaged_branch_ids=[],
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) != state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_edge_index_change_changes_fingerprint():
    state = _state()

    changed_state = replace(
        state,
        edge_index=np.array(
            [[0, 2], [2, 1]],
            dtype=np.int64,
        ),
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) != state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_identifier_change_changes_fingerprint():
    state = _state()

    changed_state = replace(
        state,
        branch_ids=np.array([10, 21], dtype=np.int64),
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) != state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_outage_order_does_not_change_fingerprint():
    state = _state()
    branch_features = state.branch_features.copy()
    status_index = BRANCH_FEATURE_COLUMNS.index("br_status")
    branch_features[:, status_index] = 0.0

    first_state = replace(
        state,
        branch_features=branch_features,
        branch_status=np.array([0.0, 0.0], dtype=np.float32),
        outaged_branch_ids=[10, 20],
    )
    second_state = replace(
        first_state,
        outaged_branch_ids=[20, 10],
    )

    assert state_fingerprint.physical_state_fingerprint(
        first_state
    ) == state_fingerprint.physical_state_fingerprint(
        second_state
    )


def test_scenario_context_changes_fingerprint():
    state = _state()

    other_scenario = replace(
        state,
        scenario_id=8,
    )
    other_load_scenario = replace(
        state,
        load_scenario_idx=4.0,
    )

    original = state_fingerprint.physical_state_fingerprint(state)

    assert original != state_fingerprint.physical_state_fingerprint(
        other_scenario
    )
    assert original != state_fingerprint.physical_state_fingerprint(
        other_load_scenario
    )


def test_derived_metrics_do_not_change_fingerprint():
    state = _state()

    changed_state = replace(
        state,
        metrics={
            "power_flow_converged": True,
            "max_loading_percent": 99.0,
            "diagnostic_note": "changed",
        },
    )

    assert state_fingerprint.physical_state_fingerprint(
        state
    ) == state_fingerprint.physical_state_fingerprint(
        changed_state
    )


def test_schema_change_invalidates_fingerprint(monkeypatch):
    state = _state()
    original = state_fingerprint.physical_state_fingerprint(state)

    monkeypatch.setattr(
        state_fingerprint,
        "state_feature_schema_fingerprint",
        lambda: "0" * 64,
    )

    assert original != state_fingerprint.physical_state_fingerprint(
        state
    )
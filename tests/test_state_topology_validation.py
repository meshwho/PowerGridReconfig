from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import grid_topology_ai.data as data_adapter_module
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.data import GridFMAdapter
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, BUS_FEATURE_COLUMNS
from grid_topology_ai.power_flow.errors import InvalidPhysicalState
from grid_topology_ai.state import validate_state_topology


SCENARIO_ID = 4


def _bus_frame() -> pd.DataFrame:
    rows = []
    for bus_id in (50, 10, 20):
        row = {name: 0.0 for name in BUS_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": SCENARIO_ID,
                "load_scenario_idx": 2.0,
                "bus": bus_id,
                "Vm": 1.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _branch_frame() -> pd.DataFrame:
    rows = []
    for branch_id, from_bus, to_bus, status in (
        (8, 50, 20, 0.0),
        (7, 10, 50, 1.0),
    ):
        row = {name: 0.0 for name in BRANCH_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": SCENARIO_ID,
                "load_scenario_idx": 2.0,
                "idx": branch_id,
                "from_bus": from_bus,
                "to_bus": to_bus,
                "br_status": status,
                "rate_a": 100.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _generator_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": SCENARIO_ID,
                "idx": 2,
                "bus": 20,
                "p_mw": 0.0,
                "q_mvar": 0.0,
                "min_p_mw": 0.0,
                "max_p_mw": 50.0,
                "min_q_mvar": -20.0,
                "max_q_mvar": 20.0,
                "in_service": 0.0,
            },
            {
                "scenario": SCENARIO_ID,
                "idx": 1,
                "bus": 10,
                "p_mw": 40.0,
                "q_mvar": 5.0,
                "min_p_mw": 0.0,
                "max_p_mw": 100.0,
                "min_q_mvar": -30.0,
                "max_q_mvar": 30.0,
                "in_service": 1.0,
            },
        ]
    )


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _bus_frame(), _branch_frame(), _generator_frame()


def _build_adapter(
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    gen_df: pd.DataFrame,
) -> GridFMAdapter:
    adapter = object.__new__(GridFMAdapter)
    adapter.bus_df = bus_df
    adapter.branch_df = branch_df
    adapter.gen_df = gen_df
    adapter.physics_config = DEFAULT_PHYSICS_CONFIG
    return adapter


def test_non_contiguous_bus_ids_build_contiguous_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus_df, branch_df, gen_df = _frames()

    topology = validate_state_topology(
        scenario_id=SCENARIO_ID,
        bus_df=bus_df,
        branch_df=branch_df,
        gen_df=gen_df,
    )

    np.testing.assert_array_equal(topology.bus_ids, [10, 20, 50])
    np.testing.assert_array_equal(topology.branch_ids, [7, 8])
    np.testing.assert_array_equal(
        topology.edge_index,
        np.array([[0, 2], [2, 1]], dtype=np.int64),
    )
    assert topology.edge_index.min() >= 0
    assert topology.edge_index.max() < len(topology.bus_ids)

    monkeypatch.setattr(
        data_adapter_module,
        "calculate_physical_metrics_from_frames",
        lambda **_: {},
    )
    state = _build_adapter(bus_df, branch_df, gen_df).build_state(
        SCENARIO_ID
    )

    np.testing.assert_array_equal(state.bus_ids, [10, 20, 50])
    np.testing.assert_array_equal(state.branch_ids, [7, 8])
    np.testing.assert_array_equal(
        state.edge_index,
        np.array([[0, 2], [2, 1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(state.branch_status, [1.0, 0.0])
    assert state.edge_index.max() < state.bus_features.shape[0]


@pytest.mark.parametrize(
    ("frame_name", "column", "match"),
    [
        ("bus", "bus", r"duplicate bus IDs: 50"),
        ("branch", "idx", r"duplicate branch IDs: 8"),
        ("gen", "idx", r"duplicate generator IDs: 2"),
    ],
)
def test_duplicate_entity_ids_are_rejected(
    frame_name: str,
    column: str,
    match: str,
) -> None:
    bus_df, branch_df, gen_df = _frames()
    frames = {
        "bus": bus_df,
        "branch": branch_df,
        "gen": gen_df,
    }
    frame = frames[frame_name]
    frame.loc[1, column] = frame.loc[0, column]

    with pytest.raises(InvalidPhysicalState, match=match):
        validate_state_topology(
            scenario_id=SCENARIO_ID,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        )


@pytest.mark.parametrize(
    ("frame_name", "column", "value", "match"),
    [
        ("bus", "bus", 10.5, r"bus IDs must contain integral values"),
        ("branch", "idx", np.nan, r"branch IDs contains NaN or infinity"),
        ("gen", "idx", np.inf, r"generator IDs contains NaN or infinity"),
    ],
)
def test_invalid_entity_ids_are_rejected(
    frame_name: str,
    column: str,
    value: float,
    match: str,
) -> None:
    bus_df, branch_df, gen_df = _frames()
    frames = {
        "bus": bus_df,
        "branch": branch_df,
        "gen": gen_df,
    }
    frame = frames[frame_name]
    frame[column] = frame[column].astype(np.float64)
    frame.loc[0, column] = value

    with pytest.raises(InvalidPhysicalState, match=match):
        validate_state_topology(
            scenario_id=SCENARIO_ID,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        )


@pytest.mark.parametrize(
    ("column", "match"),
    [
        ("from_bus", r"branch 8 references unknown from_bus=999"),
        ("to_bus", r"branch 8 references unknown to_bus=999"),
    ],
)
def test_unknown_branch_endpoint_is_rejected(
    column: str,
    match: str,
) -> None:
    bus_df, branch_df, gen_df = _frames()
    branch_df.loc[0, column] = 999

    with pytest.raises(InvalidPhysicalState, match=match):
        validate_state_topology(
            scenario_id=SCENARIO_ID,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        )


def test_generator_on_unknown_bus_is_rejected() -> None:
    bus_df, branch_df, gen_df = _frames()
    gen_df.loc[0, "bus"] = 999

    with pytest.raises(
        InvalidPhysicalState,
        match=r"generator 2 references unknown bus=999",
    ):
        validate_state_topology(
            scenario_id=SCENARIO_ID,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        )


@pytest.mark.parametrize(
    ("frame_name", "column", "value", "match"),
    [
        ("branch", "br_status", 2.0, r"branch status must contain only 0 or 1"),
        ("gen", "in_service", -1.0, r"generator status must contain only 0 or 1"),
    ],
)
def test_invalid_status_is_rejected(
    frame_name: str,
    column: str,
    value: float,
    match: str,
) -> None:
    bus_df, branch_df, gen_df = _frames()
    frames = {
        "branch": branch_df,
        "gen": gen_df,
    }
    frames[frame_name].loc[0, column] = value

    with pytest.raises(InvalidPhysicalState, match=match):
        validate_state_topology(
            scenario_id=SCENARIO_ID,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        )


@pytest.mark.parametrize(
    ("frame_name", "lower", "upper", "match"),
    [
        (
            "bus",
            "min_vm_pu",
            "max_vm_pu",
            r"bus 50 has min_vm_pu=.* greater than max_vm_pu=.*",
        ),
        (
            "gen",
            "min_p_mw",
            "max_p_mw",
            r"generator 2 has min_p_mw=.* greater than max_p_mw=.*",
        ),
        (
            "gen",
            "min_q_mvar",
            "max_q_mvar",
            r"generator 2 has min_q_mvar=.* greater than max_q_mvar=.*",
        ),
    ],
)
def test_inverted_limits_are_rejected(
    frame_name: str,
    lower: str,
    upper: str,
    match: str,
) -> None:
    bus_df, branch_df, gen_df = _frames()
    frames = {
        "bus": bus_df,
        "gen": gen_df,
    }
    frame = frames[frame_name]
    frame.loc[0, lower] = frame.loc[0, upper] + 0.1

    with pytest.raises(InvalidPhysicalState, match=match):
        validate_state_topology(
            scenario_id=SCENARIO_ID,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        )


def test_validation_does_not_mutate_source_frames() -> None:
    bus_df, branch_df, gen_df = _frames()
    expected_bus = bus_df.copy(deep=True)
    expected_branch = branch_df.copy(deep=True)
    expected_gen = gen_df.copy(deep=True)

    validate_state_topology(
        scenario_id=SCENARIO_ID,
        bus_df=bus_df,
        branch_df=branch_df,
        gen_df=gen_df,
    )

    pd.testing.assert_frame_equal(bus_df, expected_bus)
    pd.testing.assert_frame_equal(branch_df, expected_branch)
    pd.testing.assert_frame_equal(gen_df, expected_gen)

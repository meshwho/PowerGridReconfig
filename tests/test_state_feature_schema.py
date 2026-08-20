from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.power_flow.errors import InvalidPhysicalState
from grid_topology_ai.state.schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    BUS_ID_SEMANTICS,
    EDGE_INDEX_SEMANTICS,
    STATE_FEATURE_SCHEMA_VERSION,
    finite_feature_matrix,
    state_feature_schema_fingerprint,
    state_feature_schema_payload,
    state_feature_schema_provenance,
    with_branch_rating_features,
    with_bus_generator_features,
)


EXPECTED_BUS_FEATURE_COLUMNS = [
    "Pd",
    "Qd",
    "Pg",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "vn_kv",
    "GS",
    "BS",
    "min_vm_pu",
    "max_vm_pu",
    "gen_online_count",
    "gen_available",
    "gen_p_min_mw",
    "gen_p_max_mw",
    "gen_q_min_mvar",
    "gen_q_max_mvar",
    "gen_p_down_margin_mw",
    "gen_p_up_margin_mw",
    "gen_q_down_margin_mvar",
    "gen_q_up_margin_mvar",
    "gen_min_p_down_margin_mw",
    "gen_min_p_up_margin_mw",
    "gen_min_q_down_margin_mvar",
    "gen_min_q_up_margin_mvar",
    "gen_p_limit_violation_count",
    "gen_q_limit_violation_count",
]

EXPECTED_BRANCH_FEATURE_COLUMNS = [
    "pf",
    "qf",
    "pt",
    "qt",
    "r",
    "x",
    "b",
    "tap",
    "shift",
    "rate_a",
    "br_status",
    "s_from_mva",
    "s_to_mva",
    "s_max_mva",
    "loading_percent",
    "unlimited_rating",
]


def _bus_frame() -> pd.DataFrame:
    rows = []
    for bus_id in (10, 20):
        row = {name: 0.0 for name in BUS_FEATURE_COLUMNS}
        row.update(
            {
                "bus": bus_id,
                "Vm": 1.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _generator_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bus": 10,
                "p_mw": 70.0,
                "q_mvar": 20.0,
                "min_p_mw": 0.0,
                "max_p_mw": 50.0,
                "min_q_mvar": -10.0,
                "max_q_mvar": 15.0,
                "in_service": 1.0,
            },
            {
                "bus": 10,
                "p_mw": 40.0,
                "q_mvar": -30.0,
                "min_p_mw": 10.0,
                "max_p_mw": 50.0,
                "min_q_mvar": -20.0,
                "max_q_mvar": 20.0,
                "in_service": 1.0,
            },
            {
                "bus": 10,
                "p_mw": 500.0,
                "q_mvar": 500.0,
                "min_p_mw": -500.0,
                "max_p_mw": 500.0,
                "min_q_mvar": -500.0,
                "max_q_mvar": 500.0,
                "in_service": 0.0,
            },
        ]
    )


def test_schema_v3_has_stable_feature_order_and_fingerprint() -> None:
    assert STATE_FEATURE_SCHEMA_VERSION == 3
    assert BUS_FEATURE_COLUMNS == EXPECTED_BUS_FEATURE_COLUMNS
    assert BRANCH_FEATURE_COLUMNS == EXPECTED_BRANCH_FEATURE_COLUMNS

    payload = state_feature_schema_payload()
    assert payload == {
        "state_feature_schema_version": 3,
        "bus_feature_columns": EXPECTED_BUS_FEATURE_COLUMNS,
        "branch_feature_columns": EXPECTED_BRANCH_FEATURE_COLUMNS,
    }
    assert (
        state_feature_schema_fingerprint()
        == "4ecbbc19d6085176a254427d34c719fa29c2d721abb0537b36ad881065eb1930"
    )
    assert state_feature_schema_provenance() == {
        **payload,
        "state_feature_schema_fingerprint": (
            state_feature_schema_fingerprint()
        ),
        "edge_index_semantics": EDGE_INDEX_SEMANTICS,
        "bus_id_semantics": BUS_ID_SEMANTICS,
    }


def test_bus_features_include_aggregate_and_individual_generator_margins() -> None:
    enriched = with_bus_generator_features(
        _bus_frame(),
        _generator_frame(),
    )

    bus_10 = enriched.loc[enriched["bus"] == 10].iloc[0]
    assert bus_10["Pg"] == pytest.approx(110.0)
    assert bus_10["Qg"] == pytest.approx(-10.0)
    assert bus_10["gen_online_count"] == pytest.approx(2.0)
    assert bus_10["gen_available"] == pytest.approx(1.0)
    assert bus_10["gen_p_min_mw"] == pytest.approx(10.0)
    assert bus_10["gen_p_max_mw"] == pytest.approx(100.0)
    assert bus_10["gen_q_min_mvar"] == pytest.approx(-30.0)
    assert bus_10["gen_q_max_mvar"] == pytest.approx(35.0)
    assert bus_10["gen_p_down_margin_mw"] == pytest.approx(100.0)
    assert bus_10["gen_p_up_margin_mw"] == pytest.approx(-10.0)
    assert bus_10["gen_q_down_margin_mvar"] == pytest.approx(20.0)
    assert bus_10["gen_q_up_margin_mvar"] == pytest.approx(45.0)

    # Aggregated margins can hide violations between generators on one bus.
    # The minimum directional margins preserve the worst individual unit.
    assert bus_10["gen_min_p_down_margin_mw"] == pytest.approx(30.0)
    assert bus_10["gen_min_p_up_margin_mw"] == pytest.approx(-20.0)
    assert bus_10["gen_min_q_down_margin_mvar"] == pytest.approx(-10.0)
    assert bus_10["gen_min_q_up_margin_mvar"] == pytest.approx(-5.0)
    assert bus_10["gen_p_limit_violation_count"] == pytest.approx(1.0)
    assert bus_10["gen_q_limit_violation_count"] == pytest.approx(2.0)

    bus_20 = enriched.loc[enriched["bus"] == 20].iloc[0]
    assert bus_20["gen_online_count"] == pytest.approx(0.0)
    assert bus_20["gen_available"] == pytest.approx(0.0)
    assert bus_20["gen_min_p_down_margin_mw"] == pytest.approx(0.0)
    assert bus_20["gen_min_p_up_margin_mw"] == pytest.approx(0.0)
    assert bus_20["gen_min_q_down_margin_mvar"] == pytest.approx(0.0)
    assert bus_20["gen_min_q_up_margin_mvar"] == pytest.approx(0.0)
    assert bus_20["gen_p_limit_violation_count"] == pytest.approx(0.0)
    assert bus_20["gen_q_limit_violation_count"] == pytest.approx(0.0)

    features = finite_feature_matrix(
        enriched,
        BUS_FEATURE_COLUMNS,
        label="bus",
    )
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert features[0, BUS_FEATURE_COLUMNS.index("min_vm_pu")] == pytest.approx(
        0.95
    )
    assert features[0, BUS_FEATURE_COLUMNS.index("max_vm_pu")] == pytest.approx(
        1.05
    )


def test_branch_schema_uses_explicit_unlimited_rating_flag() -> None:
    frame = with_branch_rating_features(
        pd.DataFrame({"rate_a": [100.0, 0.0]})
    )

    assert frame["unlimited_rating"].tolist() == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize("rate_a", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_branch_rating_is_rejected(rate_a: float) -> None:
    frame = pd.DataFrame(
        {
            "pf": [0.0],
            "qf": [0.0],
            "pt": [0.0],
            "qt": [0.0],
            "rate_a": [rate_a],
            "br_status": [1.0],
        }
    )

    with pytest.raises(InvalidPhysicalState):
        GridFMAdapter._add_branch_loading(frame)

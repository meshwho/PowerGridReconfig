from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    physics_provenance,
    require_physics_provenance,
)
from grid_topology_ai.self_play.example_validation import (
    validate_state_physics_provenance,
)
from grid_topology_ai.state.artifacts import (
    validate_state_npz_schema_arrays,
)
from grid_topology_ai.state.schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    with_bus_generator_features,
)


def _valid_arrays() -> dict[str, object]:
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0
    return {
        "bus_features": np.zeros(
            (2, len(BUS_FEATURE_COLUMNS)),
            dtype=np.float32,
        ),
        "branch_features": branch_features,
        "edge_index": np.array([[0], [1]], dtype=np.int64),
        "bus_ids": np.array([10, 20], dtype=np.int64),
        "branch_ids": np.array([7], dtype=np.int64),
        "branch_status": np.array([1.0], dtype=np.float32),
        "metadata_json": np.array(
            json.dumps(physics_provenance(DEFAULT_PHYSICS_CONFIG))
        ),
    }


def _write_state(
    path: Path,
    *,
    remove: str | None = None,
    **overrides: object,
) -> Path:
    arrays = _valid_arrays()
    if remove is not None:
        arrays.pop(remove)
    arrays.update(overrides)
    np.savez(path, **arrays)
    return path


def test_current_state_npz_schema_is_accepted(tmp_path: Path) -> None:
    state_path = _write_state(tmp_path / "state.npz")

    validate_state_npz_schema_arrays(state_path)
    validate_state_physics_provenance(
        state_path,
        expected_physics_config=DEFAULT_PHYSICS_CONFIG,
    )


@pytest.mark.parametrize(
    "field",
    ["bus_feature_columns", "branch_feature_columns"],
)
def test_schema_provenance_requires_ordered_columns(field: str) -> None:
    payload = physics_provenance(DEFAULT_PHYSICS_CONFIG)
    payload.pop(field)

    with pytest.raises(
        ValueError,
        match="Incomplete state-feature schema provenance",
    ):
        require_physics_provenance(payload, source="damaged artifact")


@pytest.mark.parametrize("array_name", ["bus_ids", "branch_status"])
def test_state_npz_requires_identity_and_status_arrays(
    tmp_path: Path,
    array_name: str,
) -> None:
    state_path = _write_state(
        tmp_path / "state.npz",
        remove=array_name,
    )

    with pytest.raises(ValueError, match="missing required schema arrays"):
        validate_state_npz_schema_arrays(state_path)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        (
            "bus_features",
            np.zeros((2, len(BUS_FEATURE_COLUMNS) - 1), dtype=np.float32),
            "bus_features must be non-empty 2D",
        ),
        (
            "branch_features",
            np.zeros((1, len(BRANCH_FEATURE_COLUMNS) + 1), dtype=np.float32),
            "branch_features must be non-empty 2D",
        ),
    ],
)
def test_state_npz_feature_width_must_match_schema(
    tmp_path: Path,
    name: str,
    value: np.ndarray,
    match: str,
) -> None:
    state_path = _write_state(
        tmp_path / "state.npz",
        **{name: value},
    )

    with pytest.raises(ValueError, match=match):
        validate_state_npz_schema_arrays(state_path)


def test_state_npz_rejects_duplicate_bus_ids(tmp_path: Path) -> None:
    state_path = _write_state(
        tmp_path / "state.npz",
        bus_ids=np.array([10, 10], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="bus_ids must be unique"):
        validate_state_npz_schema_arrays(state_path)


def test_state_npz_rejects_branch_status_disagreement(tmp_path: Path) -> None:
    state_path = _write_state(
        tmp_path / "state.npz",
        branch_status=np.array([0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="br_status does not match"):
        validate_state_npz_schema_arrays(state_path)


def test_state_npz_rejects_float32_feature_overflow(tmp_path: Path) -> None:
    bus_features = np.zeros(
        (2, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float64,
    )
    bus_features[0, 0] = 1e39
    state_path = _write_state(
        tmp_path / "state.npz",
        bus_features=bus_features,
    )

    with pytest.raises(ValueError, match="cannot be represented in float32"):
        validate_state_npz_schema_arrays(state_path)


def test_individual_p_margin_survives_aggregate_cancellation() -> None:
    bus = pd.DataFrame(
        [
            {
                "bus": 10,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
            }
        ]
    )
    generators = pd.DataFrame(
        [
            {
                "bus": 10,
                "p_mw": 70.0,
                "q_mvar": 0.0,
                "min_p_mw": 0.0,
                "max_p_mw": 50.0,
                "min_q_mvar": -20.0,
                "max_q_mvar": 20.0,
                "in_service": 1.0,
            },
            {
                "bus": 10,
                "p_mw": 30.0,
                "q_mvar": 0.0,
                "min_p_mw": 0.0,
                "max_p_mw": 50.0,
                "min_q_mvar": -20.0,
                "max_q_mvar": 20.0,
                "in_service": 1.0,
            },
        ]
    )

    row = with_bus_generator_features(bus, generators).iloc[0]

    assert row["gen_p_up_margin_mw"] == pytest.approx(0.0)
    assert row["gen_min_p_up_margin_mw"] == pytest.approx(-20.0)
    assert row["gen_p_limit_violation_count"] == pytest.approx(1.0)

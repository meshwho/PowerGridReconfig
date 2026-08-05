from __future__ import annotations

from pathlib import Path

import numpy as np

from grid_topology_ai.state_schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)


_REQUIRED_ARRAYS = (
    "bus_features",
    "branch_features",
    "edge_index",
    "bus_ids",
    "branch_ids",
    "branch_status",
)


def validate_state_npz_schema_arrays(state_path: str | Path) -> None:
    """Validate that NPZ arrays implement the declared graph-state schema."""

    path = Path(state_path)
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in _REQUIRED_ARRAYS if name not in data.files]
            if missing:
                raise ValueError(
                    f"State NPZ is missing required schema arrays {missing}: {path}"
                )

            bus_features = _feature_matrix(
                data["bus_features"],
                expected_columns=len(BUS_FEATURE_COLUMNS),
                name="bus_features",
                path=path,
            )
            branch_features = _feature_matrix(
                data["branch_features"],
                expected_columns=len(BRANCH_FEATURE_COLUMNS),
                name="branch_features",
                path=path,
            )
            num_buses = int(bus_features.shape[0])
            num_branches = int(branch_features.shape[0])

            _integer_ids(
                data["bus_ids"],
                expected_size=num_buses,
                name="bus_ids",
                path=path,
            )
            _integer_ids(
                data["branch_ids"],
                expected_size=num_branches,
                name="branch_ids",
                path=path,
            )
            branch_status = _branch_status(
                data["branch_status"],
                expected_size=num_branches,
                path=path,
            )
            _edge_index(
                data["edge_index"],
                num_buses=num_buses,
                num_branches=num_branches,
                path=path,
            )
    except (OSError, EOFError) as exc:
        raise ValueError(f"Could not read NPZ state: {path}") from exc

    status_column = BRANCH_FEATURE_COLUMNS.index("br_status")
    feature_status = branch_features[:, status_column]
    if not np.array_equal(feature_status, branch_status):
        raise ValueError(
            f"{path}: branch_features br_status does not match branch_status"
        )


def _feature_matrix(
    value: object,
    *,
    expected_columns: int,
    name: str,
    path: Path,
) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {name} must be numeric") from exc

    expected = f"non-empty 2D with {expected_columns} columns"
    if (
        matrix.ndim != 2
        or matrix.shape[0] <= 0
        or matrix.shape[1] != expected_columns
    ):
        raise ValueError(
            f"{path}: {name} must be {expected}; "
            f"feature dimensions mismatch, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: {name} must contain only finite values")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        float32_matrix = matrix.astype(np.float32)
    if not np.isfinite(float32_matrix).all():
        raise ValueError(f"{path}: {name} cannot be represented in float32")
    return float32_matrix


def _integer_ids(
    value: object,
    *,
    expected_size: int,
    name: str,
    path: Path,
) -> np.ndarray:
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {name} must be numeric") from exc

    expected_shape = (expected_size,)
    if values.shape != expected_shape:
        raise ValueError(
            f"{path}: {name} must have shape {expected_shape}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: {name} must contain only finite values")
    if not np.equal(values, np.rint(values)).all():
        raise ValueError(f"{path}: {name} must be integer-valued")

    limits = np.iinfo(np.int64)
    if np.any(values < limits.min) or np.any(values > limits.max):
        raise ValueError(f"{path}: {name} cannot be represented as int64")

    ids = values.astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{path}: {name} must be unique")
    return ids


def _branch_status(
    value: object,
    *,
    expected_size: int,
    path: Path,
) -> np.ndarray:
    try:
        status = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: branch_status must be numeric") from exc

    expected_shape = (expected_size,)
    if status.shape != expected_shape:
        raise ValueError(
            f"{path}: branch_status must have shape {expected_shape}, "
            f"got {status.shape}"
        )
    if not np.isfinite(status).all():
        raise ValueError(f"{path}: branch_status must contain only finite values")
    if not np.isin(status, (0.0, 1.0)).all():
        raise ValueError(f"{path}: branch_status must contain only 0 or 1")
    return status


def _edge_index(
    value: object,
    *,
    num_buses: int,
    num_branches: int,
    path: Path,
) -> np.ndarray:
    try:
        edges = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: edge_index must be numeric") from exc

    expected_shape = (2, num_branches)
    if edges.shape != expected_shape:
        raise ValueError(
            f"{path}: edge_index must have shape {expected_shape}, "
            f"got {edges.shape}"
        )
    if not np.isfinite(edges).all():
        raise ValueError(f"{path}: edge_index must contain only finite values")
    if not np.equal(edges, np.rint(edges)).all():
        raise ValueError(f"{path}: edge_index must be integer-valued")

    edge_ids = edges.astype(np.int64)
    if edge_ids.min() < 0 or edge_ids.max() >= num_buses:
        raise ValueError(f"{path}: edge_index values out of bounds")
    return edge_ids

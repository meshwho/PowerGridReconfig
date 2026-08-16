from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.topology_actions import GridFMAction


_LODF_EPS = 1e-9
_STATUS_IDX = BRANCH_FEATURE_COLUMNS.index("br_status")
_X_IDX = BRANCH_FEATURE_COLUMNS.index("x")
_PF_IDX = BRANCH_FEATURE_COLUMNS.index("pf")
_RATE_IDX = BRANCH_FEATURE_COLUMNS.index("rate_a")
_LOADING_IDX = BRANCH_FEATURE_COLUMNS.index("loading_percent")


def _readonly_copy(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class LODFStructure:
    """Topology-dependent matrices reused by LODF screening.

    Dynamic branch flow, thermal rating, and loading are deliberately absent.
    They are read from the current state every time actions are ranked.
    """

    active_positions: np.ndarray
    row_by_branch_pos: np.ndarray
    transfer: np.ndarray
    denominator: np.ndarray

    @property
    def owned_bytes(self) -> int:
        return int(
            self.active_positions.nbytes
            + self.row_by_branch_pos.nbytes
            + self.transfer.nbytes
            + self.denominator.nbytes
        )


def _reduced_incidence(
    *,
    active_from: np.ndarray,
    active_to: np.ndarray,
    num_buses: int,
) -> sparse.csc_matrix:
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    rows = np.arange(active_from.size, dtype=np.int64)

    from_mask = active_from != 0
    if np.any(from_mask):
        row_parts.append(rows[from_mask])
        col_parts.append(active_from[from_mask] - 1)
        data_parts.append(np.ones(int(np.sum(from_mask)), dtype=np.float64))

    to_mask = active_to != 0
    if np.any(to_mask):
        row_parts.append(rows[to_mask])
        col_parts.append(active_to[to_mask] - 1)
        data_parts.append(-np.ones(int(np.sum(to_mask)), dtype=np.float64))

    if row_parts:
        row_index = np.concatenate(row_parts)
        col_index = np.concatenate(col_parts)
        values = np.concatenate(data_parts)
    else:
        row_index = np.empty(0, dtype=np.int64)
        col_index = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=np.float64)

    return sparse.csc_matrix(
        (values, (row_index, col_index)),
        shape=(active_from.size, num_buses - 1),
        dtype=np.float64,
    )


def build_lodf_structure(state: GridFMState) -> LODFStructure | None:
    """Build topology-only LODF matrices for one state.

    Sparse LU is the normal path. A dense pseudoinverse is retained only as a
    deterministic fallback for singular/reduced networks so screening keeps its
    previous graceful-degradation behavior.
    """

    branch_features = np.asarray(state.branch_features)
    edge_index = np.asarray(state.edge_index, dtype=np.int64)

    if branch_features.ndim != 2 or edge_index.ndim != 2:
        return None
    if edge_index.shape[0] != 2:
        return None

    num_branches = int(branch_features.shape[0])
    num_buses = int(state.bus_features.shape[0])
    if num_branches <= 1 or num_buses <= 1:
        return None
    if edge_index.shape[1] != num_branches:
        return None

    status = np.asarray(branch_features[:, _STATUS_IDX], dtype=np.float64)
    reactance = np.asarray(branch_features[:, _X_IDX], dtype=np.float64)

    active_mask = (
        (status > 0.0)
        & np.isfinite(reactance)
        & (np.abs(reactance) > _LODF_EPS)
    )
    active_positions = np.flatnonzero(active_mask).astype(np.int64, copy=False)
    if active_positions.size <= 1:
        return None

    active_from = edge_index[0, active_positions].astype(np.int64, copy=False)
    active_to = edge_index[1, active_positions].astype(np.int64, copy=False)
    if (
        np.any(active_from < 0)
        or np.any(active_to < 0)
        or np.any(active_from >= num_buses)
        or np.any(active_to >= num_buses)
    ):
        return None

    active_x = reactance[active_positions]
    active_b = 1.0 / active_x
    incidence = _reduced_incidence(
        active_from=active_from,
        active_to=active_to,
        num_buses=num_buses,
    )
    weighted_incidence = sparse.diags(active_b, format="csc") @ incidence
    bbus = incidence.T @ weighted_incidence

    try:
        factor = splu(bbus.tocsc())
        solved = factor.solve(incidence.T.toarray())
        transfer = np.asarray(weighted_incidence @ solved, dtype=np.float64)
    except Exception:
        incidence_dense = incidence.toarray()
        bbus_dense = incidence_dense.T @ (
            active_b[:, None] * incidence_dense
        )
        try:
            bbus_inverse = np.linalg.pinv(bbus_dense, rcond=1e-10)
        except Exception:
            return None
        transfer = (
            (active_b[:, None] * incidence_dense)
            @ bbus_inverse
            @ incidence_dense.T
        )

    if transfer.shape != (active_positions.size, active_positions.size):
        return None
    if not np.isfinite(transfer).all():
        return None

    denominator = 1.0 - np.diag(transfer)
    row_by_branch_pos = np.full(num_branches, -1, dtype=np.int32)
    row_by_branch_pos[active_positions] = np.arange(
        active_positions.size,
        dtype=np.int32,
    )

    return LODFStructure(
        active_positions=_readonly_copy(active_positions, np.dtype(np.int64)),
        row_by_branch_pos=_readonly_copy(
            row_by_branch_pos,
            np.dtype(np.int32),
        ),
        transfer=_readonly_copy(transfer, np.dtype(np.float64)),
        denominator=_readonly_copy(denominator, np.dtype(np.float64)),
    )


def lodf_loading_safety_score(
    loading_percent: np.ndarray,
    physics_config: PhysicsConfig | None = None,
) -> float:
    """Approximate thermal safety score used only for candidate screening."""

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    loading = np.asarray(loading_percent, dtype=np.float64)

    overload_threshold = (
        config.overload_limit_percent + config.thermal_tolerance_percent
    )
    hard_overload_threshold = (
        config.hard_overload_limit_percent
        + config.thermal_tolerance_percent
    )
    overload = np.where(
        loading > overload_threshold,
        loading - config.overload_limit_percent,
        0.0,
    )
    hard = np.where(
        loading > hard_overload_threshold,
        loading - config.hard_overload_limit_percent,
        0.0,
    )

    num_overloaded = float(np.sum(loading > overload_threshold))
    num_hard = float(np.sum(loading > hard_overload_threshold))
    hard_sq = float(np.sum(hard * hard))
    hard_sum = float(np.sum(hard))
    over_sum = float(np.sum(overload))
    max_hard = float(np.max(hard)) if hard.size else 0.0
    max_over = float(np.max(overload)) if overload.size else 0.0

    return float(
        3.0 * hard_sq
        + 1500.0 * num_hard
        + 50.0 * hard_sum
        + 30.0 * max_hard
        + 5.0 * over_sum
        + 100.0 * num_overloaded
        + 2.0 * max_over
    )


def rank_actions_with_lodf_structure(
    *,
    state: GridFMState,
    actions: list[GridFMAction],
    structure: LODFStructure,
    physics_config: PhysicsConfig | None = None,
) -> list[GridFMAction]:
    """Rank actions using cached topology math and current dynamic flows."""

    branch_features = np.asarray(state.branch_features)
    active_positions = structure.active_positions

    if branch_features.shape[0] != structure.row_by_branch_pos.shape[0]:
        return actions

    active_pf = np.asarray(
        branch_features[active_positions, _PF_IDX],
        dtype=np.float64,
    )
    active_rate = np.asarray(
        branch_features[active_positions, _RATE_IDX],
        dtype=np.float64,
    )

    scored: list[tuple[float, GridFMAction]] = []
    for action in actions:
        branch_pos = int(getattr(action, "branch_pos", -1))
        if branch_pos < 0 or branch_pos >= structure.row_by_branch_pos.size:
            scored.append((float("inf"), action))
            continue

        row = int(structure.row_by_branch_pos[branch_pos])
        if row < 0:
            scored.append((float("inf"), action))
            continue

        denom = float(structure.denominator[row])
        if not np.isfinite(denom) or abs(denom) < _LODF_EPS:
            scored.append((float("inf"), action))
            continue

        lodf_column = structure.transfer[:, row] / denom
        flow_after = active_pf + lodf_column * active_pf[row]
        flow_after = np.asarray(flow_after, dtype=np.float64)
        flow_after[row] = 0.0

        loading_after = np.divide(
            np.abs(flow_after),
            active_rate,
            out=np.zeros_like(flow_after, dtype=np.float64),
            where=active_rate > _LODF_EPS,
        ) * 100.0
        loading_after = np.nan_to_num(
            loading_after,
            nan=0.0,
            posinf=1e9,
            neginf=1e9,
        )

        score = lodf_loading_safety_score(
            loading_after,
            physics_config=physics_config,
        )
        current_loading = float(branch_features[branch_pos, _LOADING_IDX])
        score -= 1e-4 * current_loading
        scored.append((float(score), action))

    scored.sort(key=lambda item: item[0])
    return [action for _, action in scored]

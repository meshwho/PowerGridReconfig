from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grid_topology_ai.cache import LODFStructureCache
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.lodf import (
    build_lodf_structure,
    lodf_loading_safety_score,
    rank_actions_with_lodf_structure,
)
from grid_topology_ai.topology_actions import GridFMAction


_STATUS = BRANCH_FEATURE_COLUMNS.index("br_status")
_X = BRANCH_FEATURE_COLUMNS.index("x")
_PF = BRANCH_FEATURE_COLUMNS.index("pf")
_RATE = BRANCH_FEATURE_COLUMNS.index("rate_a")
_LOADING = BRANCH_FEATURE_COLUMNS.index("loading_percent")
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (1, 3))


def _state(
    *,
    scenario_id: int = 1,
    pf: tuple[float, ...] = (70.0, 50.0, 40.0, -60.0, 20.0),
    rate: tuple[float, ...] = (100.0, 100.0, 100.0, 100.0, 80.0),
) -> SimpleNamespace:
    count = len(_EDGES)
    features = np.zeros(
        (count, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    features[:, _STATUS] = 1.0
    features[:, _X] = np.asarray(
        (0.10, 0.11, 0.12, 0.13, 0.20),
        dtype=np.float32,
    )
    features[:, _PF] = np.asarray(pf, dtype=np.float32)
    features[:, _RATE] = np.asarray(rate, dtype=np.float32)
    features[:, _LOADING] = (
        np.abs(np.asarray(pf, dtype=np.float64))
        / np.asarray(rate, dtype=np.float64)
        * 100.0
    ).astype(np.float32)
    return SimpleNamespace(
        scenario_id=int(scenario_id),
        branch_ids=np.arange(100, 105, dtype=np.int64),
        branch_status=np.ones(count, dtype=np.float32),
        branch_features=features,
        edge_index=np.asarray(_EDGES, dtype=np.int64).T,
        bus_features=np.zeros((4, 1), dtype=np.float32),
        outaged_branch_ids=(),
    )


def _actions() -> list[GridFMAction]:
    return [
        GridFMAction(
            action_id=1 + index,
            action_type="switch_off_branch",
            branch_id=100 + index,
            branch_pos=index,
            target_status=0,
        )
        for index in range(5)
    ]


def _legacy_dense_rank(
    state,
    actions: list[GridFMAction],
    physics_config: PhysicsConfig,
) -> list[GridFMAction]:
    features = state.branch_features
    edge_index = state.edge_index.astype(int)
    status = features[:, _STATUS].astype(float)
    reactance = features[:, _X].astype(float)
    pf = features[:, _PF].astype(float)
    rate = features[:, _RATE].astype(float)
    active = (
        (status > 0.0)
        & np.isfinite(reactance)
        & (np.abs(reactance) > 1e-9)
        & np.isfinite(rate)
        & (rate > 1e-9)
    )
    positions = np.where(active)[0]
    row_by_position = {
        int(position): int(row)
        for row, position in enumerate(positions.tolist())
    }
    active_from = edge_index[0, positions]
    active_to = edge_index[1, positions]
    active_b = 1.0 / reactance[positions]

    incidence = np.zeros((len(positions), 4), dtype=np.float64)
    incidence[np.arange(len(positions)), active_from] = 1.0
    incidence[np.arange(len(positions)), active_to] = -1.0
    reduced = incidence[:, 1:]
    bbus = reduced.T @ (active_b[:, None] * reduced)
    inverse = np.linalg.pinv(bbus, rcond=1e-10)
    transfer = (active_b[:, None] * reduced) @ inverse @ reduced.T
    denominator = 1.0 - np.diag(transfer)
    active_pf = pf[positions]
    active_rate = rate[positions]

    scored: list[tuple[float, GridFMAction]] = []
    for action in actions:
        branch_pos = int(action.branch_pos)
        row = row_by_position.get(branch_pos)
        if row is None:
            scored.append((float("inf"), action))
            continue
        denom = float(denominator[row])
        if not np.isfinite(denom) or abs(denom) < 1e-9:
            scored.append((float("inf"), action))
            continue

        lodf_column = transfer[:, row] / denom
        flow_after = active_pf + lodf_column * active_pf[row]
        flow_after[row] = 0.0
        loading_after = np.divide(
            np.abs(flow_after),
            active_rate,
            out=np.zeros_like(flow_after, dtype=np.float64),
            where=active_rate > 1e-9,
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
        score -= 1e-4 * float(features[branch_pos, _LOADING])
        scored.append((float(score), action))

    scored.sort(key=lambda item: item[0])
    return [action for _, action in scored]


def _ids(actions: list[GridFMAction]) -> list[int]:
    return [int(action.action_id) for action in actions]


def test_sparse_lodf_ranking_matches_previous_dense_ranking() -> None:
    state = _state()
    actions = _actions()
    physics = PhysicsConfig()
    structure = build_lodf_structure(state)  # type: ignore[arg-type]
    assert structure is not None

    actual = rank_actions_with_lodf_structure(
        state=state,  # type: ignore[arg-type]
        actions=actions,
        structure=structure,
        physics_config=physics,
    )
    expected = _legacy_dense_rank(state, actions, physics)
    assert _ids(actual) == _ids(expected)


def test_cached_structure_does_not_cache_dynamic_flow_or_rating() -> None:
    first = _state()
    second = _state(
        scenario_id=2,
        pf=(5.0, 90.0, -75.0, 10.0, 55.0),
        rate=(130.0, 75.0, 140.0, 90.0, 60.0),
    )
    actions = _actions()
    physics = PhysicsConfig()
    cache = LODFStructureCache(max_bytes=4096)

    first_structure = cache.get_or_build(first)  # type: ignore[arg-type]
    second_structure = cache.get_or_build(second)  # type: ignore[arg-type]
    assert first_structure is not None
    assert second_structure is first_structure

    for state in (first, second):
        cached_rank = rank_actions_with_lodf_structure(
            state=state,  # type: ignore[arg-type]
            actions=actions,
            structure=second_structure,
            physics_config=physics,
        )
        dense_rank = _legacy_dense_rank(state, actions, physics)
        assert _ids(cached_rank) == _ids(dense_rank)

    assert cache.info()["hits"] == 1


def test_lodf_cache_and_math_stay_in_separate_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    math_source = (root / "grid_topology_ai" / "lodf.py").read_text(
        encoding="utf-8"
    )
    cache_source = (
        root / "grid_topology_ai" / "cache" / "lodf_structure.py"
    ).read_text(encoding="utf-8")
    runtime_source = (
        root
        / "scripts"
        / "self_play"
        / "generate_impact_teacher_redispatch_runtime.py"
    ).read_text(encoding="utf-8")

    assert "ByteLRUCache" not in math_source
    assert "cache_info" not in math_source
    assert "from grid_topology_ai.lodf import" in cache_source
    assert "def rank_actions_by_lodf_screening(" in runtime_source
    assert "_RUNTIME_LODF_STRUCTURE_CACHE" in runtime_source

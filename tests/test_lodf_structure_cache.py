from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import grid_topology_ai.cache.lodf_structure as cache_module
from grid_topology_ai.cache import (
    LODFStructureCache,
    lodf_structure_fingerprint,
)
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.physics.lodf import build_lodf_structure


_STATUS = BRANCH_FEATURE_COLUMNS.index("br_status")
_X = BRANCH_FEATURE_COLUMNS.index("x")
_TAP = BRANCH_FEATURE_COLUMNS.index("tap")
_PF = BRANCH_FEATURE_COLUMNS.index("pf")
_RATE = BRANCH_FEATURE_COLUMNS.index("rate_a")
_LOADING = BRANCH_FEATURE_COLUMNS.index("loading_percent")
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (1, 3))


def _state(
    *,
    scenario_id: int = 1,
    statuses: tuple[int, ...] = (1, 1, 1, 1, 1),
    reactance: tuple[float, ...] = (0.10, 0.11, 0.12, 0.13, 0.20),
    tap: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
    pf: tuple[float, ...] = (70.0, 50.0, 40.0, -60.0, 20.0),
    rate: tuple[float, ...] = (100.0, 100.0, 100.0, 100.0, 80.0),
) -> SimpleNamespace:
    count = len(_EDGES)
    features = np.zeros(
        (count, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    features[:, _STATUS] = np.asarray(statuses, dtype=np.float32)
    features[:, _X] = np.asarray(reactance, dtype=np.float32)
    features[:, _TAP] = np.asarray(tap, dtype=np.float32)
    features[:, _PF] = np.asarray(pf, dtype=np.float32)

    rate_values = np.asarray(rate, dtype=np.float64)
    flow_values = np.abs(np.asarray(pf, dtype=np.float64))
    features[:, _RATE] = rate_values.astype(np.float32)
    features[:, _LOADING] = (
        np.divide(
            flow_values,
            rate_values,
            out=np.zeros_like(flow_values),
            where=rate_values > 1e-9,
        )
        * 100.0
    ).astype(np.float32)

    return SimpleNamespace(
        scenario_id=int(scenario_id),
        branch_ids=np.arange(100, 100 + count, dtype=np.int64),
        branch_status=np.asarray(statuses, dtype=np.float32),
        branch_features=features,
        edge_index=np.asarray(_EDGES, dtype=np.int64).T,
        bus_features=np.zeros((4, 1), dtype=np.float32),
        outaged_branch_ids=tuple(
            100 + index
            for index, status in enumerate(statuses)
            if status == 0
        ),
    )


def _dense_transfer(state) -> tuple[np.ndarray, np.ndarray]:
    features = state.branch_features
    status = features[:, _STATUS].astype(float)
    reactance = features[:, _X].astype(float)
    tap = features[:, _TAP].astype(float)
    effective_tap = np.where(tap != 0.0, tap, 1.0)
    active = (
        (status > 0.0)
        & np.isfinite(reactance)
        & (np.abs(reactance) > 1e-9)
        & np.isfinite(effective_tap)
    )
    positions = np.where(active)[0]
    edge_index = state.edge_index.astype(int)
    active_from = edge_index[0, positions]
    active_to = edge_index[1, positions]
    active_b = 1.0 / (
        reactance[positions] * effective_tap[positions]
    )

    incidence = np.zeros((len(positions), 4), dtype=np.float64)
    incidence[np.arange(len(positions)), active_from] = 1.0
    incidence[np.arange(len(positions)), active_to] = -1.0
    reduced = incidence[:, 1:]
    bbus = reduced.T @ (active_b[:, None] * reduced)
    inverse = np.linalg.pinv(bbus, rcond=1e-10)
    transfer = (active_b[:, None] * reduced) @ inverse @ reduced.T
    return transfer, 1.0 - np.diag(transfer)


def test_sparse_lodf_structure_matches_dense_reference() -> None:
    state = _state(tap=(0.0, 1.10, 0.0, 0.0, 0.95))
    structure = build_lodf_structure(state)  # type: ignore[arg-type]
    assert structure is not None

    expected_transfer, expected_denominator = _dense_transfer(state)
    np.testing.assert_allclose(
        structure.transfer,
        expected_transfer,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        structure.denominator,
        expected_denominator,
        rtol=1e-10,
        atol=1e-12,
    )


def test_zero_rate_branch_remains_in_lodf_network() -> None:
    unlimited = _state(rate=(100.0, 0.0, 100.0, 100.0, 80.0))
    limited = _state(rate=(100.0, 150.0, 100.0, 100.0, 80.0))

    structure = build_lodf_structure(unlimited)  # type: ignore[arg-type]
    assert structure is not None
    np.testing.assert_array_equal(structure.active_positions, np.arange(5))
    assert lodf_structure_fingerprint(unlimited) == lodf_structure_fingerprint(
        limited
    )


def test_lodf_cache_reuses_topology_across_scenarios_and_dynamic_flows() -> None:
    first = _state()
    second = _state(
        scenario_id=99,
        pf=(10.0, -80.0, 5.0, 75.0, -30.0),
        rate=(150.0, 120.0, 90.0, 110.0, 70.0),
    )
    assert lodf_structure_fingerprint(first) == lodf_structure_fingerprint(second)

    cache = LODFStructureCache(max_bytes=4096)
    first_structure = cache.get_or_build(first)  # type: ignore[arg-type]
    second_structure = cache.get_or_build(second)  # type: ignore[arg-type]

    assert first_structure is not None
    assert second_structure is first_structure
    assert cache.info()["hits"] == 1
    assert cache.info()["misses"] == 1


def test_lodf_cache_invalidates_topology_reactance_and_tap() -> None:
    base = _state()
    opened = _state(statuses=(1, 1, 1, 1, 0))
    changed_x = _state(reactance=(0.10, 0.11, 0.12, 0.13, 0.25))
    changed_tap = _state(tap=(0.0, 1.10, 0.0, 0.0, 0.0))

    assert lodf_structure_fingerprint(base) != lodf_structure_fingerprint(opened)
    assert lodf_structure_fingerprint(base) != lodf_structure_fingerprint(changed_x)
    assert lodf_structure_fingerprint(base) != lodf_structure_fingerprint(changed_tap)

    cache = LODFStructureCache(max_bytes=8192)
    cache.get_or_build(base)  # type: ignore[arg-type]
    cache.get_or_build(opened)  # type: ignore[arg-type]
    cache.get_or_build(changed_x)  # type: ignore[arg-type]
    cache.get_or_build(changed_tap)  # type: ignore[arg-type]
    assert cache.info()["misses"] == 4


def test_lodf_cache_builds_structure_only_on_miss(monkeypatch) -> None:
    calls = 0
    original = cache_module.build_lodf_structure

    def counted(state):
        nonlocal calls
        calls += 1
        return original(state)

    monkeypatch.setattr(cache_module, "build_lodf_structure", counted)
    cache = LODFStructureCache(max_bytes=4096)
    state = _state()

    cache.get_or_build(state)  # type: ignore[arg-type]
    cache.get_or_build(state)  # type: ignore[arg-type]
    assert calls == 1


def test_lodf_cache_never_exceeds_byte_budget() -> None:
    cache = LODFStructureCache(max_bytes=500)
    for final_x in (0.20, 0.21, 0.22, 0.23):
        cache.get_or_build(
            _state(reactance=(0.10, 0.11, 0.12, 0.13, final_x))
        )  # type: ignore[arg-type]
        info = cache.info()
        assert info["bytes"] <= info["max_bytes"]

    assert cache.info()["evictions"] >= 1

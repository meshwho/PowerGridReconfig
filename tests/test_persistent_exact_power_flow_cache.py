from __future__ import annotations

import sqlite3

import numpy as np

from grid_topology_ai.cache import (
    CachedPowerFlowFailure,
    CachedPowerFlowSuccess,
    ExactPowerFlowCache,
    PersistentExactPowerFlowCache,
)
from grid_topology_ai.power_flow_problem import CanonicalPowerFlowProblem


def _problem(marker: float = 0.0) -> CanonicalPowerFlowProblem:
    bus = np.zeros((2, 13), dtype=np.float64)
    branch = np.zeros((1, 13), dtype=np.float64)
    gen = np.zeros((1, 21), dtype=np.float64)
    bus[0, 2] = 40.0 + marker
    branch[0, 3] = 0.1
    branch[0, 10] = 1.0
    gen[0, 1] = 50.0
    return CanonicalPowerFlowProblem(
        base_mva=100.0,
        bus=bus,
        branch=branch,
        gen=gen,
    )


def test_persistent_exact_cache_survives_process_style_restart(tmp_path) -> None:
    root = tmp_path / "pf-cache"
    first_l2 = PersistentExactPowerFlowCache(root, max_bytes=1024 * 1024)
    first = ExactPowerFlowCache(
        max_bytes=128 * 1024,
        persistent_cache=first_l2,
    )
    problem = _problem()
    key, cached = first.lookup(problem, physics_fingerprint="physics-a")
    assert cached is None
    solved = problem.to_ppc(copy=True)
    solved["bus"][0, 7] = 1.01
    first.store_success(key, solved)
    first_l2.close()

    second_l2 = PersistentExactPowerFlowCache(root, max_bytes=1024 * 1024)
    second = ExactPowerFlowCache(
        max_bytes=128 * 1024,
        persistent_cache=second_l2,
    )
    _key, restored = second.lookup(problem, physics_fingerprint="physics-a")

    assert isinstance(restored, CachedPowerFlowSuccess)
    np.testing.assert_array_equal(restored.bus, solved["bus"])
    np.testing.assert_array_equal(restored.branch, solved["branch"])
    np.testing.assert_array_equal(restored.gen, solved["gen"])
    info = second.info()
    assert info["l2_hits"] == 1
    assert info["misses"] == 0


def test_persistent_negative_entry_is_exact_and_restart_safe(tmp_path) -> None:
    root = tmp_path / "pf-cache"
    first_l2 = PersistentExactPowerFlowCache(root, max_bytes=1024 * 1024)
    first = ExactPowerFlowCache(persistent_cache=first_l2)
    problem = _problem()
    key, cached = first.lookup(problem, physics_fingerprint="physics-a")
    assert cached is None
    first.store_not_converged(key, "did not converge")
    first_l2.close()

    second_l2 = PersistentExactPowerFlowCache(root, max_bytes=1024 * 1024)
    second = ExactPowerFlowCache(persistent_cache=second_l2)
    _key, restored = second.lookup(problem, physics_fingerprint="physics-a")
    assert isinstance(restored, CachedPowerFlowFailure)
    assert restored.message == "did not converge"

    _other_key, other = second.lookup(
        _problem(marker=1e-12),
        physics_fingerprint="physics-a",
    )
    assert other is None


def test_persistent_cache_evicts_to_disk_budget(tmp_path) -> None:
    cache = PersistentExactPowerFlowCache(
        tmp_path / "pf-cache",
        max_bytes=256 * 1024,
    )
    array = np.ones((50, 50), dtype=np.float64)

    for marker in range(24):
        assert cache.store_success(
            bytes([marker]) * 32,
            bus=array + marker,
            branch=array,
            gen=array,
        )

    info = cache.info()
    assert info["payload_bytes"] <= int(cache.max_bytes * 0.90)
    assert info["disk_bytes"] <= cache.max_bytes
    assert info["evictions"] > 0


def test_corrupt_persistent_payload_becomes_cache_miss(tmp_path) -> None:
    root = tmp_path / "pf-cache"
    cache = PersistentExactPowerFlowCache(root, max_bytes=1024 * 1024)
    key = b"x" * 32
    assert cache.store_not_converged(key, "failure")
    path = cache.path
    cache.close()

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE entries SET payload = ? WHERE cache_key = ?",
        (sqlite3.Binary(b"corrupt"), sqlite3.Binary(key)),
    )
    connection.commit()
    connection.close()

    reopened = PersistentExactPowerFlowCache(root, max_bytes=1024 * 1024)
    assert reopened.lookup(key) is None
    info = reopened.info()
    assert info["corruptions"] == 1
    assert info["entries"] == 0


def test_stale_persistent_schema_is_disabled_not_reused(tmp_path) -> None:
    root = tmp_path / "pf-cache"
    cache = PersistentExactPowerFlowCache(root, max_bytes=1024 * 1024)
    path = cache.path
    cache.close()

    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=999")
    connection.commit()
    connection.close()

    stale = PersistentExactPowerFlowCache(root, max_bytes=1024 * 1024)
    assert stale.info()["enabled"] is False
    assert stale.lookup(b"z" * 32) is None

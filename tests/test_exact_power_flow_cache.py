from __future__ import annotations

import numpy as np

from grid_topology_ai.cache import (
    ByteLRUCache,
    CachedPowerFlowFailure,
    CachedPowerFlowSuccess,
    ExactPowerFlowCache,
    exact_power_flow_fingerprint,
)
from grid_topology_ai.power_flow_problem import CanonicalPowerFlowProblem


def _problem(*, marker: float = 0.0) -> CanonicalPowerFlowProblem:
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


def test_exact_fingerprint_is_stable_for_identical_solver_input() -> None:
    first = _problem()
    second = CanonicalPowerFlowProblem(
        base_mva=first.base_mva,
        bus=first.bus.copy(),
        branch=first.branch.copy(),
        gen=first.gen.copy(),
    )

    assert exact_power_flow_fingerprint(
        first,
        physics_fingerprint="physics-a",
    ) == exact_power_flow_fingerprint(
        second,
        physics_fingerprint="physics-a",
    )


def test_exact_fingerprint_changes_for_every_solver_input_group() -> None:
    original = _problem()
    original_key = exact_power_flow_fingerprint(
        original,
        physics_fingerprint="physics-a",
    )

    changed_bus = _problem()
    changed_bus.bus[0, 2] += 1e-12
    changed_branch = _problem()
    changed_branch.branch[0, 3] += 1e-12
    changed_gen = _problem()
    changed_gen.gen[0, 1] += 1e-12
    changed_base = CanonicalPowerFlowProblem(
        base_mva=101.0,
        bus=original.bus,
        branch=original.branch,
        gen=original.gen,
    )

    for changed in (changed_bus, changed_branch, changed_gen, changed_base):
        assert exact_power_flow_fingerprint(
            changed,
            physics_fingerprint="physics-a",
        ) != original_key

    assert exact_power_flow_fingerprint(
        original,
        physics_fingerprint="physics-b",
    ) != original_key


def test_byte_lru_evicts_cold_entries_without_exceeding_budget() -> None:
    cache: ByteLRUCache[str, str] = ByteLRUCache(max_bytes=20)
    assert cache.put("a", "A", size_bytes=10)
    assert cache.put("b", "B", size_bytes=10)

    assert cache.get("a") == "A"
    assert cache.put("c", "C", size_bytes=10)

    assert cache.get("b") is None
    assert cache.get("a") == "A"
    assert cache.get("c") == "C"
    assert cache.bytes <= cache.max_bytes
    assert cache.evictions == 1


def test_byte_lru_rejects_single_entry_larger_than_budget() -> None:
    cache: ByteLRUCache[str, str] = ByteLRUCache(max_bytes=8)

    assert not cache.put("large", "value", size_bytes=9)
    assert len(cache) == 0
    assert cache.bytes == 0


def test_exact_cache_stores_read_only_float64_solution_payload() -> None:
    problem = _problem()
    cache = ExactPowerFlowCache(max_bytes=1024 * 1024)
    key, outcome = cache.lookup(problem, physics_fingerprint="physics-a")
    assert outcome is None

    solved = problem.to_ppc(copy=True)
    solved["bus"][0, 7] = 1.01
    assert cache.store_success(key, solved)

    _key, cached = cache.lookup(problem, physics_fingerprint="physics-a")
    assert isinstance(cached, CachedPowerFlowSuccess)
    assert cached.bus.dtype == np.float64
    assert cached.branch.dtype == np.float64
    assert cached.gen.dtype == np.float64
    assert not cached.bus.flags.writeable
    assert not cached.branch.flags.writeable
    assert not cached.gen.flags.writeable

    restored = cached.to_ppc(base_mva=100.0, copy_arrays=True)
    np.testing.assert_array_equal(restored["bus"], solved["bus"])
    np.testing.assert_array_equal(restored["branch"], solved["branch"])
    np.testing.assert_array_equal(restored["gen"], solved["gen"])

    info = cache.info()
    assert info["hits"] == 1
    assert info["misses"] == 1
    assert info["bytes"] <= info["max_bytes"]


def test_exact_cache_negative_entry_requires_identical_problem() -> None:
    cache = ExactPowerFlowCache(max_bytes=4096)
    problem = _problem()
    key, outcome = cache.lookup(problem, physics_fingerprint="physics-a")
    assert outcome is None
    assert cache.store_not_converged(key, "did not converge")

    _key, cached = cache.lookup(problem, physics_fingerprint="physics-a")
    assert isinstance(cached, CachedPowerFlowFailure)
    assert cached.message == "did not converge"

    _other_key, other = cache.lookup(
        _problem(marker=1e-9),
        physics_fingerprint="physics-a",
    )
    assert other is None
    assert cache.info()["negative_hits"] == 1

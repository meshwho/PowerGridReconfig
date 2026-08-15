from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_production_runtime_has_no_global_cache_clear_protocol() -> None:
    staged = _source(
        "scripts/self_play/generate_impact_teacher_redispatch_staged.py"
    )
    runtime = _source(
        "scripts/self_play/generate_impact_teacher_redispatch_runtime.py"
    )
    combined = staged + "\n" + runtime

    for forbidden in (
        "_RUNTIME_GLOBAL_MEMORY_CLEAR_LOCK",
        "_RUNTIME_GLOBAL_MEMORY_LAST_CLEAR",
        "_GLOBAL_MEMORY_CLEAR_COOLDOWN_SEC",
        "maybe_clear_heaviest_worker_for_global_memory",
        "Manager()",
        "manager.dict()",
    ):
        assert forbidden not in combined


def test_production_runtime_uses_bounded_worker_recycling() -> None:
    staged = _source(
        "scripts/self_play/generate_impact_teacher_redispatch_staged.py"
    )

    assert "_DEFAULT_MAX_TASKS_PER_CHILD = 32" in staged
    assert "max_tasks_per_child=max_tasks_per_child" in staged
    assert "byte-bounded caches; no global cache clearing" in staged


def test_persistent_l2_is_separate_from_physics_and_exact_l1() -> None:
    runtime = _source(
        "scripts/self_play/generate_impact_teacher_redispatch_runtime.py"
    )
    exact = _source("grid_topology_ai/cache/exact_power_flow.py")
    persistent = _source("grid_topology_ai/cache/persistent_exact.py")
    physics = _source("grid_topology_ai/power_flow_problem.py")

    assert "PERSISTENT_EXACT_CACHE_DIR_ENV" in runtime
    assert "PersistentExactPowerFlowCache.from_environment()" in exact
    assert "PRAGMA max_page_count" in persistent
    assert "ByteLRUCache" not in physics
    assert "PersistentExactPowerFlowCache" not in physics

    for forbidden in (
        "warm_start",
        "tolerant_cache",
        "_TopologyCacheEntry",
        "_pending_warm",
    ):
        assert forbidden not in exact
        assert forbidden not in persistent


def test_persistent_cache_root_is_portable() -> None:
    runtime = _source(
        "scripts/self_play/generate_impact_teacher_redispatch_runtime.py"
    )

    assert "exact_pf_cache_v1" in runtime
    assert "POWERGRID_EXACT_PERSISTENT_CACHE_DIR" not in runtime
    assert "C:\\" not in runtime
    assert "D:\\" not in runtime


def test_removed_legacy_private_cache_apis_do_not_return() -> None:
    action_space = _source("grid_topology_ai/action_space.py")
    backend = _source("grid_topology_ai/pypower_backend.py")
    core = _source("grid_topology_ai/_pypower_backend_core.py")
    dc_screener = _source("grid_topology_ai/search/dc_action_screener.py")

    assert "def _make_cache_key(" not in action_space
    for forbidden in (
        "def _make_cache_key_from_state(",
        "def _make_topology_cache_key_from_state(",
        "def _power_flow_input_fingerprint(",
    ):
        assert forbidden not in backend
        assert forbidden not in core
    assert "def _score_dc_result(" not in dc_screener

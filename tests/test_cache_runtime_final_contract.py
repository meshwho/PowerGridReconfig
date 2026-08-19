from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            assert node.end_lineno is not None
            lines = source.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"Function {name!r} was not found.")


def test_production_runtime_has_no_active_global_cache_clear_protocol() -> None:
    runtime = _source(
        "scripts/self_play/generate_impact_teacher_redispatch_runtime.py"
    )
    hook = _function_source(runtime, "clear_worker_caches_if_needed")

    assert "return None" in hook
    assert "maybe_clear_heaviest_worker_for_global_memory" not in hook
    assert "clear_worker_caches(" not in hook


def test_production_runtime_does_not_force_worker_recycling() -> None:
    runtime = _source(
        "scripts/self_play/generate_impact_teacher_redispatch_runtime.py"
    )

    assert "_DEFAULT_MAX_TASKS_PER_CHILD: int | None = None" in runtime
    assert "max_tasks_per_child=max_tasks_per_child" in runtime
    assert "byte-bounded caches; no global cache clearing" in runtime


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
    dc_screener = _source("grid_topology_ai/search/dc_action_screener.py")

    assert "def _make_cache_key(" not in action_space
    for forbidden in (
        "def _make_cache_key_from_state(",
        "def _make_topology_cache_key_from_state(",
        "def _power_flow_input_fingerprint(",
    ):
        assert forbidden not in backend
    assert "def _score_dc_result(" not in dc_screener

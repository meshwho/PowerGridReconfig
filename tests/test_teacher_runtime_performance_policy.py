from __future__ import annotations

import ast
from pathlib import Path

from grid_topology_ai.cache.telemetry import (
    exact_power_flow_workload,
    print_exact_power_flow_workload_summary,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = (
    ROOT
    / "scripts"
    / "self_play"
    / "generate_impact_teacher_redispatch_runtime.py"
)


def _runtime_source() -> str:
    return RUNTIME_PATH.read_text(encoding="utf-8")


def _function_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if not matches:
        raise AssertionError(f"Function {name!r} was not found.")
    return matches[-1]


def test_native_thread_defaults_are_applied_before_numeric_stack_imports() -> None:
    source = _runtime_source()
    ast.parse(source)

    configure_call = source.index("_configure_native_math_threads()")
    first_project_import = source.index("from grid_topology_ai")
    assert configure_call < first_project_import

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert f'"{name}"' in source

    # setdefault makes one thread the default while preserving an explicit
    # user-provided value for benchmarking or other workloads.
    assert 'os.environ.setdefault(name, "1")' in source


def test_persistent_l2_is_explicit_opt_in_in_production_runtime() -> None:
    source = _runtime_source()
    configure = _function_node(
        source,
        "_configure_persistent_exact_cache",
    )

    assert '"POWERGRID_ENABLE_PERSISTENT_EXACT_CACHE"' in source
    assert "if not _persistent_cache_requested():" in source

    pop_calls = [
        node
        for node in ast.walk(configure)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "environ"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "os"
    ]
    assert any(
        len(call.args) >= 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "PERSISTENT_EXACT_CACHE_DIR_ENV"
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value is None
        for call in pop_calls
    )

    assert "Persistent exact PF cache:  disabled" in source
    assert "opt-in with" in source


def test_runtime_uses_exact_cache_telemetry_directly() -> None:
    source = _runtime_source()

    assert "return exact_power_flow_workload(" in source
    assert "print_exact_power_flow_workload_summary(" in source
    assert "base._search_workload" not in source
    assert "base._print_power_flow_workload_summary" not in source


def test_exact_workload_uses_current_l1_l2_counters_only() -> None:
    before = {
        "hits": 10,
        "misses": 20,
        "l1_hits": 6,
        "l1_misses": 24,
        "l2_hits": 4,
        "l2_misses": 20,
        "negative_hits": 1,
        "evictions": 2,
        "l2_enabled": True,
        "stock_runpf_calls": 30,
        "q_limit_resolves": 4,
    }
    after = {
        "hits": 18,
        "misses": 25,
        "l1_hits": 12,
        "l1_misses": 31,
        "l2_hits": 6,
        "l2_misses": 25,
        "negative_hits": 2,
        "evictions": 5,
        "l2_enabled": True,
        "stock_runpf_calls": 37,
        "q_limit_resolves": 6,
    }

    workload = exact_power_flow_workload(
        before=before,
        after=after,
        logical_evaluations=15,
    )

    assert workload == {
        "logical_evaluations": 15,
        "cache_hits": 8,
        "cache_misses": 5,
        "cache_hit_rate": 8 / 13,
        "positive_hits": 0,
        "negative_hits": 1,
        "l1_hits": 6,
        "l1_misses": 7,
        "l2_enabled": True,
        "l2_hits": 2,
        "l2_misses": 5,
        "positive_evictions": 0,
        "negative_evictions": 0,
        "warm_start_enabled": False,
        "warm_start_hits": 0,
        "warm_start_misses": 0,
        "warm_start_evictions": 0,
        "warm_start_distance_rejections": 0,
        "warm_start_fallbacks": 0,
        "stock_runpf_calls": 7,
        "q_limit_resolves": 2,
        "solves_per_cache_miss": 7 / 5,
    }
    assert "evictions" not in workload
    assert "tolerant_cache_hits" not in workload
    assert "cold_start_misses" not in workload


def test_exact_workload_summary_reports_ram_cache_without_legacy_labels(
    capsys,
) -> None:
    print_exact_power_flow_workload_summary(
        [
            {
                "logical_evaluations": 100,
                "cache_hits": 20,
                "cache_misses": 80,
                "positive_hits": 17,
                "negative_hits": 3,
                "l1_hits": 20,
                "l1_misses": 80,
                "l2_enabled": False,
                "l2_hits": 0,
                "l2_misses": 0,
                "positive_evictions": 1,
                "negative_evictions": 1,
                "stock_runpf_calls": 160,
                "q_limit_resolves": 80,
            }
        ]
    )
    output = capsys.readouterr().out

    assert "PF cache hits:             20" in output
    assert "  positive physical hits:  17" in output
    assert "  negative exact hits:     3" in output
    assert "  persistent L2:           disabled" in output
    assert "PF cache misses:           80" in output
    assert "  positive L1 misses:      80" in output
    assert "  positive L1 evictions:   1" in output
    assert "  negative L1 evictions:   1" in output
    assert "Stock PYPOWER solves:      160" in output
    assert "Q-limit re-solves:         80" in output
    assert "Exact PF cache" not in output
    assert "tolerant" not in output.lower()
    assert "cold start" not in output.lower()

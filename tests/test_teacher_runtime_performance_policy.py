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

    assert '"POWERGRID_ENABLE_PERSISTENT_EXACT_CACHE"' in source
    assert "if not _persistent_cache_requested():" in source
    assert "os.environ.pop(PERSISTENT_EXACT_CACHE_DIR_ENV, None)" in source
    assert "Persistent exact PF cache:  disabled" in source
    assert "opt-in with" in source


def test_runtime_installs_exact_cache_telemetry() -> None:
    source = _runtime_source()

    assert "base._search_workload = exact_power_flow_workload" in source
    assert (
        "base._print_power_flow_workload_summary = "
        "_print_power_flow_workload_summary"
    ) in source


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
        "l1_hits": 6,
        "l1_misses": 7,
        "l2_enabled": True,
        "l2_hits": 2,
        "l2_misses": 5,
        "negative_hits": 1,
        "evictions": 3,
        "stock_runpf_calls": 7,
        "q_limit_resolves": 2,
        "solves_per_cache_miss": 7 / 5,
    }
    assert "tolerant_cache_hits" not in workload
    assert "warm_start_hits" not in workload
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
                "l1_hits": 20,
                "l1_misses": 80,
                "l2_enabled": False,
                "l2_hits": 0,
                "l2_misses": 0,
                "negative_hits": 3,
                "evictions": 2,
                "stock_runpf_calls": 160,
                "q_limit_resolves": 80,
            }
        ]
    )
    output = capsys.readouterr().out

    assert "Exact PF cache hits:       20" in output
    assert "L1 RAM hits:             20" in output
    assert "L2 persistent cache:     disabled" in output
    assert "negative hits:           3" in output
    assert "Exact PF cache misses:     80" in output
    assert "L1 evictions:            2" in output
    assert "Stock PYPOWER solves:      160" in output
    assert "Q-limit re-solves:         80" in output
    assert "tolerant" not in output.lower()
    assert "warm-start" not in output.lower()
    assert "cold start" not in output.lower()

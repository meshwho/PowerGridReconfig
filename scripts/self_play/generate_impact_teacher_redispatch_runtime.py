from __future__ import annotations

import argparse
import math
import os
import sys


_NATIVE_MATH_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_EXACT_L1_CACHE_MAX_MB_ENV = "POWERGRID_EXACT_L1_CACHE_MAX_MB"
_PF_WARM_START_ENV = "POWERGRID_ENABLE_PF_WARM_START"


def _configure_cache_runtime_from_cli() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--exact-cache-max-mb", type=float, default=None)
    parser.add_argument(
        "--pf-warm-start",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parsed, remaining = parser.parse_known_args(sys.argv[1:])

    if parsed.exact_cache_max_mb is not None:
        max_mb = float(parsed.exact_cache_max_mb)
        if not math.isfinite(max_mb) or max_mb <= 0.0:
            raise SystemExit("--exact-cache-max-mb must be a positive finite number.")
        os.environ[_EXACT_L1_CACHE_MAX_MB_ENV] = f"{max_mb:.12g}"

    if parsed.pf_warm_start is not None:
        os.environ[_PF_WARM_START_ENV] = "1" if parsed.pf_warm_start else "0"

    sys.argv[:] = [sys.argv[0], *remaining]


def _configure_native_math_threads() -> None:
    """Avoid nested native thread pools inside multiprocessing workers."""

    for name in _NATIVE_MATH_THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")


# These must run before importing the numerical/cache stack.
_configure_cache_runtime_from_cli()
_configure_native_math_threads()

from pathlib import Path
from typing import Any, Sequence

from grid_topology_ai.cache import DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES
from grid_topology_ai.cache.persistent_exact import (
    DEFAULT_PERSISTENT_EXACT_CACHE_BYTES,
    PERSISTENT_EXACT_CACHE_DIR_ENV,
    PERSISTENT_EXACT_CACHE_DISABLED_ENV,
    PERSISTENT_EXACT_CACHE_MAX_BYTES_ENV,
)
from grid_topology_ai.cache.power_flow_warm_start import (
    warm_start_enabled_from_environment,
)
from grid_topology_ai.cache.telemetry import (
    exact_power_flow_workload,
    print_exact_power_flow_workload_summary,
)
from grid_topology_ai.runtime import ensure_runtime_scenario_store
from grid_topology_ai.runtime.warm_start_backend import (
    build_memory_mapped_teacher_context,
)
from scripts.self_play import generate_impact_teacher_redispatch_staged as staged


_RUNTIME_SCENARIO_STORE_DIR = "_redispatch_runtime_scenario_store_dir"
_PERSISTENT_CACHE_DIRECTORY_NAME = "exact_pf_cache_v1"
_PERSISTENT_CACHE_ENABLED_ENV = "POWERGRID_ENABLE_PERSISTENT_EXACT_CACHE"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_ORIGINAL_STAGED_INIT = staged.init_worker_context
_ORIGINAL_STAGED_RUN_PARALLEL = staged.run_parallel


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _persistent_cache_requested() -> bool:
    return (
        _env_flag(_PERSISTENT_CACHE_ENABLED_ENV)
        and not _env_flag(PERSISTENT_EXACT_CACHE_DISABLED_ENV)
    )


def _configure_persistent_exact_cache(store_dir: str | Path) -> Path | None:
    if not _persistent_cache_requested():
        # The runtime used to set this path unconditionally, which made the
        # synchronous SQLite L2 part of every PF miss. Keep L2 opt-in.
        os.environ.pop(PERSISTENT_EXACT_CACHE_DIR_ENV, None)
        return None

    default_root = (
        Path(store_dir).resolve().parent / _PERSISTENT_CACHE_DIRECTORY_NAME
    )
    configured = os.environ.get(PERSISTENT_EXACT_CACHE_DIR_ENV)
    cache_root = Path(configured).resolve() if configured else default_root
    os.environ[PERSISTENT_EXACT_CACHE_DIR_ENV] = str(cache_root)
    os.environ.setdefault(
        PERSISTENT_EXACT_CACHE_MAX_BYTES_ENV,
        str(DEFAULT_PERSISTENT_EXACT_CACHE_BYTES),
    )
    return cache_root


def _native_math_thread_summary() -> str:
    return ", ".join(
        f"{name}={os.environ.get(name, '<unset>')}"
        for name in _NATIVE_MATH_THREAD_ENV_VARS
    )


def _install_runtime_telemetry() -> None:
    base = staged.redispatch.base
    base._search_workload = exact_power_flow_workload
    base._print_power_flow_workload_summary = _print_power_flow_workload_summary


def _print_power_flow_workload_summary() -> None:
    base = staged.redispatch.base
    print_exact_power_flow_workload_summary(
        base._PARENT_WORKLOAD_BY_SCENARIO.values()
    )


def _memory_mapped_base_init(
    raw_dir_str: str,
    states_dir_str: str,
    task_config: dict[str, Any],
    scenario_ids: Sequence[int],
    memory_registry=None,
) -> None:
    _install_runtime_telemetry()
    teacher = staged.redispatch.base.teacher
    runtime_task_config = dict(task_config)
    store_dir = runtime_task_config.pop(_RUNTIME_SCENARIO_STORE_DIR, None)

    if store_dir is None:
        store_dir = ensure_runtime_scenario_store(raw_dir_str)
    _configure_persistent_exact_cache(store_dir)

    teacher._WORKER_CONTEXT = build_memory_mapped_teacher_context(
        runtime_store_dir=store_dir,
        states_dir=states_dir_str,
        task_config=runtime_task_config,
        scenario_ids=scenario_ids,
        memory_registry=None,
    )


def init_worker_context(
    raw_dir_str: str,
    states_dir_str: str,
    task_config: dict[str, Any],
    scenario_ids: Sequence[int],
    memory_registry=None,
) -> None:
    _install_runtime_telemetry()
    previous = staged._ORIGINAL_INIT_WORKER_CONTEXT
    staged._ORIGINAL_INIT_WORKER_CONTEXT = _memory_mapped_base_init
    try:
        _ORIGINAL_STAGED_INIT(
            raw_dir_str,
            states_dir_str,
            task_config,
            scenario_ids,
            None,
        )
    finally:
        staged._ORIGINAL_INIT_WORKER_CONTEXT = previous


def run_parallel(
    scenario_batches: list[list[int]],
    scenario_ids: Sequence[int],
    raw_dir,
    states_dir,
    task_config: dict[str, Any],
    checkpoint_path,
    num_workers: int,
    verbose_success: bool,
):
    store_dir = ensure_runtime_scenario_store(Path(raw_dir))
    persistent_root = _configure_persistent_exact_cache(store_dir)
    runtime_task_config = dict(task_config)
    runtime_task_config[_RUNTIME_SCENARIO_STORE_DIR] = str(store_dir)
    print(f"Memory-mapped runtime store: {store_dir}")
    if persistent_root is None:
        print(
            "Persistent exact PF cache:  disabled "
            f"(opt-in with {_PERSISTENT_CACHE_ENABLED_ENV}=1)"
        )
    else:
        print(f"Persistent exact PF cache:  {persistent_root}")
    print(
        "Exact L1 PF cache:          "
        f"{DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES / (1024.0 * 1024.0):.1f} MiB / worker"
    )
    warm_start_text = (
        "enabled (real PF solve is still required)"
        if warm_start_enabled_from_environment()
        else "disabled (opt-in with --pf-warm-start)"
    )
    print(f"PF warm start:             {warm_start_text}")
    print(f"Native math threads:       {_native_math_thread_summary()}")

    return _ORIGINAL_STAGED_RUN_PARALLEL(
        scenario_batches=scenario_batches,
        scenario_ids=scenario_ids,
        raw_dir=raw_dir,
        states_dir=states_dir,
        task_config=runtime_task_config,
        checkpoint_path=checkpoint_path,
        num_workers=num_workers,
        verbose_success=verbose_success,
    )


def _install_runtime_store() -> None:
    _install_runtime_telemetry()
    staged.init_worker_context = init_worker_context
    staged.run_parallel = run_parallel


def main() -> None:
    _install_runtime_store()
    staged.main()


if __name__ == "__main__":
    main()

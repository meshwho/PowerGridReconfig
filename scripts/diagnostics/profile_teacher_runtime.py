from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any


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


def _configure_native_math_threads() -> None:
    for name in _NATIVE_MATH_THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")


# Match the production runtime before NumPy and the numerical stack are imported.
_configure_native_math_threads()

from grid_topology_ai.teacher_config import load_teacher_task_config
from grid_topology_ai.teacher_runtime_profile import (
    TeacherRuntimeProfiler,
    cache_counter_delta,
    cache_snapshot,
    process_memory_snapshot,
)


def load_task_config(path: Path) -> dict[str, Any]:
    return load_teacher_task_config(path)


def _redispatch_module():
    """Import the production mmap teacher runtime only when profiling starts."""

    from scripts.self_play import generate_impact_teacher_redispatch_runtime

    return generate_impact_teacher_redispatch_runtime


def _worker_components(runtime=None) -> tuple[Any, Any]:
    module = runtime if runtime is not None else _redispatch_module()
    teacher = module.staged.redispatch.base.teacher
    ctx = teacher._require_worker_context()
    return ctx.get("backend"), ctx.get("action_space")


def profile_scenario(
    profiler: TeacherRuntimeProfiler,
    scenario_id: int,
    runtime=None,
) -> dict[str, Any]:
    module = runtime if runtime is not None else _redispatch_module()
    backend, action_space = _worker_components(module)
    profiler.reset()

    memory_before = process_memory_snapshot()
    cache_before = cache_snapshot(backend, action_space)

    started = time.perf_counter()
    results = module.staged.redispatch.process_scenario_batch([int(scenario_id)])
    elapsed_sec = time.perf_counter() - started

    memory_after = process_memory_snapshot()
    cache_after = cache_snapshot(backend, action_space)

    if len(results) != 1:
        raise RuntimeError(
            f"Expected one teacher result for scenario {scenario_id}, got {len(results)}."
        )

    result = results[0]
    return {
        "scenario_id": int(scenario_id),
        "ok": bool(result.get("ok", False)),
        "reason": result.get("reason"),
        "elapsed_sec": float(elapsed_sec),
        "profile": profiler.snapshot(),
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "cache_before": cache_before,
        "cache_after": cache_after,
        "cache_delta": cache_counter_delta(cache_before, cache_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the production redispatch teacher hot path without changing "
            "teacher semantics or checkpoint identity."
        )
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--scenario-ids", type=int, nargs="+", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--exact-cache-max-mb",
        type=float,
        default=None,
        help="Exact L1 PF cache size per worker, matching the production runtime.",
    )
    parser.add_argument(
        "--pf-warm-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable the production PF warm-start cache.",
    )
    parser.add_argument(
        "--states-dir",
        type=Path,
        default=None,
        help=(
            "Optional diagnostic state directory. If omitted, temporary state "
            "files are deleted after profiling."
        ),
    )
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be >= 1")

    if args.exact_cache_max_mb is not None:
        max_mb = float(args.exact_cache_max_mb)
        if not math.isfinite(max_mb) or max_mb <= 0.0:
            raise ValueError("--exact-cache-max-mb must be a positive finite number.")
        os.environ[_EXACT_L1_CACHE_MAX_MB_ENV] = f"{max_mb:.12g}"

    os.environ[_PF_WARM_START_ENV] = "1" if args.pf_warm_start else "0"

    scenario_ids = [int(value) for value in args.scenario_ids]
    task_config = load_task_config(args.task_config)
    runtime = _redispatch_module()
    runtime._install_runtime_store()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")

    def run(states_dir: Path) -> None:
        runtime.init_worker_context(
            str(args.raw_dir),
            str(states_dir),
            task_config,
            scenario_ids,
            None,
        )

        profiler = TeacherRuntimeProfiler()
        with profiler.installed():
            for repeat_index in range(int(args.repeat)):
                for scenario_id in scenario_ids:
                    record = profile_scenario(
                        profiler,
                        scenario_id,
                        runtime=runtime,
                    )
                    record["repeat"] = int(repeat_index)
                    encoded = json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    print(encoded, flush=True)

                    if args.output is not None:
                        with args.output.open("a", encoding="utf-8") as handle:
                            handle.write(encoded + "\n")

    if args.states_dir is not None:
        args.states_dir.mkdir(parents=True, exist_ok=True)
        run(args.states_dir)
        return

    with tempfile.TemporaryDirectory(prefix="powergrid_teacher_profile_") as temp_dir:
        run(Path(temp_dir))


if __name__ == "__main__":
    main()

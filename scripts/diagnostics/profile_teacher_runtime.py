from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from grid_topology_ai.teacher_runtime_profile import (
    TeacherRuntimeProfiler,
    cache_counter_delta,
    cache_snapshot,
    process_memory_snapshot,
)
from scripts.self_play import generate_impact_teacher_redispatch as redispatch


def load_task_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher task config must contain a JSON object: {path}")

    nested = payload.get("task_config")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(payload)


def _worker_components() -> tuple[Any, Any]:
    ctx = redispatch.base.teacher._require_worker_context()
    return ctx.get("backend"), ctx.get("action_space")


def profile_scenario(
    profiler: TeacherRuntimeProfiler,
    scenario_id: int,
) -> dict[str, Any]:
    backend, action_space = _worker_components()
    profiler.reset()

    memory_before = process_memory_snapshot()
    cache_before = cache_snapshot(backend, action_space)

    started = time.perf_counter()
    results = redispatch.process_scenario_batch([int(scenario_id)])
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
            "Profile the current redispatch teacher hot path without changing "
            "teacher semantics or checkpoint identity."
        )
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--scenario-ids", type=int, nargs="+", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
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

    scenario_ids = [int(value) for value in args.scenario_ids]
    task_config = load_task_config(args.task_config)

    redispatch._install_overrides()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")

    def run(states_dir: Path) -> None:
        redispatch.init_worker_context(
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
                    record = profile_scenario(profiler, scenario_id)
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

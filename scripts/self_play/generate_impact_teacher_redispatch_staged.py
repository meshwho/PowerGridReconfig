from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from grid_topology_ai.cache import LODFStructureCache
from grid_topology_ai.lodf import (
    build_lodf_structure,
    rank_actions_with_lodf_structure,
)
from grid_topology_ai.teacher_config import (
    ensure_teacher_checkpoint_config,
    teacher_run_id,
)
from scripts.self_play import generate_impact_teacher_redispatch as redispatch


_RUNTIME_READY_EVENT = "_redispatch_worker_ready_event"
_RUNTIME_READY_COUNT = "_redispatch_worker_ready_count"
_RUNTIME_READY_LOCK = "_redispatch_worker_ready_lock"
_RUNTIME_EXPECTED_WORKERS = "_redispatch_expected_workers"
_RUNTIME_GLOBAL_MEMORY_CLEAR_LOCK = "_redispatch_global_memory_clear_lock"
_RUNTIME_GLOBAL_MEMORY_LAST_CLEAR = "_redispatch_global_memory_last_clear"
_RUNTIME_LODF_STRUCTURE_CACHE = "_redispatch_lodf_structure_cache"
_GLOBAL_MEMORY_CLEAR_COOLDOWN_SEC = 5.0
_WORKER_START_BARRIER_TIMEOUT_SEC = 900.0

_ORIGINAL_INIT_WORKER_CONTEXT = redispatch.init_worker_context
_ORIGINAL_RUN_PARALLEL = redispatch.run_parallel
_ORIGINAL_CLEAR_WORKER_CACHES = redispatch.base.teacher.clear_worker_caches
_ORIGINAL_GLOBAL_MEMORY_GUARD = (
    redispatch.base.teacher.maybe_clear_heaviest_worker_for_global_memory
)


def _worker_run_id() -> str:
    teacher = redispatch.base.teacher
    ctx = teacher._require_worker_context()
    states_dir = Path(ctx["state_store"].output_dir)
    return teacher_run_id(states_dir, ctx["task_config"])


def rank_actions_by_lodf_screening(
    state,
    actions,
    physics_config=None,
):
    """Rank with topology-only LODF reuse and current dynamic branch values."""

    if not actions:
        return actions

    teacher = redispatch.base.teacher
    ctx = getattr(teacher, "_WORKER_CONTEXT", None)
    cache = (
        ctx.get(_RUNTIME_LODF_STRUCTURE_CACHE)
        if isinstance(ctx, dict)
        else None
    )
    structure = (
        cache.get_or_build(state)
        if isinstance(cache, LODFStructureCache)
        else build_lodf_structure(state)
    )
    if structure is None:
        return actions

    return rank_actions_with_lodf_structure(
        state=state,
        actions=actions,
        structure=structure,
        physics_config=physics_config,
    )


def clear_worker_caches(reason: str = "manual") -> None:
    """Clear bounded worker caches through their owning components."""

    _ORIGINAL_CLEAR_WORKER_CACHES(reason=reason)
    teacher = redispatch.base.teacher
    ctx = getattr(teacher, "_WORKER_CONTEXT", None)
    if not isinstance(ctx, dict):
        return

    cache = ctx.get(_RUNTIME_LODF_STRUCTURE_CACHE)
    if isinstance(cache, LODFStructureCache):
        cache.clear(reset_counters=True)


def init_worker_context(
    raw_dir_str: str,
    states_dir_str: str,
    task_config: dict[str, Any],
    scenario_ids: Sequence[int],
    memory_registry=None,
) -> None:
    runtime_task_config = dict(task_config)
    ready_event = runtime_task_config.pop(_RUNTIME_READY_EVENT, None)
    ready_count = runtime_task_config.pop(_RUNTIME_READY_COUNT, None)
    ready_lock = runtime_task_config.pop(_RUNTIME_READY_LOCK, None)
    expected_workers = runtime_task_config.pop(_RUNTIME_EXPECTED_WORKERS, None)
    global_clear_lock = runtime_task_config.pop(
        _RUNTIME_GLOBAL_MEMORY_CLEAR_LOCK,
        None,
    )
    global_last_clear = runtime_task_config.pop(
        _RUNTIME_GLOBAL_MEMORY_LAST_CLEAR,
        None,
    )

    _ORIGINAL_INIT_WORKER_CONTEXT(
        raw_dir_str,
        states_dir_str,
        runtime_task_config,
        scenario_ids,
        memory_registry,
    )

    teacher = redispatch.base.teacher
    ctx = teacher._require_worker_context()
    ctx[_RUNTIME_GLOBAL_MEMORY_CLEAR_LOCK] = global_clear_lock
    ctx[_RUNTIME_GLOBAL_MEMORY_LAST_CLEAR] = global_last_clear
    ctx[_RUNTIME_LODF_STRUCTURE_CACHE] = (
        None
        if bool(runtime_task_config.get("disable_cache", False))
        else LODFStructureCache()
    )

    # Spawned workers import this module without running main(), so install the
    # runtime overrides explicitly in the initializer as well as in the parent.
    redispatch.base._worker_run_id = _worker_run_id
    teacher.rank_actions_by_lodf_screening = rank_actions_by_lodf_screening
    teacher.clear_worker_caches = clear_worker_caches

    # These controls are runtime-only. Keeping them outside task_config is
    # important because provenance hashes task_config to build the teacher run id.
    teacher.maybe_clear_heaviest_worker_for_global_memory = (
        maybe_clear_heaviest_worker_for_global_memory
    )
    teacher.clear_worker_caches_if_needed = clear_worker_caches_if_needed

    if ready_event is None:
        return

    if ready_count is None or ready_lock is None or expected_workers is None:
        raise RuntimeError("Incomplete worker start barrier configuration.")

    expected = int(expected_workers)

    if not ready_event.is_set():
        with ready_lock:
            if not ready_event.is_set():
                ready_count.value += 1
                ready = int(ready_count.value)
                print(
                    f"[worker {os.getpid()}] initialized "
                    f"({ready}/{expected}), waiting for pool",
                    flush=True,
                )
                if ready >= expected:
                    ready_event.set()

    if not ready_event.wait(timeout=_WORKER_START_BARRIER_TIMEOUT_SEC):
        raise RuntimeError(
            "Timed out waiting for all teacher workers to initialize."
        )


def _partition_batches(
    scenario_batches: Sequence[Sequence[int]],
    worker_count: int,
) -> list[list[list[int]]]:
    workers = max(int(worker_count), 1)
    shards: list[list[list[int]]] = [[] for _ in range(workers)]

    for index, batch in enumerate(scenario_batches):
        shards[index % workers].append([int(value) for value in batch])

    return [shard for shard in shards if shard]


def _shard_scenario_ids(shard_batches: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(scenario_id)
                for batch in shard_batches
                for scenario_id in batch
            }
        )
    )


def _handle_batch_results(
    batch_results: Sequence[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    checkpoint_path,
    verbose_success: bool,
) -> tuple[int, int]:
    teacher = redispatch.base.teacher
    saved = 0
    skipped = 0

    for result in batch_results:
        teacher.append_scenario_checkpoint(
            checkpoint_path=checkpoint_path,
            result=result,
        )

        if result["ok"]:
            rows.extend(result["rows"])
            saved += 1
            if verbose_success:
                teacher.print_success(result)
        else:
            skipped += 1
            teacher.print_failure(result)

    return saved, skipped


def maybe_clear_heaviest_worker_for_global_memory() -> None:
    teacher = redispatch.base.teacher
    ctx = teacher._require_worker_context()
    cfg = ctx["task_config"]

    clear_lock = ctx.get(_RUNTIME_GLOBAL_MEMORY_CLEAR_LOCK)
    last_clear = ctx.get(_RUNTIME_GLOBAL_MEMORY_LAST_CLEAR)

    if clear_lock is None or last_clear is None:
        _ORIGINAL_GLOBAL_MEMORY_GUARD()
        return

    min_free_mb = float(cfg.get("min_free_system_memory_mb", 0.0))
    if min_free_mb <= 0.0:
        return

    available_mb = teacher.get_system_available_memory_mb()
    if available_mb is None or available_mb >= min_free_mb:
        return

    registry = ctx.get("memory_registry")
    if registry is None:
        return

    current_pid = int(os.getpid())
    max_age_sec = float(cfg.get("memory_registry_max_age_sec", 120.0))

    with clear_lock:
        now = time.time()
        if now - float(last_clear.value) < _GLOBAL_MEMORY_CLEAR_COOLDOWN_SEC:
            return

        available_mb = teacher.get_system_available_memory_mb()
        if available_mb is None or available_mb >= min_free_mb:
            return

        teacher.update_worker_memory_registry()

        heaviest_pid: int | None = None
        heaviest_mb = -1.0

        try:
            for pid_raw, info in list(registry.items()):
                pid = int(pid_raw)
                rss_mb = float(info.get("rss_mb", 0.0))
                timestamp = float(info.get("timestamp", 0.0))

                if now - timestamp > max_age_sec:
                    continue

                if rss_mb > heaviest_mb:
                    heaviest_mb = rss_mb
                    heaviest_pid = pid
        except Exception:
            return

        if heaviest_pid != current_pid:
            return

        last_clear.value = now

    teacher.clear_worker_caches(
        reason=(
            f"global_memory_low_available_{available_mb:.1f}_mb_"
            f"lt_{min_free_mb:.1f}_mb_heaviest_{heaviest_mb:.1f}_mb"
        )
    )
    teacher.update_worker_memory_registry()


def clear_worker_caches_if_needed() -> None:
    """Keep only memory-pressure cache clearing in the staged teacher."""

    teacher = redispatch.base.teacher
    ctx = teacher._require_worker_context()
    cfg = ctx["task_config"]

    ctx["processed_in_worker"] = int(ctx.get("processed_in_worker", 0)) + 1

    teacher.update_worker_memory_registry()
    maybe_clear_heaviest_worker_for_global_memory()

    memory_mb = teacher.get_process_memory_mb()
    max_memory_mb = float(cfg.get("max_worker_memory_mb", 0.0))

    should_clear_by_memory = (
        memory_mb is not None
        and max_memory_mb > 0.0
        and memory_mb >= max_memory_mb
    )

    if should_clear_by_memory:
        teacher.clear_worker_caches(
            reason=(
                f"worker_memory_guard_{memory_mb:.1f}_mb_ge_"
                f"{max_memory_mb:.1f}_mb"
            )
        )
        teacher.update_worker_memory_registry()


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
    if not scenario_batches:
        return [], 0, 0

    workers = min(max(int(num_workers), 1), len(scenario_batches))
    runtime_task_config = dict(task_config)

    if workers == 1:
        return _ORIGINAL_RUN_PARALLEL(
            scenario_batches=scenario_batches,
            scenario_ids=scenario_ids,
            raw_dir=raw_dir,
            states_dir=states_dir,
            task_config=runtime_task_config,
            checkpoint_path=checkpoint_path,
            num_workers=workers,
            verbose_success=verbose_success,
        )

    teacher = redispatch.base.teacher
    shards = _partition_batches(scenario_batches, workers)
    workers = len(shards)
    shard_sizes = [_shard_scenario_ids(shard) for shard in shards]

    init_concurrency = min(
        redispatch._worker_init_concurrency(),
        workers,
    )

    runtime_task_config[redispatch._RUNTIME_WORKER_INIT_SEMAPHORE] = (
        mp.BoundedSemaphore(init_concurrency)
    )
    runtime_task_config[_RUNTIME_READY_EVENT] = mp.Event()
    runtime_task_config[_RUNTIME_READY_COUNT] = mp.Value("i", 0)
    runtime_task_config[_RUNTIME_READY_LOCK] = mp.Lock()
    runtime_task_config[_RUNTIME_EXPECTED_WORKERS] = workers
    runtime_task_config[_RUNTIME_GLOBAL_MEMORY_CLEAR_LOCK] = mp.Lock()
    runtime_task_config[_RUNTIME_GLOBAL_MEMORY_LAST_CLEAR] = mp.Value("d", 0.0)

    counts = [len(values) for values in shard_sizes]
    print(
        f"Partitioned adapters: {workers} workers, "
        f"{min(counts)}-{max(counts)} scenarios per worker"
    )
    print(
        f"Worker start barrier: {workers} workers must initialize "
        "before beam search starts"
    )
    print(f"Worker init concurrency: {init_concurrency}")
    print("Periodic worker cache clearing: disabled; memory guards remain active")
    print(
        "Global memory cache clearing: one worker at a time with "
        f"{_GLOBAL_MEMORY_CLEAR_COOLDOWN_SEC:.0f}s cooldown"
    )
    print(f"\nParallel sharded mode: {workers} workers")
    print(f"Batches:                {len(scenario_batches)}")

    manager = None
    memory_registry = None
    executors: list[ProcessPoolExecutor] = []
    futures = []
    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0

    if float(task_config.get("min_free_system_memory_mb", 0.0)) > 0.0:
        manager = mp.Manager()
        memory_registry = manager.dict()

    max_tasks_per_child = int(task_config.get("max_tasks_per_child", 0))

    try:
        for shard_batches, shard_scenarios in zip(shards, shard_sizes):
            executor_kwargs: dict[str, Any] = {
                "max_workers": 1,
                "initializer": init_worker_context,
                "initargs": (
                    str(raw_dir),
                    str(states_dir),
                    runtime_task_config,
                    shard_scenarios,
                    memory_registry,
                ),
            }
            if max_tasks_per_child > 0:
                executor_kwargs["max_tasks_per_child"] = max_tasks_per_child

            executor = ProcessPoolExecutor(**executor_kwargs)
            executors.append(executor)

            for batch in shard_batches:
                futures.append(
                    executor.submit(teacher.process_scenario_batch, batch)
                )

        iterator = as_completed(futures)
        if teacher.tqdm is not None:
            iterator = teacher.tqdm(
                iterator,
                total=len(futures),
                desc="Teacher batches",
                unit="batch",
                dynamic_ncols=True,
            )

        for future in iterator:
            saved, skipped = _handle_batch_results(
                future.result(),
                rows=rows,
                checkpoint_path=checkpoint_path,
                verbose_success=verbose_success,
            )
            total_saved += saved
            total_skipped += skipped
    finally:
        for executor in executors:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass

    return rows, total_saved, total_skipped


def _install_staged_start() -> None:
    redispatch.init_worker_context = init_worker_context
    redispatch.run_parallel = run_parallel
    redispatch.base.teacher.ensure_checkpoint_config = (
        ensure_teacher_checkpoint_config
    )
    redispatch.base._worker_run_id = _worker_run_id
    redispatch.base.teacher.rank_actions_by_lodf_screening = (
        rank_actions_by_lodf_screening
    )
    redispatch.base.teacher.clear_worker_caches = clear_worker_caches
    redispatch.base.teacher.maybe_clear_heaviest_worker_for_global_memory = (
        maybe_clear_heaviest_worker_for_global_memory
    )
    redispatch.base.teacher.clear_worker_caches_if_needed = (
        clear_worker_caches_if_needed
    )


def main() -> None:
    _install_staged_start()
    redispatch.main()


if __name__ == "__main__":
    main()

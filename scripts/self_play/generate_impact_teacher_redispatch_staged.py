from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Sequence

from scripts.self_play import generate_impact_teacher_redispatch as redispatch


_RUNTIME_READY_EVENT = "_redispatch_worker_ready_event"
_RUNTIME_READY_COUNT = "_redispatch_worker_ready_count"
_RUNTIME_READY_LOCK = "_redispatch_worker_ready_lock"
_RUNTIME_EXPECTED_WORKERS = "_redispatch_expected_workers"
_RUNTIME_CLEAR_CACHES_EVERY = 0
_RUNTIME_GLOBAL_MEMORY_CLEAR_LOCK = "_redispatch_global_memory_clear_lock"
_RUNTIME_GLOBAL_MEMORY_LAST_CLEAR = "_redispatch_global_memory_last_clear"
_GLOBAL_MEMORY_CLEAR_COOLDOWN_SEC = 5.0
_WORKER_START_BARRIER_TIMEOUT_SEC = 900.0

_ORIGINAL_INIT_WORKER_CONTEXT = redispatch.init_worker_context
_ORIGINAL_RUN_PARALLEL = redispatch.run_parallel
_ORIGINAL_GLOBAL_MEMORY_GUARD = (
    redispatch.base.teacher.maybe_clear_heaviest_worker_for_global_memory
)


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

    _ORIGINAL_INIT_WORKER_CONTEXT(
        raw_dir_str,
        states_dir_str,
        runtime_task_config,
        scenario_ids,
        memory_registry,
    )

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

    clear_lock = cfg.get(_RUNTIME_GLOBAL_MEMORY_CLEAR_LOCK)
    last_clear = cfg.get(_RUNTIME_GLOBAL_MEMORY_LAST_CLEAR)

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
    runtime_task_config["clear_caches_every"] = _RUNTIME_CLEAR_CACHES_EVERY

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
    redispatch.base.teacher.maybe_clear_heaviest_worker_for_global_memory = (
        maybe_clear_heaviest_worker_for_global_memory
    )


def main() -> None:
    _install_staged_start()
    redispatch.main()


if __name__ == "__main__":
    main()

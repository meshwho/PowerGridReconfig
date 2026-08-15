from __future__ import annotations

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


_RUNTIME_LODF_STRUCTURE_CACHE = "_redispatch_lodf_structure_cache"
_DEFAULT_MAX_TASKS_PER_CHILD = 32

_ORIGINAL_INIT_WORKER_CONTEXT = redispatch.init_worker_context


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


def _bounded_worker_housekeeping() -> None:
    """Bounded caches and process recycling replace global cache clearing."""

    return None


def init_worker_context(
    raw_dir_str: str,
    states_dir_str: str,
    task_config: dict[str, Any],
    scenario_ids: Sequence[int],
    memory_registry=None,
) -> None:
    runtime_task_config = dict(task_config)

    _ORIGINAL_INIT_WORKER_CONTEXT(
        raw_dir_str,
        states_dir_str,
        runtime_task_config,
        scenario_ids,
        None,
    )

    teacher = redispatch.base.teacher
    ctx = teacher._require_worker_context()
    ctx.pop("memory_registry", None)
    ctx[_RUNTIME_LODF_STRUCTURE_CACHE] = (
        None
        if bool(runtime_task_config.get("disable_cache", False))
        else LODFStructureCache()
    )

    # Spawned workers import this module without running main(), so install the
    # runtime overrides explicitly in the initializer as well as in the parent.
    redispatch.base._worker_run_id = _worker_run_id
    teacher.rank_actions_by_lodf_screening = rank_actions_by_lodf_screening
    teacher.clear_worker_caches_if_needed = _bounded_worker_housekeeping


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


def _effective_max_tasks_per_child(task_config: dict[str, Any]) -> int:
    configured = int(task_config.get("max_tasks_per_child", 0))
    return configured if configured > 0 else _DEFAULT_MAX_TASKS_PER_CHILD


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
    teacher = redispatch.base.teacher
    shards = _partition_batches(scenario_batches, workers)
    workers = len(shards)
    shard_sizes = [_shard_scenario_ids(shard) for shard in shards]
    max_tasks_per_child = _effective_max_tasks_per_child(runtime_task_config)

    counts = [len(values) for values in shard_sizes]
    print(
        f"Partitioned adapters: {workers} workers, "
        f"{min(counts)}-{max(counts)} scenarios per worker"
    )
    print(f"Worker recycle interval: {max_tasks_per_child} batches")
    print("Worker cache policy: byte-bounded caches; no global cache clearing")
    print(f"\nParallel sharded mode: {workers} workers")
    print(f"Batches:                {len(scenario_batches)}")

    executors: list[ProcessPoolExecutor] = []
    futures = []
    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0

    try:
        for shard_batches, shard_scenarios in zip(shards, shard_sizes):
            executor = ProcessPoolExecutor(
                max_workers=1,
                initializer=init_worker_context,
                initargs=(
                    str(raw_dir),
                    str(states_dir),
                    runtime_task_config,
                    shard_scenarios,
                    None,
                ),
                max_tasks_per_child=max_tasks_per_child,
            )
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
    redispatch.base.teacher.clear_worker_caches_if_needed = (
        _bounded_worker_housekeeping
    )


def main() -> None:
    _install_staged_start()
    redispatch.main()


if __name__ == "__main__":
    main()

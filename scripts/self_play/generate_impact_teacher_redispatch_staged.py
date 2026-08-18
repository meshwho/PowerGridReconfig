from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Sequence

from grid_topology_ai.cache import LODFStructureCache
from grid_topology_ai.contracts import (
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    PHYSICS_CONFIG_CONTRACT_VERSION,
    TOPOLOGY_ACTION_CONTRACT_VERSION,
)
from grid_topology_ai.lodf import (
    build_lodf_structure,
    rank_actions_with_lodf_structure,
)
from grid_topology_ai.outcome_contract import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
)
from grid_topology_ai.teacher_config import (
    ensure_teacher_checkpoint_config,
    teacher_run_id,
    teacher_source_identity,
)
from grid_topology_ai.teacher_resume_index import (
    append_resume_delta,
    load_resume_index,
    write_resume_snapshot,
)
from scripts.self_play import generate_impact_teacher_redispatch as redispatch


_RUNTIME_LODF_STRUCTURE_CACHE = "_redispatch_lodf_structure_cache"
_DEFAULT_MAX_TASKS_PER_CHILD: int | None = None

_ORIGINAL_INIT_WORKER_CONTEXT = redispatch.init_worker_context
_ORIGINAL_IMPACT_BEAM_SEARCH_CONFIG = redispatch.base.teacher.ImpactBeamSearchConfig
_ORIGINAL_PROCESS_ONE_SCENARIO = redispatch.base.teacher.process_one_scenario_fast
_ORIGINAL_LOAD_SCENARIO_CHECKPOINTS = redispatch.base.load_scenario_checkpoints
_ORIGINAL_APPEND_SCENARIO_CHECKPOINT = redispatch.base.append_scenario_checkpoint


def _worker_run_id() -> str:
    teacher = redispatch.base.teacher
    ctx = teacher._require_worker_context()
    states_dir = Path(ctx["state_store"].output_dir)
    return teacher_run_id(states_dir, ctx["task_config"])


def _ensure_source_bound_checkpoint_config(
    config_path: Path,
    config: dict[str, Any],
) -> None:
    bound_config = dict(config)
    raw_dir = bound_config.get("raw_dir")
    transitions_path = bound_config.get("transitions_path")
    if raw_dir is None or transitions_path is None:
        raise ValueError(
            "Teacher checkpoint config requires raw_dir and transitions_path."
        )
    bound_config["source_identity"] = teacher_source_identity(
        raw_dir,
        transitions_path,
    )
    ensure_teacher_checkpoint_config(config_path, bound_config)


def _resume_contract_fingerprint() -> str:
    base = redispatch.base
    payload = {
        "checkpoint_version": int(base.teacher.CHECKPOINT_VERSION),
        "physics_config_contract_version": PHYSICS_CONFIG_CONTRACT_VERSION,
        "topology_action_contract_version": TOPOLOGY_ACTION_CONTRACT_VERSION,
        "outcome_objective_version": OUTCOME_OBJECTIVE_VERSION,
        "outcome_value_target_contract_version": (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        ),
        "terminal_outcome_evidence_schema_version": (
            TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
        ),
        "teacher_selection_mode": str(base._TEACHER_SELECTION_MODE),
        "required_checkpoint_row_fields": list(
            base._REQUIRED_CHECKPOINT_ROW_FIELDS
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_placeholder(scenario_id: int) -> dict[str, Any]:
    return {
        "version": redispatch.base.teacher.CHECKPOINT_VERSION,
        "scenario_id": int(scenario_id),
        "ok": False,
        "reason": "resume_index",
        "rows": [],
    }


def _load_scenario_checkpoints(
    checkpoint_path: Path,
    allowed_scenario_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    allowed = {int(value) for value in allowed_scenario_ids}
    contract_fingerprint = _resume_contract_fingerprint()
    indexed = load_resume_index(
        checkpoint_path=checkpoint_path,
        contract_fingerprint=contract_fingerprint,
        allowed_scenario_ids=allowed_scenario_ids,
    )

    # During an incomplete run the caller only needs the completed IDs to build
    # the pending list. Keep the expensive checkpoint rows on disk until final
    # assembly, when every requested scenario is already checkpointed.
    if indexed is not None and indexed != allowed:
        return {
            scenario_id: _resume_placeholder(scenario_id)
            for scenario_id in sorted(indexed)
        }

    results = _ORIGINAL_LOAD_SCENARIO_CHECKPOINTS(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=allowed_scenario_ids,
    )

    try:
        write_resume_snapshot(
            checkpoint_path=checkpoint_path,
            contract_fingerprint=contract_fingerprint,
            completed_scenario_ids=results,
        )
    except OSError:
        pass

    return results


def _append_scenario_checkpoint(
    checkpoint_path: Path,
    result: dict[str, Any],
) -> None:
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint_start = int(checkpoint_path.stat().st_size)
    except FileNotFoundError:
        checkpoint_start = 0

    _ORIGINAL_APPEND_SCENARIO_CHECKPOINT(
        checkpoint_path=checkpoint_path,
        result=result,
    )

    try:
        append_resume_delta(
            checkpoint_path=checkpoint_path,
            contract_fingerprint=_resume_contract_fingerprint(),
            scenario_id=int(result["scenario_id"]),
            complete=bool(
                redispatch.base._checkpoint_result_is_current(result)
            ),
            checkpoint_start=checkpoint_start,
        )
    except (OSError, ValueError):
        # The sidecar is only an accelerator. A stale or missing index falls
        # back to the canonical checkpoint scan on the next resume.
        pass


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
    """Bounded caches and process lifetime replace global cache clearing."""

    return None


def _quiet_impact_beam_search_config(*args, **kwargs):
    kwargs["show_progress"] = False
    return _ORIGINAL_IMPACT_BEAM_SEARCH_CONFIG(*args, **kwargs)


def _timed_process_one_scenario(scenario_id: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = _ORIGINAL_PROCESS_ONE_SCENARIO(int(scenario_id))
    result["runtime_seconds"] = time.perf_counter() - started
    return result


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
    teacher.ImpactBeamSearchConfig = _quiet_impact_beam_search_config
    teacher.process_one_scenario_fast = _timed_process_one_scenario
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
    del verbose_success

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
        else:
            skipped += 1

    return saved, skipped


def _run_timed_batch(
    process_batch: Callable[[list[int]], list[dict[str, Any]]],
    batch: list[int],
) -> tuple[list[dict[str, Any]], float]:
    """Run one worker batch quietly and return its actual worker wall time."""

    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            results = process_batch(batch)
    return results, time.perf_counter() - started


def _effective_max_tasks_per_child(
    task_config: dict[str, Any],
) -> int | None:
    configured = int(task_config.get("max_tasks_per_child", 0))
    return configured if configured > 0 else _DEFAULT_MAX_TASKS_PER_CHILD


def _scenario_runtime_line(result: dict[str, Any]) -> str:
    scenario_id = int(result["scenario_id"])
    seconds = float(result.get("runtime_seconds", 0.0))
    if bool(result.get("ok", False)):
        status = "saved"
    else:
        reason = result.get("reason")
        status = "skipped" if reason is None else f"skipped ({reason})"
    return f"scenario {scenario_id} | {seconds:.1f}s | {status}"


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
    recycle_text = (
        "disabled"
        if max_tasks_per_child is None
        else f"{max_tasks_per_child} batches"
    )
    print(f"Worker recycle interval: {recycle_text}")
    print("Worker cache policy: byte-bounded caches; no global cache clearing")
    print(f"\nParallel sharded mode: {workers} workers")
    print(f"Batches:                {len(scenario_batches)}")

    executors: list[ProcessPoolExecutor] = []
    futures = []
    rows: list[dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0
    completed_batches = 0

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
                    executor.submit(
                        _run_timed_batch,
                        teacher.process_scenario_batch,
                        batch,
                    )
                )

        progress_bar = None
        iterator = as_completed(futures)
        if teacher.tqdm is not None:
            progress_bar = teacher.tqdm(
                iterator,
                total=len(futures),
                desc="Teacher batches",
                unit="batch",
                dynamic_ncols=True,
            )
            iterator = progress_bar

        for future in iterator:
            batch_results, batch_seconds = future.result()
            saved, skipped = _handle_batch_results(
                batch_results,
                rows=rows,
                checkpoint_path=checkpoint_path,
                verbose_success=verbose_success,
            )
            total_saved += saved
            total_skipped += skipped
            completed_batches += 1

            for result in batch_results:
                line = _scenario_runtime_line(result)
                if progress_bar is not None:
                    progress_bar.write(line)
                else:
                    print(line, flush=True)

            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "worker": f"{batch_seconds:.1f}s",
                        "saved": saved,
                        "skipped": skipped,
                    },
                    refresh=True,
                )
            else:
                print(
                    f"Teacher batch {completed_batches}/{len(futures)} | "
                    f"worker={batch_seconds:.1f}s | "
                    f"saved={saved} | skipped={skipped}",
                    flush=True,
                )
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
    redispatch.base.load_scenario_checkpoints = _load_scenario_checkpoints
    redispatch.base.append_scenario_checkpoint = _append_scenario_checkpoint
    redispatch.base.teacher.ensure_checkpoint_config = (
        _ensure_source_bound_checkpoint_config
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

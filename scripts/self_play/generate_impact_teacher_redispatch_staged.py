from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Sequence

from scripts.self_play import generate_impact_teacher_redispatch as redispatch


_RUNTIME_READY_EVENT = "_redispatch_worker_ready_event"
_RUNTIME_READY_COUNT = "_redispatch_worker_ready_count"
_RUNTIME_READY_LOCK = "_redispatch_worker_ready_lock"
_RUNTIME_EXPECTED_WORKERS = "_redispatch_expected_workers"
_WORKER_START_BARRIER_TIMEOUT_SEC = 900.0

_ORIGINAL_INIT_WORKER_CONTEXT = redispatch.init_worker_context
_ORIGINAL_RUN_PARALLEL = redispatch.run_parallel


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
    workers = max(int(num_workers), 1)

    if workers == 1:
        return _ORIGINAL_RUN_PARALLEL(
            scenario_batches=scenario_batches,
            scenario_ids=scenario_ids,
            raw_dir=raw_dir,
            states_dir=states_dir,
            task_config=task_config,
            checkpoint_path=checkpoint_path,
            num_workers=workers,
            verbose_success=verbose_success,
        )

    runtime_task_config = dict(task_config)
    runtime_task_config[_RUNTIME_READY_EVENT] = mp.Event()
    runtime_task_config[_RUNTIME_READY_COUNT] = mp.Value("i", 0)
    runtime_task_config[_RUNTIME_READY_LOCK] = mp.Lock()
    runtime_task_config[_RUNTIME_EXPECTED_WORKERS] = workers

    print(
        f"Worker start barrier: {workers} workers must initialize "
        "before beam search starts"
    )

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


def _install_staged_start() -> None:
    redispatch.init_worker_context = init_worker_context
    redispatch.run_parallel = run_parallel


def main() -> None:
    _install_staged_start()
    redispatch.main()


if __name__ == "__main__":
    main()

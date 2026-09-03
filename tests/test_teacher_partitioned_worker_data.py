from __future__ import annotations

from pathlib import Path

import grid_topology_ai.teacher_runtime as teacher


def test_parallel_runner_submits_batches_to_shared_worker_pool(monkeypatch) -> None:
    batches = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
    ]
    executors = []

    class FakeFuture:
        def result(self):
            return [], 0.0

    class FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.submitted = []
            executors.append(self)

        def submit(self, fn, process_batch, batch):
            assert fn is teacher._run_timed_batch
            self.submitted.append(list(batch))
            return FakeFuture()

        def shutdown(self, *, wait, cancel_futures):
            return None

    monkeypatch.setattr(
        teacher,
        "ensure_runtime_scenario_store",
        lambda raw_dir: Path("runtime-store"),
    )
    monkeypatch.setattr(teacher, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(teacher, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(teacher, "_worker_init_concurrency", lambda: 3)
    monkeypatch.setattr(teacher, "tqdm", None)
    monkeypatch.setattr(teacher, "process_scenario_batch", lambda batch: [])

    result = teacher.run_parallel(
        scenario_batches=batches,
        scenario_ids=list(range(1, 13)),
        raw_dir="raw",
        states_dir="states",
        task_config={},
        checkpoint_path="checkpoint.jsonl",
        num_workers=3,
        verbose_success=False,
    )

    assert result == ([], 0, 0)
    assert len(executors) == 1

    executor = executors[0]
    assert executor.kwargs["max_workers"] == 3
    assert "max_tasks_per_child" not in executor.kwargs
    assert executor.kwargs["initargs"][3] == tuple(range(1, 13))
    assert executor.submitted == batches

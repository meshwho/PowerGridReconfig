from __future__ import annotations

from pathlib import Path

from scripts.self_play import generate_impact_teacher_redispatch_runtime as teacher


def test_partition_batches_keeps_each_scenario_in_one_worker_shard() -> None:
    batches = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
    ]

    shards = teacher._partition_batches(batches, 3)

    assert shards == [
        [[1, 2], [7, 8]],
        [[3, 4], [9, 10]],
        [[5, 6], [11, 12]],
    ]
    assert teacher._shard_scenario_ids(shards[0]) == (1, 2, 7, 8)
    assert teacher._shard_scenario_ids(shards[1]) == (3, 4, 9, 10)
    assert teacher._shard_scenario_ids(shards[2]) == (5, 6, 11, 12)


def test_parallel_runner_routes_batches_to_matching_adapter_shards(monkeypatch) -> None:
    batches = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
    ]
    task_config = {
        "min_free_system_memory_mb": 0.0,
        "max_tasks_per_child": 0,
    }
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
        task_config=task_config,
        checkpoint_path="checkpoint.jsonl",
        num_workers=3,
        verbose_success=False,
    )

    assert result == ([], 0, 0)
    assert task_config == {
        "min_free_system_memory_mb": 0.0,
        "max_tasks_per_child": 0,
    }
    assert len(executors) == 3

    expected = [
        ((1, 2, 7, 8), [[1, 2], [7, 8]]),
        ((3, 4, 9, 10), [[3, 4], [9, 10]]),
        ((5, 6, 11, 12), [[5, 6], [11, 12]]),
    ]

    for executor, (scenario_ids, submitted) in zip(executors, expected):
        assert executor.kwargs["max_workers"] == 1
        assert executor.kwargs["initargs"][3] == scenario_ids
        assert executor.submitted == submitted

        adapter_scenarios = set(scenario_ids)
        for batch in submitted:
            assert set(batch) <= adapter_scenarios

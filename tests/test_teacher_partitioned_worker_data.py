from __future__ import annotations

from scripts.self_play import generate_impact_teacher_redispatch_staged as staged


def test_partition_batches_keeps_each_scenario_in_one_worker_shard() -> None:
    batches = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
    ]

    shards = staged._partition_batches(batches, 3)

    assert shards == [
        [[1, 2], [7, 8]],
        [[3, 4], [9, 10]],
        [[5, 6], [11, 12]],
    ]
    assert staged._shard_scenario_ids(shards[0]) == (1, 2, 7, 8)
    assert staged._shard_scenario_ids(shards[1]) == (3, 4, 9, 10)
    assert staged._shard_scenario_ids(shards[2]) == (5, 6, 11, 12)


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
            return []

    class FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.submitted = []
            executors.append(self)

        def submit(self, fn, batch):
            self.submitted.append(list(batch))
            return FakeFuture()

        def shutdown(self, *, wait, cancel_futures):
            return None

    teacher = staged.redispatch.base.teacher
    monkeypatch.setattr(staged, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(staged, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(staged.redispatch, "_worker_init_concurrency", lambda: 1)
    monkeypatch.setattr(teacher, "tqdm", None)
    monkeypatch.setattr(teacher, "process_scenario_batch", lambda batch: [])

    result = staged.run_parallel(
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
        (((1, 2, 7, 8)), [[1, 2], [7, 8]]),
        (((3, 4, 9, 10)), [[3, 4], [9, 10]]),
        (((5, 6, 11, 12)), [[5, 6], [11, 12]]),
    ]

    for executor, (scenario_ids, submitted) in zip(executors, expected):
        assert executor.kwargs["max_workers"] == 1
        assert executor.kwargs["initargs"][3] == scenario_ids
        assert executor.submitted == submitted

        adapter_scenarios = set(scenario_ids)
        for batch in submitted:
            assert set(batch) <= adapter_scenarios

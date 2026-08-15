from __future__ import annotations

from pathlib import Path

from scripts.pipelines import run_teacher_redispatch as entrypoint
from scripts.self_play import generate_impact_teacher_redispatch_staged as staged


def test_production_entrypoint_uses_runtime_teacher_module() -> None:
    assert entrypoint.REDISPATCH_TEACHER_MODULE == (
        "scripts.self_play.generate_impact_teacher_redispatch_runtime"
    )


def test_worker_waits_for_pool_after_initialization(monkeypatch) -> None:
    events: list[str] = []

    class FakeEvent:
        def __init__(self) -> None:
            self.ready = False

        def is_set(self) -> bool:
            return self.ready

        def set(self) -> None:
            events.append("set")
            self.ready = True

        def wait(self, timeout=None) -> bool:
            events.append("wait")
            return self.ready

    class FakeCount:
        value = 0

    class FakeLock:
        def __enter__(self):
            events.append("lock-enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("lock-exit")

    ready_event = FakeEvent()
    ready_count = FakeCount()
    ready_lock = FakeLock()

    def fake_init(
        raw_dir_str,
        states_dir_str,
        task_config,
        scenario_ids,
        memory_registry,
    ) -> None:
        assert staged._RUNTIME_READY_EVENT not in task_config
        assert staged._RUNTIME_READY_COUNT not in task_config
        assert staged._RUNTIME_READY_LOCK not in task_config
        assert staged._RUNTIME_EXPECTED_WORKERS not in task_config
        events.append("init")

    monkeypatch.setattr(
        staged,
        "_ORIGINAL_INIT_WORKER_CONTEXT",
        fake_init,
    )

    staged.init_worker_context(
        raw_dir_str="raw",
        states_dir_str="states",
        task_config={
            staged._RUNTIME_READY_EVENT: ready_event,
            staged._RUNTIME_READY_COUNT: ready_count,
            staged._RUNTIME_READY_LOCK: ready_lock,
            staged._RUNTIME_EXPECTED_WORKERS: 1,
        },
        scenario_ids=[1],
        memory_registry=None,
    )

    assert events == [
        "init",
        "lock-enter",
        "set",
        "lock-exit",
        "wait",
    ]
    assert ready_count.value == 1


def test_parallel_start_barrier_is_runtime_only(monkeypatch, tmp_path: Path) -> None:
    event = object()
    count = object()
    lock = object()
    semaphore = object()
    executors = []

    class FakeFuture:
        def result(self):
            return []

    class FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            executors.append(self)

        def submit(self, fn, batch):
            return FakeFuture()

        def shutdown(self, *, wait, cancel_futures):
            return None

    monkeypatch.setattr(staged.mp, "Event", lambda: event)
    monkeypatch.setattr(staged.mp, "Value", lambda kind, value: count)
    monkeypatch.setattr(staged.mp, "Lock", lambda: lock)
    monkeypatch.setattr(staged.mp, "BoundedSemaphore", lambda value: semaphore)
    monkeypatch.setattr(staged, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(staged, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(staged.redispatch, "_worker_init_concurrency", lambda: 1)
    monkeypatch.setattr(staged.redispatch.base.teacher, "tqdm", None)

    task_config = {
        "physics_config_fingerprint": "stable",
        "min_free_system_memory_mb": 0.0,
        "max_tasks_per_child": 0,
    }

    result = staged.run_parallel(
        scenario_batches=[[1], [2]],
        scenario_ids=[1, 2],
        raw_dir=tmp_path,
        states_dir=tmp_path / "states",
        task_config=task_config,
        checkpoint_path=tmp_path / "teacher_checkpoint.jsonl",
        num_workers=10,
        verbose_success=False,
    )

    assert result == ([], 0, 0)
    assert task_config == {
        "physics_config_fingerprint": "stable",
        "min_free_system_memory_mb": 0.0,
        "max_tasks_per_child": 0,
    }
    assert len(executors) == 2

    for executor in executors:
        runtime_config = executor.kwargs["initargs"][2]
        assert runtime_config[staged._RUNTIME_READY_EVENT] is event
        assert runtime_config[staged._RUNTIME_READY_COUNT] is count
        assert runtime_config[staged._RUNTIME_READY_LOCK] is lock
        assert runtime_config[staged._RUNTIME_EXPECTED_WORKERS] == 2
        assert (
            runtime_config[staged.redispatch._RUNTIME_WORKER_INIT_SEMAPHORE]
            is semaphore
        )

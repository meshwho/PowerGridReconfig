from __future__ import annotations

from pathlib import Path

from scripts.pipelines import run_teacher_redispatch as entrypoint
from scripts.self_play import generate_impact_teacher_redispatch_staged as staged


def test_production_entrypoint_uses_staged_teacher_module() -> None:
    assert entrypoint.REDISPATCH_TEACHER_MODULE == (
        "scripts.self_play.generate_impact_teacher_redispatch_staged"
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
    captured: dict[str, object] = {}

    monkeypatch.setattr(staged.mp, "Event", lambda: event)
    monkeypatch.setattr(staged.mp, "Value", lambda kind, value: count)
    monkeypatch.setattr(staged.mp, "Lock", lambda: lock)

    def fake_run_parallel(**kwargs):
        captured.update(kwargs)
        return [], 0, 0

    monkeypatch.setattr(
        staged,
        "_ORIGINAL_RUN_PARALLEL",
        fake_run_parallel,
    )

    task_config = {"physics_config_fingerprint": "stable"}

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
    assert task_config == {"physics_config_fingerprint": "stable"}

    runtime_config = captured["task_config"]
    assert isinstance(runtime_config, dict)
    assert runtime_config[staged._RUNTIME_READY_EVENT] is event
    assert runtime_config[staged._RUNTIME_READY_COUNT] is count
    assert runtime_config[staged._RUNTIME_READY_LOCK] is lock
    assert runtime_config[staged._RUNTIME_EXPECTED_WORKERS] == 10

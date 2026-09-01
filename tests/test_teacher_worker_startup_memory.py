from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from grid_topology_ai.data import GridFMAdapter
import grid_topology_ai.teacher_runtime as teacher


def test_worker_init_semaphore_wraps_heavy_context_initialization(monkeypatch) -> None:
    events: list[str] = []

    class FakeSemaphore:
        def acquire(self) -> None:
            events.append("acquire")

        def release(self) -> None:
            events.append("release")

    semaphore = FakeSemaphore()
    previous_context = teacher._WORKER_CONTEXT

    def fake_build_context(**kwargs):
        assert teacher._RUNTIME_WORKER_INIT_SEMAPHORE not in kwargs["task_config"]
        events.append("init")
        return {
            "task_config": dict(kwargs["task_config"]),
            "state_store": SimpleNamespace(output_dir=Path("states")),
        }

    monkeypatch.setattr(
        teacher,
        "ensure_runtime_scenario_store",
        lambda raw_dir: Path("runtime-store"),
    )
    monkeypatch.setattr(
        teacher,
        "build_memory_mapped_teacher_context",
        fake_build_context,
    )
    monkeypatch.setattr(teacher.gc, "collect", lambda: events.append("gc"))

    try:
        teacher.init_worker_context(
            raw_dir_str="raw",
            states_dir_str="states",
            task_config={
                "physics": "unchanged",
                teacher._RUNTIME_WORKER_INIT_SEMAPHORE: semaphore,
            },
            scenario_ids=[1, 2],
            memory_registry=None,
        )
    finally:
        teacher._WORKER_CONTEXT = previous_context

    assert events == ["acquire", "init", "gc", "release"]


def test_parallel_runtime_adds_init_semaphore_without_mutating_task_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sentinel = object()
    executors = []

    class FakeFuture:
        def result(self):
            return [], 0.0

    class FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            executors.append(self)

        def submit(self, fn, process_batch, batch):
            assert fn is teacher._run_timed_batch
            return FakeFuture()

        def shutdown(self, *, wait, cancel_futures):
            return None

    monkeypatch.setenv(teacher._WORKER_INIT_CONCURRENCY_ENV, "1")
    monkeypatch.setattr(
        teacher.mp,
        "BoundedSemaphore",
        lambda value: sentinel if value == 1 else None,
    )
    monkeypatch.setattr(
        teacher,
        "ensure_runtime_scenario_store",
        lambda raw_dir: tmp_path / "runtime-store",
    )
    monkeypatch.setattr(teacher, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(teacher, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(teacher, "tqdm", None)

    task_config = {"physics_config_fingerprint": "stable"}
    result = teacher.run_parallel(
        scenario_batches=[[1], [2]],
        scenario_ids=[1, 2],
        raw_dir=tmp_path,
        states_dir=tmp_path / "states",
        task_config=task_config,
        checkpoint_path=tmp_path / "teacher_checkpoint.jsonl",
        num_workers=8,
        verbose_success=False,
    )

    assert result == ([], 0, 0)
    assert task_config == {"physics_config_fingerprint": "stable"}
    assert len(executors) == 1
    executor = executors[0]
    assert executor.kwargs["max_workers"] == 2
    runtime_config = executor.kwargs["initargs"][2]
    assert runtime_config[teacher._RUNTIME_WORKER_INIT_SEMAPHORE] is sentinel


def test_parquet_reader_returns_loaded_frame_without_reset_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "bus_data.parquet"
    path.touch()
    frame = pd.DataFrame({"scenario": [7]}, index=pd.Index([123]))

    adapter = object.__new__(GridFMAdapter)
    adapter.raw_dir = tmp_path
    adapter._scenario_filter = None

    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: frame)

    loaded = adapter._read_required_parquet("bus_data.parquet")

    assert loaded is frame
    assert loaded.index.tolist() == [123]

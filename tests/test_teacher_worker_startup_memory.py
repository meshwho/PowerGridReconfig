from __future__ import annotations

from pathlib import Path

import pandas as pd

from grid_topology_ai.data_adapter import GridFMAdapter
from scripts.self_play import generate_impact_teacher_redispatch as redispatch


def test_worker_init_semaphore_wraps_heavy_context_initialization(monkeypatch) -> None:
    events: list[str] = []

    class FakeSemaphore:
        def acquire(self) -> None:
            events.append("acquire")

        def release(self) -> None:
            events.append("release")

    semaphore = FakeSemaphore()

    def fake_init(
        raw_dir_str,
        states_dir_str,
        task_config,
        scenario_ids,
        memory_registry,
    ) -> None:
        assert redispatch._RUNTIME_WORKER_INIT_SEMAPHORE not in task_config
        events.append("init")

    monkeypatch.setattr(
        redispatch,
        "_ORIGINAL_INIT_WORKER_CONTEXT",
        fake_init,
    )
    monkeypatch.setattr(
        redispatch.gc,
        "collect",
        lambda: events.append("gc"),
    )

    redispatch.init_worker_context(
        raw_dir_str="raw",
        states_dir_str="states",
        task_config={
            "physics": "unchanged",
            redispatch._RUNTIME_WORKER_INIT_SEMAPHORE: semaphore,
        },
        scenario_ids=[1, 2],
        memory_registry=None,
    )

    assert events == ["acquire", "init", "gc", "release"]


def test_parallel_runtime_adds_init_semaphore_without_changing_checkpoint_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    monkeypatch.setenv(
        redispatch._WORKER_INIT_CONCURRENCY_ENV,
        "1",
    )
    monkeypatch.setattr(
        redispatch.mp,
        "BoundedSemaphore",
        lambda value: sentinel if value == 1 else None,
    )

    def fake_run_parallel(**kwargs):
        captured.update(kwargs)
        return [], 0, 0

    monkeypatch.setattr(
        redispatch,
        "_ORIGINAL_RUN_PARALLEL",
        fake_run_parallel,
    )

    task_config = {"physics_config_fingerprint": "stable"}

    result = redispatch.run_parallel(
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
    runtime_config = captured["task_config"]
    assert isinstance(runtime_config, dict)
    assert runtime_config[redispatch._RUNTIME_WORKER_INIT_SEMAPHORE] is sentinel


def test_parquet_reader_returns_loaded_frame_without_reset_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "bus_data.parquet"
    path.touch()

    frame = pd.DataFrame(
        {"scenario": [7]},
        index=pd.Index([123]),
    )

    adapter = object.__new__(GridFMAdapter)
    adapter.raw_dir = tmp_path
    adapter._scenario_filter = None

    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: frame,
    )

    loaded = adapter._read_required_parquet("bus_data.parquet")

    assert loaded is frame
    assert loaded.index.tolist() == [123]

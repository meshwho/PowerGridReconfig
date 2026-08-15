from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.pipelines import run_teacher_redispatch as entrypoint
from scripts.self_play import generate_impact_teacher_redispatch_staged as staged


def test_production_entrypoint_uses_runtime_teacher_module() -> None:
    assert entrypoint.REDISPATCH_TEACHER_MODULE == (
        "scripts.self_play.generate_impact_teacher_redispatch_runtime"
    )


def test_worker_init_installs_bounded_runtime_policy(monkeypatch) -> None:
    teacher = staged.redispatch.base.teacher
    previous_context = getattr(teacher, "_WORKER_CONTEXT", None)

    def fake_init(
        raw_dir_str,
        states_dir_str,
        task_config,
        scenario_ids,
        memory_registry,
    ) -> None:
        assert memory_registry is None
        teacher._WORKER_CONTEXT = {
            "task_config": dict(task_config),
            "state_store": SimpleNamespace(output_dir=Path("states")),
            "memory_registry": object(),
        }

    monkeypatch.setattr(staged, "_ORIGINAL_INIT_WORKER_CONTEXT", fake_init)
    try:
        staged.init_worker_context(
            raw_dir_str="raw",
            states_dir_str="states",
            task_config={"disable_cache": False},
            scenario_ids=[1],
            memory_registry=object(),
        )
        context = teacher._WORKER_CONTEXT
        assert "memory_registry" not in context
        assert isinstance(
            context[staged._RUNTIME_LODF_STRUCTURE_CACHE],
            staged.LODFStructureCache,
        )
        assert (
            teacher.clear_worker_caches_if_needed
            is staged._bounded_worker_housekeeping
        )
    finally:
        teacher._WORKER_CONTEXT = previous_context


def test_parallel_workers_are_recycled_by_default(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr(staged, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(staged, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(staged.redispatch.base.teacher, "tqdm", None)

    task_config = {
        "physics_config_fingerprint": "stable",
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
    assert task_config["max_tasks_per_child"] == 0
    assert len(executors) == 2
    assert all(
        executor.kwargs["max_tasks_per_child"]
        == staged._DEFAULT_MAX_TASKS_PER_CHILD
        for executor in executors
    )
    assert all(executor.kwargs["initargs"][-1] is None for executor in executors)


def test_explicit_worker_recycle_interval_is_respected() -> None:
    assert staged._effective_max_tasks_per_child({"max_tasks_per_child": 7}) == 7
    assert (
        staged._effective_max_tasks_per_child({"max_tasks_per_child": 0})
        == staged._DEFAULT_MAX_TASKS_PER_CHILD
    )

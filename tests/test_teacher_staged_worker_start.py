from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from scripts.pipelines import run_teacher_redispatch as entrypoint
from scripts.self_play import generate_impact_teacher_redispatch_runtime as teacher


def test_production_entrypoint_uses_runtime_teacher_module() -> None:
    assert entrypoint.REDISPATCH_TEACHER_MODULE == (
        "scripts.self_play.generate_impact_teacher_redispatch_runtime"
    )


def test_native_thread_defaults_are_applied_before_project_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "scripts"
        / "self_play"
        / "generate_impact_teacher_redispatch_runtime.py"
    ).read_text(encoding="utf-8")
    ast.parse(source)

    configure_call = source.index("_configure_native_math_threads()")
    first_project_import = source.index("from grid_topology_ai")
    assert configure_call < first_project_import

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert f'"{name}"' in source

    assert 'os.environ.setdefault(name, "1")' in source


def test_worker_init_installs_bounded_runtime_policy(monkeypatch) -> None:
    previous_context = teacher._WORKER_CONTEXT

    def fake_context(**kwargs):
        assert kwargs["memory_registry"] is None
        return {
            "task_config": dict(kwargs["task_config"]),
            "state_store": SimpleNamespace(output_dir=Path("states")),
            "memory_registry": object(),
        }

    monkeypatch.setattr(
        teacher,
        "ensure_runtime_scenario_store",
        lambda raw_dir: Path("runtime-store"),
    )
    monkeypatch.setattr(
        teacher,
        "build_memory_mapped_teacher_context",
        fake_context,
    )

    try:
        teacher.init_worker_context(
            raw_dir_str="raw",
            states_dir_str="states",
            task_config={"disable_cache": False},
            scenario_ids=[1],
            memory_registry=object(),
        )
        context = teacher._WORKER_CONTEXT
        assert context is not None
        assert "memory_registry" not in context
        assert isinstance(
            context[teacher._RUNTIME_LODF_STRUCTURE_CACHE],
            teacher.LODFStructureCache,
        )
        assert teacher.clear_worker_caches_if_needed() is None
    finally:
        teacher._WORKER_CONTEXT = previous_context


def test_parallel_workers_are_not_recycled_by_default(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr(
        teacher,
        "ensure_runtime_scenario_store",
        lambda raw_dir: tmp_path / "runtime-store",
    )
    monkeypatch.setattr(teacher, "_worker_init_concurrency", lambda: 2)
    monkeypatch.setattr(teacher, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(teacher, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(teacher, "tqdm", None)

    task_config = {
        "physics_config_fingerprint": "stable",
        "max_tasks_per_child": 0,
    }
    result = teacher.run_parallel(
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
        executor.kwargs["max_tasks_per_child"] is None
        for executor in executors
    )
    assert all(executor.kwargs["initargs"][-1] is None for executor in executors)


def test_explicit_worker_recycle_interval_is_respected() -> None:
    assert teacher._effective_max_tasks_per_child({"max_tasks_per_child": 7}) == 7
    assert (
        teacher._effective_max_tasks_per_child({"max_tasks_per_child": 0})
        == teacher._DEFAULT_MAX_TASKS_PER_CHILD
    )

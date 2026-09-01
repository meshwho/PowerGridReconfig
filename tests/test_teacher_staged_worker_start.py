from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path
from types import SimpleNamespace

import grid_topology_ai.cli as light_cli
import grid_topology_ai.teacher_runtime as teacher


def test_unified_cli_uses_packaged_teacher_runtime() -> None:
    source = inspect.getsource(light_cli._teacher)
    assert "from grid_topology_ai.teacher_runtime import main as teacher_main" in source


def test_native_thread_defaults_are_applied_before_project_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "grid_topology_ai" / "teacher_runtime.py").read_text(
        encoding="utf-8"
    )
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


def test_worker_entrypoints_are_importable_and_picklable() -> None:
    entrypoints = teacher._multiprocessing_entrypoints()

    assert entrypoints == (
        teacher.init_worker_context,
        teacher._run_timed_batch,
        teacher.process_scenario_batch,
    )
    for entrypoint in entrypoints:
        assert entrypoint.__module__ == "grid_topology_ai.teacher_runtime"
        assert pickle.loads(pickle.dumps(entrypoint)) is entrypoint


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


def test_parallel_workers_use_persistent_pool_without_recycling(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
            assert process_batch is teacher.process_scenario_batch
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

    result = teacher.run_parallel(
        scenario_batches=[[1], [2]],
        scenario_ids=[1, 2],
        raw_dir=tmp_path,
        states_dir=tmp_path / "states",
        task_config={"disable_cache": False},
        checkpoint_path=tmp_path / "teacher_checkpoint.jsonl",
        num_workers=10,
        verbose_success=False,
    )

    assert result == ([], 0, 0)
    assert len(executors) == 1

    executor = executors[0]
    assert executor.kwargs["max_workers"] == 2
    assert executor.kwargs["initializer"] is teacher.init_worker_context
    assert "max_tasks_per_child" not in executor.kwargs
    assert executor.kwargs["initargs"][3] == (1, 2)
    assert executor.kwargs["initargs"][-1] is None

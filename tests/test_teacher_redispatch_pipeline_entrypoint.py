from __future__ import annotations

from pathlib import Path

from scripts.pipelines import run_teacher_redispatch as entrypoint
from scripts.self_play import generate_impact_teacher_redispatch_staged as staged


def test_entrypoint_adds_redispatch_teacher_and_full_run_name() -> None:
    argv = entrypoint.canonical_argv(
        [
            "run_teacher_redispatch.py",
            "--dataset-name",
            "case118_bootstrap_v1",
        ]
    )

    assert argv[argv.index("--teacher-module") + 1] == (
        entrypoint.REDISPATCH_TEACHER_MODULE
    )
    assert argv[argv.index("--run-name") + 1] == (
        "case118_bootstrap_v1_teacher_redispatch_v1"
    )


def test_entrypoint_uses_separate_smoke_run_name() -> None:
    argv = entrypoint.canonical_argv(
        [
            "run_teacher_redispatch.py",
            "--dataset-name",
            "case118_bootstrap_v1",
            "--profile",
            "smoke",
        ]
    )

    assert argv[argv.index("--run-name") + 1] == (
        "case118_bootstrap_v1_teacher_redispatch_smoke_v1"
    )


def test_entrypoint_replaces_auto_workers_with_safe_count(monkeypatch) -> None:
    monkeypatch.setattr(entrypoint, "_safe_auto_workers", lambda: 3)

    argv = entrypoint.canonical_argv(
        [
            "run_teacher_redispatch.py",
            "--dataset-name",
            "case118_bootstrap_v1",
            "--num-workers",
            "auto",
        ]
    )

    assert argv[argv.index("--num-workers") + 1] == "3"


def test_entrypoint_preserves_explicit_worker_count(monkeypatch) -> None:
    monkeypatch.setattr(entrypoint, "_safe_auto_workers", lambda: 1)

    argv = entrypoint.canonical_argv(
        [
            "run_teacher_redispatch.py",
            "--dataset-name",
            "case118_bootstrap_v1",
            "--num-workers",
            "2",
        ]
    )

    assert argv[argv.index("--num-workers") + 1] == "2"


def test_entrypoint_extracts_worker_init_concurrency() -> None:
    argv = [
        "run_teacher_redispatch.py",
        "--dataset-name",
        "case118_bootstrap_v1",
        "--worker-init-concurrency",
        "2",
    ]

    concurrency = entrypoint._pop_worker_init_concurrency(argv)

    assert concurrency == 2
    assert "--worker-init-concurrency" not in argv
    assert "2" not in argv


def test_entrypoint_defaults_to_serial_worker_initialization() -> None:
    argv = [
        "run_teacher_redispatch.py",
        "--dataset-name",
        "case118_bootstrap_v1",
    ]

    assert entrypoint._pop_worker_init_concurrency(argv) == 1


def test_entrypoint_extracts_runtime_pf_cache_dir() -> None:
    argv = [
        "run_teacher_redispatch.py",
        "--dataset-name",
        "case118_bootstrap_v1",
        "--pf-cache-dir",
        "cache/power-flow",
    ]

    cache_dir = entrypoint._pop_pf_cache_dir(argv)

    assert cache_dir == "cache/power-flow"
    assert "--pf-cache-dir" not in argv
    assert "cache/power-flow" not in argv


def test_pf_cache_dir_is_removed_before_pipeline_checkpoint_config(monkeypatch) -> None:
    captured: dict[str, object] = {}
    argv = [
        "run_teacher_redispatch.py",
        "--dataset-name",
        "case118_bootstrap_v1",
        "--num-workers",
        "2",
        "--pf-cache-dir",
        "cache/power-flow",
    ]

    def fake_pipeline_main() -> None:
        captured["argv"] = list(entrypoint.sys.argv)
        captured["cache_dir"] = entrypoint.os.environ.get(
            entrypoint._PF_CACHE_DIR_ENV
        )

    monkeypatch.delenv(entrypoint._PF_CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(entrypoint.sys, "argv", argv)
    monkeypatch.setattr(entrypoint.pipeline, "main", fake_pipeline_main)

    entrypoint.main()

    forwarded = captured["argv"]
    assert isinstance(forwarded, list)
    assert "--pf-cache-dir" not in forwarded
    assert "cache/power-flow" not in forwarded
    assert captured["cache_dir"] == "cache/power-flow"


def test_staged_worker_attaches_persistent_cache_from_runtime_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeBackend:
        enable_cache = True
        _persistent_exact_cache = None

    backend = FakeBackend()
    worker_context = {"backend": backend}
    cache_root = tmp_path / "pf-cache"

    monkeypatch.setenv(staged._PF_CACHE_DIR_ENV, str(cache_root))
    monkeypatch.setattr(
        staged.redispatch.base.teacher,
        "_require_worker_context",
        lambda: worker_context,
    )

    staged._attach_persistent_pf_cache()

    store = backend._persistent_exact_cache
    assert store is not None
    assert store.root == cache_root
    assert store.namespace == "exact"
    assert store.database_path == cache_root / "exact" / "cache.sqlite3"
    store.close()


def test_staged_worker_skips_pf_cache_until_worker_context_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(staged._PF_CACHE_DIR_ENV, str(tmp_path / "pf-cache"))

    def missing_context():
        raise RuntimeError("worker context is not initialized")

    monkeypatch.setattr(
        staged.redispatch.base.teacher,
        "_require_worker_context",
        missing_context,
    )

    staged._attach_persistent_pf_cache()


def test_staged_worker_does_not_attach_disk_cache_when_cache_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeBackend:
        enable_cache = False
        _persistent_exact_cache = None

    backend = FakeBackend()
    monkeypatch.setenv(staged._PF_CACHE_DIR_ENV, str(tmp_path / "pf-cache"))
    monkeypatch.setattr(
        staged.redispatch.base.teacher,
        "_require_worker_context",
        lambda: {"backend": backend},
    )

    staged._attach_persistent_pf_cache()

    assert backend._persistent_exact_cache is None

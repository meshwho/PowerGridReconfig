from __future__ import annotations

from scripts.pipelines import run_teacher_redispatch as entrypoint


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

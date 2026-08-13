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

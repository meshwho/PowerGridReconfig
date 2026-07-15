from __future__ import annotations

from pathlib import Path

import pytest

from grid_topology_ai.self_play.run_state import (
    RunState,
    resolve_run_state,
)


def _complete_iteration(run_dir: Path, iteration: int) -> None:
    iteration_dir = run_dir / f"iter_{iteration:03d}"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_new_run_starts_from_first_iteration(tmp_path: Path) -> None:
    state = resolve_run_state(
        run_dir=tmp_path,
        resume=False,
    )

    assert state == RunState(
        completed_iterations=(),
        incomplete_directories=(),
        start_iteration=1,
    )


def test_resume_without_iterations_starts_from_first(
    tmp_path: Path,
) -> None:
    state = resolve_run_state(
        run_dir=tmp_path,
        resume=True,
    )

    assert state.start_iteration == 1
    assert state.completed_iterations == ()
    assert state.incomplete_directories == ()


def test_resume_continues_after_completed_iterations(
    tmp_path: Path,
) -> None:
    _complete_iteration(tmp_path, 1)
    _complete_iteration(tmp_path, 2)

    state = resolve_run_state(
        run_dir=tmp_path,
        resume=True,
    )

    assert state.completed_iterations == (1, 2)
    assert state.incomplete_directories == ()
    assert state.start_iteration == 3


def test_resume_rejects_incomplete_iteration(
    tmp_path: Path,
) -> None:
    (tmp_path / "iter_001").mkdir()

    with pytest.raises(RuntimeError, match="incomplete iteration"):
        resolve_run_state(
            run_dir=tmp_path,
            resume=True,
        )


def test_regular_run_rejects_completed_artifacts(
    tmp_path: Path,
) -> None:
    _complete_iteration(tmp_path, 1)

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        resolve_run_state(
            run_dir=tmp_path,
            resume=False,
        )


def test_regular_run_rejects_replay_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "replay_buffer" / "buffer_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        resolve_run_state(
            run_dir=tmp_path,
            resume=False,
        )


def test_regular_run_rejects_learning_curve(
    tmp_path: Path,
) -> None:
    (tmp_path / "learning_curve.csv").write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        resolve_run_state(
            run_dir=tmp_path,
            resume=False,
        )


def test_resume_rejects_iteration_gap(tmp_path: Path) -> None:
    _complete_iteration(tmp_path, 1)
    _complete_iteration(tmp_path, 3)

    with pytest.raises(RuntimeError, match="not contiguous"):
        resolve_run_state(
            run_dir=tmp_path,
            resume=True,
        )


def test_ignores_non_numeric_iteration_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "iter_abc").mkdir()
    (tmp_path / "iter_backup").mkdir()

    state = resolve_run_state(
        run_dir=tmp_path,
        resume=False,
    )

    assert state == RunState(
        completed_iterations=(),
        incomplete_directories=(),
        start_iteration=1,
    )

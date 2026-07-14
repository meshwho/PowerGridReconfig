from pathlib import Path

import pytest

from scripts.self_play.loop import determine_start_iteration


def complete_iteration(root: Path, iteration: int) -> None:
    iteration_dir = root / f"iter_{iteration:03d}"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_new_run_starts_from_first_iteration(tmp_path: Path) -> None:
    start, completed = determine_start_iteration(
        checkpoint_dir=tmp_path,
        n_iterations=3,
        resume=False,
    )

    assert start == 1
    assert completed == []


def test_resume_continues_after_completed_iterations(
    tmp_path: Path,
) -> None:
    complete_iteration(tmp_path, 1)
    complete_iteration(tmp_path, 2)

    start, completed = determine_start_iteration(
        checkpoint_dir=tmp_path,
        n_iterations=3,
        resume=True,
    )

    assert start == 3
    assert completed == [1, 2]


def test_resume_rejects_incomplete_iteration(
    tmp_path: Path,
) -> None:
    (tmp_path / "iter_001").mkdir()

    with pytest.raises(RuntimeError, match="incomplete iteration"):
        determine_start_iteration(
            checkpoint_dir=tmp_path,
            n_iterations=3,
            resume=True,
        )


def test_regular_run_rejects_existing_artifacts(
    tmp_path: Path,
) -> None:
    complete_iteration(tmp_path, 1)

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        determine_start_iteration(
            checkpoint_dir=tmp_path,
            n_iterations=3,
            resume=False,
        )


def test_resume_rejects_iteration_gap(tmp_path: Path) -> None:
    complete_iteration(tmp_path, 1)
    complete_iteration(tmp_path, 3)

    with pytest.raises(RuntimeError, match="not contiguous"):
        determine_start_iteration(
            checkpoint_dir=tmp_path,
            n_iterations=4,
            resume=True,
        )
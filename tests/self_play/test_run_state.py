from __future__ import annotations

from pathlib import Path

import pytest

from grid_topology_ai.self_play.artifacts import load_json, save_json
from grid_topology_ai.self_play import run_state as run_state_module
from grid_topology_ai.self_play.completion import write_iteration_completion_marker
from grid_topology_ai.self_play.run_state import (
    RunState,
    resolve_run_state,
)


def _complete_iteration(run_dir: Path, iteration: int) -> None:
    iteration_dir = run_dir / f"iter_{iteration:03d}"
    iteration_dir.mkdir(parents=True)

    metadata_path = iteration_dir / "metadata.json"
    candidate_checkpoint = iteration_dir / "candidate_checkpoint.pt"
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    best_metrics = run_dir / "checkpoints" / "best_metrics.json"
    pool_metadata = run_dir / "inputs" / "pool_metadata.json"
    replay_iteration = run_dir / "replay_buffer" / f"buffer_iter_{iteration:03d}.jsonl.gz"
    replay_manifest = run_dir / "replay_buffer" / "buffer_manifest.json"
    learning_curve = run_dir / "learning_curve.csv"

    save_json({"iteration": iteration, "accepted": True}, metadata_path)
    candidate_checkpoint.write_bytes(f"candidate-{iteration}".encode())
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint.write_bytes(b"best")
    save_json({"score": 1.0}, best_metrics)
    pool_metadata.parent.mkdir(parents=True, exist_ok=True)
    save_json({"last_updated_iteration": iteration}, pool_metadata)
    replay_iteration.parent.mkdir(parents=True, exist_ok=True)
    replay_iteration.write_bytes(f"replay-{iteration}".encode())

    files = []
    if replay_manifest.exists():
        files = list(load_json(replay_manifest).get("files", []))
    files.append({"iteration": iteration, "path": replay_iteration.name})
    save_json({"files": files}, replay_manifest)

    rows = ["iteration,accepted,status"]
    if learning_curve.exists():
        rows = learning_curve.read_text(encoding="utf-8").strip().splitlines()
    rows.append(f"{iteration},True,ACCEPTED")
    learning_curve.write_text("\n".join(rows) + "\n", encoding="utf-8")

    write_iteration_completion_marker(
        path=iteration_dir / "iteration_complete.json",
        iteration=iteration,
        accepted=True,
        status="ACCEPTED",
        metadata_path=metadata_path,
        candidate_checkpoint=candidate_checkpoint,
        best_checkpoint_after=best_checkpoint,
        best_metrics_path=best_metrics,
        pool_metadata_path=pool_metadata,
        replay_manifest_path=replay_manifest,
        replay_iteration_path=replay_iteration,
        learning_curve_path=learning_curve,
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

    with pytest.raises(RuntimeError, match="iteration_complete.json"):
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


def test_metadata_alone_is_not_complete(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter_001"
    iteration_dir.mkdir()
    save_json({"iteration": 1, "accepted": True}, iteration_dir / "metadata.json")

    with pytest.raises(RuntimeError) as exc_info:
        resolve_run_state(
            run_dir=tmp_path,
            resume=True,
        )

    message = str(exc_info.value)
    assert "iteration_complete.json" in message
    assert "metadata.json is no longer proof" in message


def test_resume_rejects_corrupt_completion_marker(tmp_path: Path) -> None:
    _complete_iteration(tmp_path, 1)
    save_json(
        {"iteration": 1, "accepted": True, "tampered": True},
        tmp_path / "iter_001" / "metadata.json",
    )

    with pytest.raises(RuntimeError, match="Invalid iteration completion marker"):
        resolve_run_state(
            run_dir=tmp_path,
            resume=True,
        )


def test_expected_marker_validation_error_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iteration_dir = tmp_path / "iter_001"
    iteration_dir.mkdir()
    (iteration_dir / "iteration_complete.json").write_text("{}", encoding="utf-8")

    def fail(**kwargs: object) -> None:
        raise ValueError("corrupt marker")

    monkeypatch.setattr(run_state_module, "validate_iteration_completion", fail)

    with pytest.raises(RuntimeError, match="Invalid iteration completion marker") as exc_info:
        resolve_run_state(run_dir=tmp_path, resume=True)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_unexpected_marker_validation_bug_is_not_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iteration_dir = tmp_path / "iter_001"
    iteration_dir.mkdir()
    (iteration_dir / "iteration_complete.json").write_text("{}", encoding="utf-8")

    def fail(**kwargs: object) -> None:
        raise RuntimeError("validator bug")

    monkeypatch.setattr(run_state_module, "validate_iteration_completion", fail)

    with pytest.raises(RuntimeError, match="^validator bug$"):
        resolve_run_state(run_dir=tmp_path, resume=True)

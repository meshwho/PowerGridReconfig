from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grid_topology_ai.self_play.completion import (
    COMPLETION_MARKER_FILENAME,
    validate_iteration_completion,
)


@dataclass(frozen=True, slots=True)
class RunState:
    completed_iterations: tuple[int, ...]
    incomplete_directories: tuple[Path, ...]
    start_iteration: int


def _scan_iteration_directories(
    run_dir: Path,
) -> tuple[tuple[int, ...], tuple[Path, ...]]:
    completed: set[int] = set()
    incomplete: list[Path] = []

    for iteration_dir in sorted(run_dir.glob("iter_*")):
        if not iteration_dir.is_dir():
            continue

        suffix = iteration_dir.name.removeprefix("iter_")

        try:
            iteration = int(suffix)
        except ValueError:
            continue

        marker_path = iteration_dir / COMPLETION_MARKER_FILENAME
        if not marker_path.exists():
            incomplete.append(iteration_dir)
            continue

        try:
            validate_iteration_completion(
                iteration_dir=iteration_dir,
                expected_iteration=iteration,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Invalid iteration completion marker: {marker_path}"
            ) from exc

        completed.add(iteration)

    return tuple(sorted(completed)), tuple(sorted(incomplete))


def resolve_run_state(
    *,
    run_dir: Path,
    resume: bool,
) -> RunState:
    completed, incomplete = _scan_iteration_directories(run_dir)
    replay_manifest = run_dir / "replay_buffer" / "buffer_manifest.json"
    learning_curve = run_dir / "learning_curve.csv"

    has_existing_run = bool(
        completed
        or incomplete
        or replay_manifest.exists()
        or learning_curve.exists()
    )

    if not resume:
        if has_existing_run:
            raise RuntimeError(
                "Existing self-play run artifacts were found in "
                f"{run_dir}. Refusing to overwrite them. "
                "Use --resume to continue the run, or remove the "
                "existing runtime artifacts before starting again."
            )

        return RunState((), (), 1)

    if incomplete:
        formatted = "\n".join(f"  - {path}" for path in incomplete)

        raise RuntimeError(
            "Cannot safely resume because incomplete iteration "
            "directories were found:\n"
            f"{formatted}\n"
            f"Missing required completion marker: {COMPLETION_MARKER_FILENAME}. "
            "metadata.json is no longer proof that an iteration fully completed. "
            "Do not blindly delete these directories because replay, pool, or best "
            "checkpoint artifacts may already have been updated. Manual inspection "
            "is required before running again with --resume."
        )

    if not completed:
        return RunState((), (), 1)

    expected = tuple(range(1, max(completed) + 1))

    if completed != expected:
        raise RuntimeError(
            "Completed iterations are not contiguous. "
            f"Found={list(completed)}, expected={list(expected)}. "
            "Manual inspection is required before resuming."
        )

    return RunState(completed, (), max(completed) + 1)

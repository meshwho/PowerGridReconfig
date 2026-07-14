from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

        if (iteration_dir / "metadata.json").is_file():
            completed.add(iteration)
        else:
            incomplete.append(iteration_dir)

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
            "Remove or archive these incomplete directories before "
            "running again with --resume."
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

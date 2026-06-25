from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DIFFICULTIES: tuple[str, ...] = ("simple", "medium", "hard")
SPLITS: tuple[str, ...] = ("train", "val")
DIFFICULTY_COLUMN_CANDIDATES: tuple[str, ...] = (
    "difficulty_class",
    "difficulty",
    "class",
    "category",
)


@dataclass(frozen=True)
class TeacherProfile:
    """Search settings for one difficulty class."""

    depth: int
    beam_width: int
    candidate_pool: int
    top_k: int
    lodf_top_k: int
    max_steps: int
    max_teacher_steps: int
    batch_size: int
    auto_worker_max: int


DEFAULT_TEACHER_PROFILES: dict[str, TeacherProfile] = {
    "simple": TeacherProfile(
        depth=4,
        beam_width=10,
        candidate_pool=60,
        top_k=30,
        lodf_top_k=30,
        max_steps=5,
        max_teacher_steps=5,
        batch_size=3,
        auto_worker_max=8,
    ),
    "medium": TeacherProfile(
        depth=5,
        beam_width=20,
        candidate_pool=160,
        top_k=70,
        lodf_top_k=70,
        max_steps=5,
        max_teacher_steps=5,
        batch_size=2,
        auto_worker_max=7,
    ),
    "hard": TeacherProfile(
        depth=6,
        beam_width=30,
        candidate_pool=220,
        top_k=100,
        lodf_top_k=100,
        max_steps=6,
        max_teacher_steps=6,
        batch_size=2,
        auto_worker_max=6,
    ),
}


@dataclass(frozen=True)
class Paths:
    dataset_name: str
    run_name: str
    project_root: Path
    raw_dir: Path
    transitions_root: Path
    split_dir: Path
    output_root: Path
    logs_dir: Path
    state_path: Path

class PipelineError(RuntimeError):
    """Raised for an expected pipeline validation failure."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def discover_project_root() -> Path:
    """Find the repository root without hard-coding a user-specific path."""

    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parent]

    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if (parent / "grid_topology_ai").is_dir() and (parent / "scripts").is_dir():
                return parent

    return Path.cwd().resolve()


def resolve_from_root(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()

def validate_path_component(
    value: str,
    option_name: str,
) -> str:
    """
    Validate a directory name supplied through a CLI option.

    Dataset and run names must be names, not full paths.
    Full paths are passed separately through --raw-dir,
    --transitions-root and --output-root.
    """

    normalized = str(value).strip()

    if not normalized:
        raise PipelineError(
            f"{option_name} must not be empty."
        )

    if normalized in {".", ".."}:
        raise PipelineError(
            f"{option_name} must be a normal directory name."
        )

    if "/" in normalized or "\\" in normalized:
        raise PipelineError(
            f"{option_name} must be a directory name, not a path: "
            f"{normalized!r}"
        )

    return normalized

def detect_difficulty_column(frame: pd.DataFrame) -> str:
    for column in DIFFICULTY_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column

    raise PipelineError(
        "Difficulty column was not found. Expected one of: "
        + ", ".join(DIFFICULTY_COLUMN_CANDIDATES)
        + f". Available columns: {frame.columns.tolist()}"
    )


def normalize_difficulty(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def ensure_required_transition_columns(frame: pd.DataFrame, path: Path) -> str:
    if frame.empty:
        raise PipelineError(f"Transitions CSV is empty: {path}")

    if "scenario_id" not in frame.columns:
        raise PipelineError(f"Transitions CSV must contain scenario_id: {path}")

    difficulty_column = detect_difficulty_column(frame)
    normalized = normalize_difficulty(frame[difficulty_column])
    unknown = sorted(set(normalized.dropna()) - set(DIFFICULTIES))

    if unknown:
        raise PipelineError(
            f"Unsupported difficulty values in {path}: {unknown}. "
            f"Expected only {list(DIFFICULTIES)}."
        )

    scenario_class_counts = (
        frame.assign(_difficulty=normalized)
        .groupby("scenario_id", dropna=False)["_difficulty"]
        .nunique(dropna=False)
    )
    conflicting = scenario_class_counts[scenario_class_counts > 1]

    if not conflicting.empty:
        preview = conflicting.index.tolist()[:20]
        raise PipelineError(
            "The same scenario_id has more than one difficulty class. "
            f"Conflicting scenarios: {preview}"
        )

    return difficulty_column


def deterministic_scenario_subset(
    scenario_ids: Sequence[int],
    limit: int,
    seed: int,
) -> set[int]:
    unique_ids = np.array(sorted({int(value) for value in scenario_ids}), dtype=np.int64)

    if limit <= 0 or len(unique_ids) <= limit:
        return {int(value) for value in unique_ids.tolist()}

    rng = np.random.default_rng(seed)
    selected = rng.choice(unique_ids, size=int(limit), replace=False)
    return {int(value) for value in selected.tolist()}


def split_transitions(
    source_path: Path,
    split_name: str,
    split_dir: Path,
    classes: Sequence[str],
    limit_per_class: int,
    seed: int,
    force: bool,
) -> dict[str, dict[str, object]]:
    banner(f"Splitting {split_name} transitions by difficulty")

    if not source_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {source_path}")

    frame = pd.read_csv(source_path)
    difficulty_column = ensure_required_transition_columns(frame, source_path)
    normalized = normalize_difficulty(frame[difficulty_column])
    frame = frame.copy()
    frame[difficulty_column] = normalized

    split_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, object]] = {}

    print(f"Source:             {source_path}")
    print(f"Difficulty column:  {difficulty_column}")
    print(f"Rows:               {len(frame)}")
    print(f"Scenarios:          {frame['scenario_id'].nunique()}")
    print(f"Limit per class:    {limit_per_class or 'all'}")

    split_offset = 0 if split_name == "train" else 100_000

    for class_index, difficulty in enumerate(classes):
        class_frame = frame.loc[frame[difficulty_column] == difficulty].copy()

        if class_frame.empty:
            raise PipelineError(
                f"No {difficulty!r} rows found in {source_path}."
            )

        selected_ids = deterministic_scenario_subset(
            scenario_ids=class_frame["scenario_id"].astype(int).tolist(),
            limit=limit_per_class,
            seed=int(seed) + split_offset + class_index,
        )
        class_frame = class_frame.loc[
            class_frame["scenario_id"].astype(int).isin(selected_ids)
        ].copy()

        output_path = split_dir / f"transitions_{split_name}_{difficulty}.csv"

        if output_path.exists() and not force:
            existing = pd.read_csv(output_path)
            same_ids = set(existing["scenario_id"].astype(int)) == selected_ids
            same_rows = len(existing) == len(class_frame)

            if same_ids and same_rows:
                print(
                    f"{difficulty:8s}: existing split is valid - skipping "
                    f"({len(selected_ids)} scenarios)"
                )
            else:
                raise PipelineError(
                    f"Existing split does not match the requested configuration: {output_path}. "
                    "Use --force to rebuild it."
                )
        else:
            class_frame.to_csv(output_path, index=False)
            print(
                f"{difficulty:8s}: rows={len(class_frame):6d}, "
                f"scenarios={len(selected_ids):6d}, output={output_path}"
            )

        stats[difficulty] = {
            "source": str(source_path),
            "output": str(output_path),
            "rows": int(len(class_frame)),
            "scenarios": int(len(selected_ids)),
            "difficulty_column": difficulty_column,
        }

    return stats


def validate_teacher_examples(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "examples.csv is missing"

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return False, f"cannot read examples.csv: {exc}"

    if frame.empty:
        return False, "examples.csv is empty"

    required = {
        "scenario_id",
        "state_path",
        "mcts_policy_json",
        "outcome_value_target",
    }
    missing = sorted(required - set(frame.columns))

    if missing:
        return False, f"missing required columns: {missing}"

    return True, "ok"


def archive_incomplete_output(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = output_dir.with_name(f"{output_dir.name}.incomplete_{timestamp}")
    output_dir.rename(archived)
    return archived


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        import psutil  # type: ignore

        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=5)
        for item in alive:
            item.kill()
    except Exception:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def run_live_command(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    printable = subprocess.list2cmdline([str(item) for item in command])
    print(f"Command: {printable}")
    print(f"Log:     {log_path}")

    if dry_run:
        print("DRY RUN - command was not executed.")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    popen_kwargs: dict[str, object] = {
        "cwd": str(cwd),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen([str(item) for item in command], **popen_kwargs)

        try:
            if process.stdout is None:
                raise PipelineError("Could not capture teacher process output.")

            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()
        except KeyboardInterrupt:
            print("\nInterrupted. Terminating teacher process tree...")
            terminate_process_tree(process)
            raise

    if return_code != 0:
        raise PipelineError(
            f"Teacher command failed with exit code {return_code}. See log: {log_path}"
        )


def build_teacher_command(
    python_executable: str,
    teacher_module: str,
    raw_dir: Path,
    transitions_path: Path,
    output_dir: Path,
    profile: TeacherProfile,
    num_workers: str,
    use_lodf: bool,
    lodf_min_candidate_count: int,
    max_worker_memory_mb: float,
    auto_worker_memory_mb: float,
    auto_worker_memory_reserve_mb: float,
    value_reward_scale: str,
    quiet_success: bool,
) -> list[str]:
    command = [
        python_executable,
        "-u",
        "-m",
        teacher_module,
        str(raw_dir),
        "--transitions",
        str(transitions_path),
        "--output-dir",
        str(output_dir),
        "--depth",
        str(profile.depth),
        "--beam-width",
        str(profile.beam_width),
        "--candidate-pool",
        str(profile.candidate_pool),
        "--top-k",
        str(profile.top_k),
        "--max-steps",
        str(profile.max_steps),
        "--max-teacher-steps",
        str(profile.max_teacher_steps),
        "--pf-alg",
        "3",
        "--pf-max-iter",
        "30",
        "--num-workers",
        str(num_workers),
        "--batch-size",
        str(profile.batch_size),
        "--clear-caches-every",
        str(profile.batch_size),
        "--max-worker-memory-mb",
        str(max_worker_memory_mb),
        "--max-tasks-per-child",
        "0",
        "--auto-worker-memory-mb",
        str(auto_worker_memory_mb),
        "--auto-worker-memory-reserve-mb",
        str(auto_worker_memory_reserve_mb),
        "--auto-worker-max",
        str(profile.auto_worker_max),
        "--value-target-mode",
        "tanh_step_reward_discounted_average",
        "--value-reward-scale",
        str(value_reward_scale),
        "--add-handoff-example",
    ]

    if use_lodf:
        command.extend(
            [
                "--use-lodf-screening",
                "--lodf-screen-top-k",
                str(profile.lodf_top_k),
                "--lodf-min-candidate-count",
                str(lodf_min_candidate_count),
            ]
        )

    if quiet_success:
        command.append("--quiet-success")

    return command


def run_teacher(
    paths: Paths,
    split_name: str,
    difficulty: str,
    profile: TeacherProfile,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, object]:
    transitions_path = paths.split_dir / f"transitions_{split_name}_{difficulty}.csv"
    output_dir = paths.output_root / split_name / difficulty
    examples_path = output_dir / "examples.csv"
    log_path = paths.logs_dir / f"teacher_{split_name}_{difficulty}.log"

    if not transitions_path.exists():
        raise FileNotFoundError(
            f"Split transitions are missing: {transitions_path}. Run stage 'split' first."
        )

    valid, reason = validate_teacher_examples(examples_path)

    if valid and not args.force:
        frame = pd.read_csv(examples_path)
        print(
            f"{split_name}/{difficulty}: valid examples.csv already exists - "
            f"skipping ({len(frame)} examples, "
            f"{frame['scenario_id'].nunique()} scenarios)."
        )
        return {
            "status": "skipped_existing",
            "examples": int(len(frame)),
            "scenarios": int(frame["scenario_id"].nunique()),
            "output_dir": str(output_dir),
            "examples_path": str(examples_path),
            "log_path": str(log_path),
        }

    if output_dir.exists():
        if args.force:
            shutil.rmtree(output_dir)
        elif args.keep_incomplete:
            raise PipelineError(
                f"Teacher output exists but is not complete ({reason}): {output_dir}. "
                "Remove it, use --force, or omit --keep-incomplete to archive it automatically."
            )
        else:
            archived = archive_incomplete_output(output_dir)
            print(f"Archived incomplete output: {archived}")

    output_dir.mkdir(parents=True, exist_ok=True)

    banner(f"Running teacher: {split_name}/{difficulty}")
    print(f"Transitions:          {transitions_path}")
    print(f"Output:               {output_dir}")
    print(f"Depth:                {profile.depth}")
    print(f"Beam width:           {profile.beam_width}")
    print(f"Candidate pool:       {profile.candidate_pool}")
    print(f"Top-K:                {profile.top_k}")
    print(f"LODF enabled:         {not args.disable_lodf}")
    print(f"LODF top-K:           {profile.lodf_top_k}")
    print(f"Batch size:           {profile.batch_size}")
    print(f"Worker mode:          {args.num_workers}")
    print(f"Auto worker maximum:  {profile.auto_worker_max}")

    command = build_teacher_command(
        python_executable=args.python_executable,
        teacher_module=args.teacher_module,
        raw_dir=paths.raw_dir,
        transitions_path=transitions_path,
        output_dir=output_dir,
        profile=profile,
        num_workers=args.num_workers,
        use_lodf=not args.disable_lodf,
        lodf_min_candidate_count=args.lodf_min_candidate_count,
        max_worker_memory_mb=args.max_worker_memory_mb,
        auto_worker_memory_mb=args.auto_worker_memory_mb,
        auto_worker_memory_reserve_mb=args.auto_worker_memory_reserve_mb,
        value_reward_scale=args.value_reward_scale,
        quiet_success=args.quiet_success,
    )

    run_live_command(
        command=command,
        cwd=paths.project_root,
        env=env,
        log_path=log_path,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return {
            "status": "dry_run",
            "output_dir": str(output_dir),
            "examples_path": str(examples_path),
            "log_path": str(log_path),
        }

    valid, reason = validate_teacher_examples(examples_path)
    if not valid:
        raise PipelineError(
            f"Teacher finished, but output validation failed: {reason}. Path: {examples_path}"
        )

    frame = pd.read_csv(examples_path)
    return {
        "status": "completed",
        "examples": int(len(frame)),
        "scenarios": int(frame["scenario_id"].nunique()),
        "output_dir": str(output_dir),
        "examples_path": str(examples_path),
        "log_path": str(log_path),
    }


def merge_examples(
    paths: Paths,
    split_name: str,
    classes: Sequence[str],
    force: bool,
) -> dict[str, object]:
    banner(f"Merging teacher examples: {split_name}")

    output_path = paths.output_root / f"examples_{split_name}.csv"

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        print(
            f"Merged file already exists - rebuilding it to reflect the current class outputs: "
            f"{output_path}"
        )

    parts: list[pd.DataFrame] = []
    source_stats: dict[str, dict[str, int]] = {}

    for difficulty in classes:
        source_path = paths.output_root / split_name / difficulty / "examples.csv"
        valid, reason = validate_teacher_examples(source_path)

        if not valid:
            raise PipelineError(
                f"Cannot merge {split_name}/{difficulty}: {reason}. Path: {source_path}"
            )

        frame = pd.read_csv(source_path)
        frame = frame.copy()

        if "difficulty_class" in frame.columns:
            existing_values = set(normalize_difficulty(frame["difficulty_class"]))
            if existing_values != {difficulty}:
                raise PipelineError(
                    f"Unexpected difficulty_class values in {source_path}: {existing_values}"
                )
        else:
            frame["difficulty_class"] = difficulty

        frame["teacher_split"] = split_name
        frame["source_examples_csv"] = str(source_path)

        parts.append(frame)
        source_stats[difficulty] = {
            "examples": int(len(frame)),
            "scenarios": int(frame["scenario_id"].nunique()),
        }

        print(
            f"{difficulty:8s}: examples={len(frame):7d}, "
            f"scenarios={frame['scenario_id'].nunique():6d}"
        )

    merged = pd.concat(parts, ignore_index=True, sort=False)

    if "state_id" in merged.columns:
        duplicated_state_ids = int(merged["state_id"].duplicated().sum())
        if duplicated_state_ids:
            raise PipelineError(
                f"Duplicate state_id values found while merging {split_name}: "
                f"{duplicated_state_ids}"
            )

    duplicate_step_rows = int(
        merged.duplicated(subset=["scenario_id", "step"], keep=False).sum()
    ) if "step" in merged.columns else 0

    if duplicate_step_rows:
        raise PipelineError(
            f"Duplicate (scenario_id, step) rows found while merging {split_name}: "
            f"{duplicate_step_rows}"
        )

    merged.to_csv(output_path, index=False)

    print(f"Merged examples:  {len(merged)}")
    print(f"Merged scenarios: {merged['scenario_id'].nunique()}")
    print(f"Saved:            {output_path}")

    return {
        "output": str(output_path),
        "examples": int(len(merged)),
        "scenarios": int(merged["scenario_id"].nunique()),
        "sources": source_stats,
    }


def resolve_state_path(project_root: Path, row: pd.Series) -> Path | None:
    raw_value = str(row.get("state_path", "")).strip()
    if not raw_value:
        return None

    path = Path(raw_value)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(project_root / path)

        source_csv = str(row.get("source_examples_csv", "")).strip()
        if source_csv:
            source_path = Path(source_csv)
            if not source_path.is_absolute():
                source_path = project_root / source_path
            candidates.append(source_path.parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def audit_merged_examples(
    paths: Paths,
    splits: Sequence[str],
    check_state_files: bool,
) -> dict[str, object]:
    banner("Auditing merged teacher datasets")

    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, object] = {}
    required = {
        "scenario_id",
        "state_path",
        "mcts_policy_json",
        "outcome_value_target",
        "difficulty_class",
        "teacher_split",
    }

    for split_name in splits:
        path = paths.output_root / f"examples_{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Merged examples file not found: {path}")

        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise PipelineError(f"{split_name}: missing columns: {missing}")

        targets = pd.to_numeric(frame["outcome_value_target"], errors="coerce")
        if targets.isna().any():
            raise PipelineError(
                f"{split_name}: {int(targets.isna().sum())} invalid outcome_value_target values."
            )

        outside = int((targets.abs() > 1.0).sum())
        if outside:
            raise PipelineError(
                f"{split_name}: {outside} outcome_value_target values are outside [-1, 1]."
            )

        invalid_policies = 0
        for value in frame["mcts_policy_json"]:
            try:
                policy = json.loads(value)
                if not isinstance(policy, dict) or not policy:
                    invalid_policies += 1
            except Exception:
                invalid_policies += 1

        if invalid_policies:
            raise PipelineError(
                f"{split_name}: {invalid_policies} invalid mcts_policy_json values."
            )

        missing_states = 0
        if check_state_files:
            for _, row in frame.iterrows():
                if resolve_state_path(paths.project_root, row) is None:
                    missing_states += 1

            if missing_states:
                raise PipelineError(
                    f"{split_name}: {missing_states} referenced state files were not found."
                )

        class_counts = (
            normalize_difficulty(frame["difficulty_class"])
            .value_counts()
            .reindex(DIFFICULTIES, fill_value=0)
            .to_dict()
        )

        frames[split_name] = frame
        audit[split_name] = {
            "path": str(path),
            "examples": int(len(frame)),
            "scenarios": int(frame["scenario_id"].nunique()),
            "class_example_counts": {key: int(value) for key, value in class_counts.items()},
            "missing_state_files": int(missing_states),
            "outcome_value_min": float(targets.min()),
            "outcome_value_max": float(targets.max()),
        }

        print(f"{split_name.capitalize()} examples:   {len(frame)}")
        print(f"{split_name.capitalize()} scenarios:  {frame['scenario_id'].nunique()}")
        print(f"{split_name.capitalize()} classes:    {class_counts}")
        print(f"{split_name.capitalize()} missing states: {missing_states}")

    if "train" in frames and "val" in frames:
        train_ids = set(frames["train"]["scenario_id"].astype(int))
        val_ids = set(frames["val"]["scenario_id"].astype(int))
        overlap = train_ids & val_ids

        print(f"Train/val scenario overlap: {len(overlap)}")

        if overlap:
            preview = sorted(overlap)[:20]
            raise PipelineError(
                f"Train/validation leakage: {len(overlap)} overlapping scenario IDs. "
                f"Examples: {preview}"
            )

        audit["train_val_overlap"] = 0

    print("AUDIT PASSED")
    return audit


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "created_at": utc_now_iso(),
            "completed_teachers": {},
        }

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "created_at": utc_now_iso(),
            "completed_teachers": {},
        }


def save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def build_runtime_environment(args: argparse.Namespace, paths: Paths) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    if args.temp_dir:
        temp_dir = resolve_from_root(paths.project_root, args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        env["TEMP"] = str(temp_dir)
        env["TMP"] = str(temp_dir)

    if args.julia_depot:
        julia_depot = resolve_from_root(paths.project_root, args.julia_depot)
        julia_depot.mkdir(parents=True, exist_ok=True)
        env["JULIA_DEPOT_PATH"] = str(julia_depot)

    if args.julia_bin:
        julia_bin = resolve_from_root(paths.project_root, args.julia_bin)
        if not julia_bin.exists():
            raise FileNotFoundError(f"Julia bin directory not found: {julia_bin}")
        env["PATH"] = str(julia_bin) + os.pathsep + env.get("PATH", "")

    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split GridFM transitions by difficulty, run one teacher per class, "
            "merge the generated examples, and audit the final datasets."
        )
    )

    parser.add_argument(
        "--stage",
        choices=["all", "split", "teacher", "merge", "audit"],
        default="all",
        help="Pipeline stage to execute. Default: all.",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "smoke"],
        default="full",
        help="Smoke uses a small deterministic sample per class.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Maximum scenarios per class and split. Overrides --profile.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLITS),
        default=list(SPLITS),
        help="Dataset splits to process. Default: train val.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=list(DIFFICULTIES),
        default=list(DIFFICULTIES),
        help="Difficulty classes to process. Default: simple medium hard.",
    )

    parser.add_argument(
        "--project-root",
        default=str(discover_project_root()),
        help=(
            "Repository root. By default it is detected automatically."
        ),
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help=(
            "Dataset directory name. For example: "
            "case118_bootstrap_v1. "
            "It is used to build automatic raw and transitions paths."
        ),
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional teacher run directory name. "
            "If omitted, it is generated from dataset name and profile."
        ),
    )

    parser.add_argument(
        "--raw-dir",
        default=None,
        help=(
            "Explicit GridFM raw directory. "
            "Default: data/gridfm_generated/<dataset-name>/raw."
        ),
    )

    parser.add_argument(
        "--transitions-root",
        default=None,
        help=(
            "Explicit transitions directory. "
            "Default: data/gridfm_transitions/<dataset-name>."
        ),
)
    parser.add_argument(
        "--train-file",
        default="transitions_train.csv",
    )
    parser.add_argument(
        "--val-file",
        default="transitions_val.csv",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Explicit pipeline output directory. "
            "For smoke runs the default is "
            "data/_scratch/<run-name>. "
            "For full runs the default is "
            "data/self_play/<run-name>."
        ),
    )

    parser.add_argument(
        "--teacher-module",
        default="scripts.self_play.generate_impact_teacher_parallel_fast",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python used to start teacher workers. Default: current interpreter.",
    )
    parser.add_argument(
        "--num-workers",
        default="auto",
        help="Teacher --num-workers value. Default: auto.",
    )
    parser.add_argument("--lodf-min-candidate-count", type=int, default=8)
    parser.add_argument("--disable-lodf", action="store_true")
    parser.add_argument("--max-worker-memory-mb", type=float, default=1000.0)
    parser.add_argument("--auto-worker-memory-mb", type=float, default=1200.0)
    parser.add_argument(
        "--auto-worker-memory-reserve-mb",
        type=float,
        default=2048.0,
    )
    parser.add_argument("--value-reward-scale", default="7000")
    parser.add_argument(
        "--quiet-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional TEMP/TMP directory. Relative paths use project root.",
    )
    parser.add_argument(
        "--julia-depot",
        default=None,
        help="Optional JULIA_DEPOT_PATH. Relative paths use project root.",
    )
    parser.add_argument(
        "--julia-bin",
        default=None,
        help="Optional Julia bin directory added to PATH.",
    )

    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Do not archive incomplete teacher output automatically.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-state-audit",
        action="store_true",
        help="Skip checking every referenced .npz state file.",
    )

    args = parser.parse_args()

    if args.limit_per_class is None:
        args.limit_per_class = 20 if args.profile == "smoke" else 0

    if args.limit_per_class < 0:
        parser.error("--limit-per-class must be >= 0")

    return args


def main() -> None:
    args = parse_args()

    project_root = Path(
        args.project_root
    ).expanduser().resolve()

    dataset_name = validate_path_component(
        value=args.dataset_name,
        option_name="--dataset-name",
    )

    default_run_name = (
        f"{dataset_name}_teacher_smoke_v1"
        if args.profile == "smoke"
        else f"{dataset_name}_teacher_v1"
    )

    run_name = validate_path_component(
        value=(
            args.run_name
            if args.run_name is not None
            else default_run_name
        ),
        option_name="--run-name",
    )

    raw_dir_value: str | Path

    if args.raw_dir is not None:
        raw_dir_value = args.raw_dir
    else:
        raw_dir_value = (
            Path("data")
            / "gridfm_generated"
            / dataset_name
            / "raw"
        )

    transitions_root_value: str | Path

    if args.transitions_root is not None:
        transitions_root_value = args.transitions_root
    else:
        transitions_root_value = (
            Path("data")
            / "gridfm_transitions"
            / dataset_name
        )

    if args.output_root is not None:
        output_root_value: str | Path = args.output_root
    else:
        output_base = (
            Path("data") / "_scratch"
            if args.profile == "smoke"
            else Path("data") / "self_play"
        )

        output_root_value = output_base / run_name

    output_root = resolve_from_root(
        project_root=project_root,
        value=output_root_value,
    )

    paths = Paths(
        dataset_name=dataset_name,
        run_name=run_name,
        project_root=project_root,
        raw_dir=resolve_from_root(
            project_root=project_root,
            value=raw_dir_value,
        ),
        transitions_root=resolve_from_root(
            project_root=project_root,
            value=transitions_root_value,
        ),
        split_dir=output_root / "split_transitions",
        output_root=output_root,
        logs_dir=output_root / "logs",
        state_path=output_root / "pipeline_state.json",
    )
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    env = build_runtime_environment(args, paths)
    state = load_state(paths.state_path)
    state.update(
        {
            "updated_at": utc_now_iso(),
            "status": "running",
            "stage": args.stage,
            "profile": args.profile,
            "dataset_name": paths.dataset_name,
            "run_name": paths.run_name,
            "project_root": str(paths.project_root),
            "raw_dir": str(paths.raw_dir),
            "transitions_root": str(paths.transitions_root),
            "split_dir": str(paths.split_dir),
            "output_root": str(paths.output_root),
            "splits": list(args.splits),
            "classes": list(args.classes),
            "limit_per_class": int(args.limit_per_class),
            "seed": int(args.seed),
            "python_executable": str(args.python_executable),
            "teacher_module": str(args.teacher_module),
            "lodf_enabled": not bool(args.disable_lodf),
            "teacher_profiles": {
                name: asdict(profile)
                for name, profile in DEFAULT_TEACHER_PROFILES.items()
            },
        }
    )
    save_state(paths.state_path, state)

    banner("Teacher pipeline by difficulty")
    print(f"Stage:             {args.stage}")
    print(f"Profile:           {args.profile}")
    print(f"Dataset name:      {paths.dataset_name}")
    print(f"Run name:          {paths.run_name}")
    print(f"Project root:      {paths.project_root}")
    print(f"Python:            {args.python_executable}")
    print(f"Raw dir:           {paths.raw_dir}")
    print(f"Transitions root:  {paths.transitions_root}")
    print(f"Split dir:         {paths.split_dir}")
    print(f"Output root:       {paths.output_root}")
    print(f"Splits:            {args.splits}")
    print(f"Classes:           {args.classes}")
    print(f"Limit per class:   {args.limit_per_class or 'all'}")
    print(f"LODF enabled:      {not args.disable_lodf}")
    print(f"Dry run:           {args.dry_run}")

    if not paths.raw_dir.exists() and args.stage in {"all", "teacher"}:
        raise FileNotFoundError(f"Raw directory not found: {paths.raw_dir}")

    try:
        if args.stage in {"all", "split"}:
            split_stats: dict[str, object] = {}

            for split_name in args.splits:
                source_name = args.train_file if split_name == "train" else args.val_file
                source_path = paths.transitions_root / source_name
                split_stats[split_name] = split_transitions(
                    source_path=source_path,
                    split_name=split_name,
                    split_dir=paths.split_dir,
                    classes=args.classes,
                    limit_per_class=int(args.limit_per_class),
                    seed=int(args.seed),
                    force=bool(args.force),
                )

            state["split_stats"] = split_stats
            state["updated_at"] = utc_now_iso()
            save_state(paths.state_path, state)

        if args.stage in {"all", "teacher"}:
            completed_teachers = dict(state.get("completed_teachers", {}))

            for split_name in args.splits:
                for difficulty in args.classes:
                    result = run_teacher(
                        paths=paths,
                        split_name=split_name,
                        difficulty=difficulty,
                        profile=DEFAULT_TEACHER_PROFILES[difficulty],
                        args=args,
                        env=env,
                    )
                    completed_teachers[f"{split_name}/{difficulty}"] = result
                    state["completed_teachers"] = completed_teachers
                    state["updated_at"] = utc_now_iso()
                    save_state(paths.state_path, state)

        if args.stage in {"all", "merge"}:
            merge_stats: dict[str, object] = {}

            for split_name in args.splits:
                merge_stats[split_name] = merge_examples(
                    paths=paths,
                    split_name=split_name,
                    classes=args.classes,
                    force=bool(args.force),
                )

            state["merge_stats"] = merge_stats
            state["updated_at"] = utc_now_iso()
            save_state(paths.state_path, state)

        if args.stage in {"all", "audit"}:
            audit = audit_merged_examples(
                paths=paths,
                splits=args.splits,
                check_state_files=not bool(args.skip_state_audit),
            )
            state["audit"] = audit
            state["updated_at"] = utc_now_iso()
            save_state(paths.state_path, state)

        state["status"] = "completed"
        state["updated_at"] = utc_now_iso()
        save_state(paths.state_path, state)

        banner("Teacher pipeline completed")
        for split_name in args.splits:
            print(f"{split_name}: {paths.output_root / f'examples_{split_name}.csv'}")
        print(f"State: {paths.state_path}")

    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["updated_at"] = utc_now_iso()
        save_state(paths.state_path, state)
        raise SystemExit(130)
    except Exception as exc:
        state["status"] = "failed"
        state["updated_at"] = utc_now_iso()
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        save_state(paths.state_path, state)
        raise


if __name__ == "__main__":
    main()

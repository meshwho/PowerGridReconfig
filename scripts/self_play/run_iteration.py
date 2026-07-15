from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import os
import sys
import traceback
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.config import (
    EvaluationConfig,
    GenerationConfig,
    TrainingConfig,
)
from grid_topology_ai.evaluation.checkpoint import (
    EvaluationRequest,
    evaluate_checkpoint,
)
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
)
from grid_topology_ai.self_play.generation import (
    GenerationRequest,
    generate_self_play_examples,
)
from grid_topology_ai.training.graph_policy_value import (
    TrainingRequest,
    train_graph_policy_value_model,
)
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows


def discover_project_root(start: str | Path | None = None) -> Path:
    """
    Find repository root by walking upward until project markers are found.
    """

    current = Path.cwd() if start is None else Path(start).resolve()

    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (
            (candidate / "grid_topology_ai").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not discover project root. Run from inside PowerGridReconfig."
    )


class _TeeTextIO(io.TextIOBase):
    def __init__(self, console_stream: io.TextIOBase, log_file: io.TextIOBase) -> None:
        self._console_stream = console_stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._console_stream.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._console_stream.flush()
        self._log_file.flush()


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """
    Temporarily switch process cwd for a stage API call.

    The self-play loop executes stages sequentially, and this context manager
    changes the process-global cwd to preserve the previous child-process cwd
    behavior.
    """

    previous = Path.cwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _stage_output(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_tee = _TeeTextIO(sys.stdout, log_file)
        stderr_tee = _TeeTextIO(sys.stderr, log_file)

        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            try:
                yield
            except Exception:
                traceback.print_exc()
                raise
            finally:
                stdout_tee.flush()
                stderr_tee.flush()


def write_selected_transitions_csv(
    *,
    transitions_csv: str | Path,
    scenario_ids: list[int],
    output_path: str | Path,
) -> Path:
    """
    Create a temporary transitions CSV containing only sampled scenario IDs.

    This preserves the selected-transitions artifact and validates sampled IDs.
    """

    transitions_csv = Path(transitions_csv)
    output_path = Path(output_path)

    if not transitions_csv.exists():
        raise FileNotFoundError(f"Transitions CSV not found: {transitions_csv}")

    if not scenario_ids:
        raise ValueError("scenario_ids must not be empty.")

    df = pd.read_csv(transitions_csv)

    if "scenario_id" not in df.columns:
        raise ValueError(
            f"Transitions CSV must contain scenario_id column: {transitions_csv}"
        )

    selected_ids = {int(value) for value in scenario_ids}

    selected = df[df["scenario_id"].astype(int).isin(selected_ids)].copy()

    if selected.empty:
        raise ValueError(
            "No selected scenario IDs were found in transitions CSV. "
            f"CSV: {transitions_csv}"
        )

    found_ids = set(int(value) for value in selected["scenario_id"].unique())
    missing_ids = sorted(selected_ids - found_ids)

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} selected scenario IDs are missing in "
            f"{transitions_csv}. Examples: {missing_ids[:20]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)

    return output_path


def ensure_outcome_value_targets(
    examples_csv: str | Path,
    *,
    gamma: float,
) -> Path:
    """
    Ensure examples.csv contains strict outcome_value_target.

    Older generation outputs may lack outcome_value_target.
    GraphSelfPlayDataset requires it.
    """

    examples_csv = Path(examples_csv)

    if not examples_csv.exists():
        raise FileNotFoundError(f"Examples CSV not found: {examples_csv}")

    df = pd.read_csv(examples_csv)

    if df.empty:
        raise ValueError(f"Examples CSV is empty: {examples_csv}")

    if "outcome_value_target" in df.columns:
        print(f"outcome_value_target already exists: {examples_csv}")
        return examples_csv

    rows = df.to_dict(orient="records")

    add_outcome_value_targets_to_rows(
        rows=rows,
        gamma=float(gamma),
        group_keys=("scenario_id",),
    )

    updated = pd.DataFrame(rows)
    updated.to_csv(examples_csv, index=False)

    print(f"Added outcome_value_target to: {examples_csv}")

    return examples_csv


def run_generate(
    *,
    project_root: str | Path,
    raw_dir: str | Path,
    transitions_csv: str | Path,
    scenario_ids: list[int],
    checkpoint: str | Path,
    output_dir: str | Path,
    config: GenerationConfig,
    base_seed: int,
    iteration: int,
) -> Path:
    """
    Generate model-guided self-play examples for sampled scenarios.

    Uses the self-play generation Python API.
    """

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_transitions_csv = write_selected_transitions_csv(
        transitions_csv=transitions_csv,
        scenario_ids=scenario_ids,
        output_path=output_dir / "selected_transitions.csv",
    )

    log_path = output_dir / "generate.log"
    iteration_seed = int(base_seed) + int(iteration)

    request = GenerationRequest(
        raw_dir=Path(raw_dir),
        transitions_csv=selected_transitions_csv,
        output_dir=output_dir,
        checkpoint=Path(checkpoint),
        config=config,
        seed=iteration_seed,
        clear_cache_between_scenarios=True,
    )

    with _working_directory(project_root):
        with _stage_output(log_path):
            examples_csv = generate_self_play_examples(request)

    if not examples_csv.exists():
        raise FileNotFoundError(f"Examples CSV was not created: {examples_csv}")

    ensure_outcome_value_targets(
        examples_csv=examples_csv,
        gamma=config.gamma,
    )

    return examples_csv


def run_train(
    *,
    project_root: str | Path,
    examples_csv: str | Path,
    init_checkpoint: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    iteration: int,
) -> Path:
    """
    Fine-tune model from previous best checkpoint.

    Uses the graph policy/value training Python API.
    """

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_checkpoint = output_dir / "candidate_checkpoint.pt"
    metrics_csv = output_dir / "train_metrics.csv"
    log_path = output_dir / "train.log"

    request = TrainingRequest(
        project_root=project_root.resolve(),
        examples_csv=Path(examples_csv),
        output_path=candidate_checkpoint,
        config=config,
        init_checkpoint=Path(init_checkpoint),
        validation_examples_csv=None,
        use_amp=False,
        normalize_features=True,
        save_best=True,
        tensorboard_log_dir=None,
        run_name=f"self_play_iter_{int(iteration):03d}",
        metrics_csv=metrics_csv,
    )

    with _working_directory(project_root):
        with _stage_output(log_path):
            result_path = train_graph_policy_value_model(request)

    if Path(result_path) != candidate_checkpoint:
        raise RuntimeError(
            "Training API returned unexpected checkpoint path: "
            f"{result_path} != {candidate_checkpoint}"
        )

    if not candidate_checkpoint.exists():
        raise FileNotFoundError(
            f"Candidate checkpoint was not created: {candidate_checkpoint}"
        )

    return candidate_checkpoint


def run_evaluate(
    *,
    project_root: str | Path,
    checkpoint: str | Path,
    eval_csv: str | Path,
    eval_raw_dir: str | Path,
    output_dir: str | Path,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """
    Evaluate candidate checkpoint on fixed eval set.
    """

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / config.output_json_name
    output_csv = output_dir / config.output_csv_name
    log_path = output_dir / "evaluate.log"

    request = EvaluationRequest(
        raw_dir=Path(eval_raw_dir),
        transitions_csv=Path(eval_csv),
        checkpoint=Path(checkpoint),
        config=config,
        output_csv=output_csv,
        output_json=output_json,
        limit=None,
        quiet=True,
    )

    with _working_directory(project_root):
        with _stage_output(log_path):
            metrics = evaluate_checkpoint(request)

    if not isinstance(metrics, Mapping):
        raise TypeError("Evaluation API returned non-mapping metrics.")

    if not output_json.exists():
        raise FileNotFoundError(f"Evaluation JSON was not created: {output_json}")

    return load_json(output_json)


def save_iteration_metadata(
    *,
    iteration: int,
    path: str | Path,
    accepted: bool,
    parent_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    train_batch_csv: str | Path,
    raw_examples_csv: str | Path | None,
    metrics: dict[str, Any],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save reproducibility metadata for one self-play iteration.
    """

    path = Path(path)

    parent_checkpoint = Path(parent_checkpoint)
    candidate_checkpoint = Path(candidate_checkpoint)
    train_batch_csv = Path(train_batch_csv)

    payload: dict[str, Any] = {
        "iteration": int(iteration),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": bool(accepted),
        "parent_checkpoint": str(parent_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "train_batch_csv": str(train_batch_csv),
        "raw_examples_csv": None if raw_examples_csv is None else str(raw_examples_csv),
        "hashes": {},
        "metrics": metrics,
        "config": config,
    }

    for name, file_path in {
        "parent_checkpoint_sha256": parent_checkpoint,
        "candidate_checkpoint_sha256": candidate_checkpoint,
        "train_batch_csv_sha256": train_batch_csv,
    }.items():
        if file_path.exists():
            payload["hashes"][name] = sha256_file(file_path)

    if raw_examples_csv is not None:
        raw_examples_path = Path(raw_examples_csv)

        if raw_examples_path.exists():
            payload["hashes"]["raw_examples_csv_sha256"] = sha256_file(
                raw_examples_path
            )

    if extra is not None:
        payload["extra"] = extra

    save_json(payload, path)

    return path

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.config import (
    EvaluationConfig,
    GenerationConfig,
    TrainingConfig,
)
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
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


def run_command(
    command: list[str],
    *,
    cwd: str | Path,
    log_path: str | Path | None = None,
) -> None:
    """
    Run subprocess command.

    Output is streamed to console and optionally saved to a log file.
    """

    cwd = Path(cwd)
    log_file = None

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    print("")
    print("=" * 100)
    print("RUN COMMAND")
    print("=" * 100)
    print(" ".join(str(part) for part in command))
    print(f"cwd: {cwd}")

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    try:
        for line in process.stdout:
            print(line, end="")

            if log_file is not None:
                log_file.write(line)

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(
                returncode=return_code,
                cmd=command,
            )

    finally:
        if log_file is not None:
            log_file.close()


def write_selected_transitions_csv(
    *,
    transitions_csv: str | Path,
    scenario_ids: list[int],
    output_path: str | Path,
) -> Path:
    """
    Create a temporary transitions CSV containing only sampled scenario IDs.

    This is needed because scripts.self_play.generate currently reads scenario
    IDs from --transitions and does not accept --scenario-ids directly.
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

    scripts.self_play.generate may produce old-style examples without
    outcome_value_target. GraphSelfPlayDataset requires it.
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


def _append_bool_flag(
    command: list[str],
    flag: str,
    enabled: bool,
) -> None:
    if bool(enabled):
        command.append(flag)


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

    Uses scripts.self_play.generate because it supports --checkpoint.
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

    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.self_play.generate",
        str(Path(raw_dir)),
        "--transitions",
        str(selected_transitions_csv),
        "--output-dir",
        str(output_dir),
        "--checkpoint",
        str(Path(checkpoint)),
        "--simulations",
        str(config.simulations),
        "--depth",
        str(config.depth),
        "--max-steps",
        str(config.max_steps),
        "--top-k",
        str(config.top_k),
        "--gamma",
        str(config.gamma),
        "--c-puct",
        str(config.c_puct),
        "--prior-exponent",
        str(config.prior_exponent),
        "--selection-temperature",
        str(config.selection_temperature),
        "--seed",
        str(iteration_seed),
        "--pf-alg",
        str(config.pf_alg),
        "--terminal-unsolved-penalty",
        str(config.terminal_unsolved_penalty),
        "--terminal-handoff-penalty",
        str(config.terminal_handoff_penalty),
        "--terminal-failure-penalty",
        str(config.terminal_failure_penalty),
        "--terminal-penalty-weight",
        str(config.terminal_penalty_weight),
        "--stop-policy",
        config.stop_policy,
        "--clear-cache-between-scenarios",
    ]

    _append_bool_flag(
        command,
        "--use-root-noise",
        config.use_root_noise,
    )

    _append_bool_flag(
        command,
        "--use-continuation-gate",
        config.use_continuation_gate,
    )

    run_command(
        command,
        cwd=project_root,
        log_path=log_path,
    )

    examples_csv = output_dir / "examples.csv"

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

    Requires train_graph_baseline.py to support --init-checkpoint.
    """

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_checkpoint = output_dir / "candidate_checkpoint.pt"
    metrics_csv = output_dir / "train_metrics.csv"
    log_path = output_dir / "train.log"

    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.self_play.train_graph_baseline",
        str(Path(examples_csv)),
        "--init-checkpoint",
        str(Path(init_checkpoint)),
        "--output",
        str(candidate_checkpoint),
        "--metrics-csv",
        str(metrics_csv),
        "--run-name",
        f"self_play_iter_{int(iteration):03d}",
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--lr",
        str(config.learning_rate),
        "--value-loss-weight",
        str(config.value_loss_weight),
        "--value-huber-delta",
        str(config.value_huber_delta),
        "--device",
        config.device,
        "--num-workers",
        str(config.num_workers),
        "--model-type",
        config.model_type,
        "--hidden-dim",
        str(config.hidden_dim),
        "--num-layers",
        str(config.num_layers),
        "--dropout",
        str(config.dropout),
        "--save-best",
    ]

    if config.save_multiple_best:
        command.append("--save-multiple-best")

    if config.no_tensorboard:
        command.append("--no-tensorboard")

    run_command(
        command,
        cwd=project_root,
        log_path=log_path,
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

    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.evaluation.evaluate_checkpoint",
        str(Path(eval_raw_dir)),
        "--transitions",
        str(Path(eval_csv)),
        "--checkpoint",
        str(Path(checkpoint)),
        "--output-csv",
        str(output_csv),
        "--output-json",
        str(output_json),
        "--simulations",
        str(config.simulations),
        "--depth",
        str(config.depth),
        "--max-steps",
        str(config.max_steps),
        "--top-k",
        str(config.top_k),
        "--gamma",
        str(config.gamma),
        "--c-puct",
        str(config.c_puct),
        "--prior-exponent",
        str(config.prior_exponent),
        "--num-workers",
        str(config.num_workers),
        "--batch-size",
        str(config.batch_size),
        "--device",
        config.device,
        "--quiet",
    ]

    _append_bool_flag(
        command,
        "--use-continuation-gate",
        config.use_continuation_gate,
    )

    _append_bool_flag(
        command,
        "--allow-handoff-with-hard-overloads",
        config.allow_handoff_with_hard_overloads,
    )

    run_command(
        command,
        cwd=project_root,
        log_path=log_path,
    )

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

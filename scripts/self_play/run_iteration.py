from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


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
    config: dict[str, Any],
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

    gamma = float(config.get("gamma", 0.95))

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
        str(int(config.get("simulations", 150))),
        "--depth",
        str(int(config.get("depth", 4))),
        "--max-steps",
        str(int(config.get("max_steps", 5))),
        "--top-k",
        str(int(config.get("top_k", 30))),
        "--gamma",
        str(gamma),
        "--c-puct",
        str(float(config.get("c_puct", 2.0))),
        "--prior-exponent",
        str(float(config.get("prior_exponent", 0.5))),
        "--selection-temperature",
        str(float(config.get("selection_temperature", 0.0))),
        "--seed",
        str(int(config.get("seed", 42)) + int(iteration)),
        "--pf-alg",
        str(int(config.get("pf_alg", 3))),
        "--terminal-unsolved-penalty",
        str(float(config.get("terminal_unsolved_penalty", 500.0))),
        "--terminal-handoff-penalty",
        str(float(config.get("terminal_handoff_penalty", 150.0))),
        "--terminal-failure-penalty",
        str(float(config.get("terminal_failure_penalty", 1000.0))),
        "--terminal-penalty-weight",
        str(float(config.get("terminal_penalty_weight", 0.10))),
        "--stop-policy",
        str(config.get("stop_policy", "no_hard_overloads")),
        "--clear-cache-between-scenarios",
    ]

    _append_bool_flag(
        command,
        "--use-root-noise",
        bool(config.get("use_root_noise", True)),
    )

    _append_bool_flag(
        command,
        "--use-continuation-gate",
        bool(config.get("use_continuation_gate", True)),
    )

    run_command(
        command,
        cwd=project_root,
        log_path=log_path,
    )

    examples_csv = output_dir / "examples.csv"

    ensure_outcome_value_targets(
        examples_csv=examples_csv,
        gamma=gamma,
    )

    return examples_csv


def run_train(
    *,
    project_root: str | Path,
    examples_csv: str | Path,
    init_checkpoint: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
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
        str(int(config.get("epochs", config.get("epochs_per_iteration", 10)))),
        "--batch-size",
        str(int(config.get("batch_size", 64))),
        "--lr",
        str(float(config.get("learning_rate", 0.0003))),
        "--value-loss-weight",
        str(float(config.get("value_loss_weight", 1.0))),
        "--value-huber-delta",
        str(float(config.get("value_huber_delta", 1.0))),
        "--device",
        str(config.get("device", "auto")),
        "--num-workers",
        str(int(config.get("num_workers", 0))),
        "--model-type",
        str(config.get("model_type", "graph_v2")),
        "--hidden-dim",
        str(int(config.get("hidden_dim", 128))),
        "--num-layers",
        str(int(config.get("num_layers", 3))),
        "--dropout",
        str(float(config.get("dropout", 0.0))),
        "--save-best",
    ]

    if bool(config.get("save_multiple_best", False)):
        command.append("--save-multiple-best")

    if bool(config.get("no_tensorboard", True)):
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
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate candidate checkpoint on fixed eval set.
    """

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / str(config.get("output_json_name", "eval_metrics.json"))
    output_csv = output_dir / str(config.get("output_csv_name", "eval_results.csv"))
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
        str(int(config.get("simulations", 150))),
        "--depth",
        str(int(config.get("depth", 4))),
        "--max-steps",
        str(int(config.get("max_steps", 5))),
        "--top-k",
        str(int(config.get("top_k", 30))),
        "--gamma",
        str(float(config.get("gamma", 0.95))),
        "--c-puct",
        str(float(config.get("c_puct", 2.0))),
        "--prior-exponent",
        str(float(config.get("prior_exponent", 0.5))),
        "--num-workers",
        str(int(config.get("num_workers", 1))),
        "--batch-size",
        str(int(config.get("batch_size", 5))),
        "--device",
        str(config.get("device", "cpu")),
        "--quiet",
    ]

    _append_bool_flag(
        command,
        "--use-continuation-gate",
        bool(config.get("use_continuation_gate", True)),
    )

    _append_bool_flag(
        command,
        "--allow-handoff-with-hard-overloads",
        bool(config.get("allow_handoff_with_hard_overloads", False)),
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


def accept_candidate(
    *,
    new_metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    policy: dict[str, Any],
) -> bool:
    """
    Acceptance policy for candidate checkpoint.
    """

    metric = str(policy.get("metric", "solve_rate"))
    min_improvement = float(policy.get("min_improvement", 0.0))

    if metric not in new_metrics:
        raise KeyError(f"Metric {metric!r} not found in new_metrics.")

    if metric not in best_metrics:
        raise KeyError(f"Metric {metric!r} not found in best_metrics.")

    improvement = float(new_metrics[metric]) - float(best_metrics[metric])

    # Never replace the best model on an exact tie or numerical noise.
    comparison_epsilon = 1e-12

    if improvement <= comparison_epsilon:
        return False

    if improvement + comparison_epsilon < min_improvement:
        return False

    simple_guard = float(policy.get("max_simple_solve_rate_drop", 0.05))

    if (
        "solve_rate_simple" in new_metrics
        and "solve_rate_simple" in best_metrics
    ):
        if (
            float(new_metrics["solve_rate_simple"])
            < float(best_metrics["solve_rate_simple"]) - simple_guard
        ):
            return False

    max_failed = policy.get("reject_if_failed_scenarios_above")

    if max_failed is not None and "failed_scenarios" in new_metrics:
        if int(new_metrics["failed_scenarios"]) > int(max_failed):
            return False

    return True
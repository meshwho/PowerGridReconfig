from __future__ import annotations

import io
import math
import os
import sys
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.config import (
    EvaluationConfig,
    GenerationConfig,
    TrainingConfig,
)
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import (
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
    require_checkpoint_contracts,
)
from grid_topology_ai.evaluation.checkpoint import (
    EvaluationRequest,
    evaluate_checkpoint,
)
from grid_topology_ai.self_play.artifacts import load_json, save_json, sha256_file
from grid_topology_ai.self_play.example_validation import (
    validate_example_contract_versions,
    validate_example_outcome_contracts,
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
            # Intentional top-level logging boundary:
            # print the traceback and re-raise the original exception unchanged.
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


def _coerce_integer_scenario_id(value: object) -> int:
    if pd.isna(value):
        raise ValueError("scenario_id must not be missing.")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("scenario_id must not be boolean.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(f"scenario_id must be integer-valued, got {value!r}.")
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("scenario_id must not be empty.")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"scenario_id cannot be converted to an integer: {value!r}.") from exc
    if not number.is_integer():
        raise ValueError(f"scenario_id must be integer-valued, got {value!r}.")
    if str(int(number)) != text and not text.endswith(".0"):
        raise ValueError(f"scenario_id is not an unambiguous integer: {value!r}.")
    return int(number)


def split_examples_by_scenario(
    *,
    examples_csv: str | Path,
    train_output_csv: str | Path,
    validation_output_csv: str | Path,
    metadata_output_json: str | Path,
    validation_fraction: float,
    min_validation_scenarios: int,
    seed: int,
) -> dict[str, object]:
    examples_path = Path(examples_csv)
    train_path = Path(train_output_csv)
    validation_path = Path(validation_output_csv)
    metadata_path = Path(metadata_output_json)

    output_paths = {train_path.resolve(), validation_path.resolve(), metadata_path.resolve()}
    if len(output_paths) != 3:
        raise ValueError("Split output paths must be distinct.")
    if examples_path.resolve() in output_paths:
        raise ValueError("Split output paths must not overwrite examples_csv.")
    if not examples_path.is_file():
        raise FileNotFoundError(f"Examples CSV not found: {examples_path}")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    if int(min_validation_scenarios) <= 0:
        raise ValueError("min_validation_scenarios must be > 0.")

    try:
        df = pd.read_csv(examples_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Examples CSV is empty: {examples_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse examples CSV: {examples_path}") from exc
    if df.empty:
        raise ValueError(f"Examples CSV contains no rows: {examples_path}")
    if "scenario_id" not in df.columns:
        raise ValueError("Examples CSV must contain scenario_id column.")
    physics_config = validate_example_contract_versions(
        df,
        source_path=examples_path,
    )

    scenario_ids = df["scenario_id"].map(_coerce_integer_scenario_id)
    df = df.copy()
    df["scenario_id"] = scenario_ids
    all_scenarios = sorted(int(value) for value in scenario_ids.unique())
    total_scenarios = len(all_scenarios)
    if total_scenarios < 2:
        raise ValueError("Scenario-level split requires at least two unique scenario_id values.")

    n_validation = max(
        int(min_validation_scenarios),
        int(math.ceil(total_scenarios * float(validation_fraction))),
    )
    if total_scenarios - n_validation < 1:
        raise ValueError(
            "Requested validation split leaves no training scenarios: "
            f"total_scenarios={total_scenarios}, validation_scenarios={n_validation}."
        )

    rng = np.random.default_rng(int(seed))
    validation_ids = sorted(int(value) for value in rng.choice(all_scenarios, size=n_validation, replace=False).tolist())
    validation_set = set(validation_ids)
    train_ids = sorted(value for value in all_scenarios if value not in validation_set)
    train_set = set(train_ids)

    train_df = df[df["scenario_id"].isin(train_set)].copy()
    validation_df = df[df["scenario_id"].isin(validation_set)].copy()

    if train_df.empty or validation_df.empty:
        raise RuntimeError("Scenario split produced an empty train or validation CSV.")
    if train_set & validation_set:
        raise RuntimeError("Scenario split leakage detected between train and validation IDs.")
    if train_set | validation_set != set(all_scenarios):
        raise RuntimeError("Scenario split does not cover all input scenarios.")
    if len(train_df) + len(validation_df) != len(df):
        raise RuntimeError("Scenario split did not preserve every input row exactly once.")
    if "state_id" in df.columns:
        train_states = set(train_df["state_id"].astype(str).tolist())
        val_states = set(validation_df["state_id"].astype(str).tolist())
        overlap = train_states & val_states
        if overlap:
            raise RuntimeError(f"state_id leakage detected across split: {sorted(overlap)[:5]}")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    metadata: dict[str, object] = {
        "schema_version": 1,
        **physics_provenance(physics_config),
        "source_csv": str(examples_path),
        "train_csv": str(train_path),
        "validation_csv": str(validation_path),
        "seed": int(seed),
        "validation_fraction_target": float(validation_fraction),
        "min_validation_scenarios": int(min_validation_scenarios),
        "total_examples": int(len(df)),
        "train_examples": int(len(train_df)),
        "validation_examples": int(len(validation_df)),
        "total_scenarios": int(total_scenarios),
        "train_scenarios": int(len(train_ids)),
        "validation_scenarios": int(len(validation_ids)),
        "validation_scenario_ids": validation_ids,
        "source_csv_sha256": sha256_file(examples_path),
        "train_csv_sha256": sha256_file(train_path),
        "validation_csv_sha256": sha256_file(validation_path),
    }
    save_json(metadata, metadata_path)
    return metadata


def ensure_outcome_value_targets(
    examples_csv: str | Path,
    *,
    gamma: float,
) -> Path:
    """
    Ensure examples.csv contains strict outcome_value_target.

    Only freshly generated rows carrying the current physical-objective
    provenance may receive current value targets. Legacy solved labels are not
    scientifically upgradeable and must be regenerated.
    """

    examples_csv = Path(examples_csv)

    if not examples_csv.exists():
        raise FileNotFoundError(f"Examples CSV not found: {examples_csv}")

    df = pd.read_csv(examples_csv)

    if df.empty:
        raise ValueError(f"Examples CSV is empty: {examples_csv}")

    if "physical_objective_schema_version" not in df.columns:
        raise ValueError(
            "Examples CSV predates the current physical-objective contract. "
            "Regenerate episodes with python -m scripts.self_play.generate; "
            "legacy solved labels cannot be upgraded in place."
        )

    if "outcome_value_target" in df.columns:
        validate_example_contract_versions(df, source_path=examples_csv)
        validate_example_outcome_contracts(df, source_path=examples_csv)
        print(f"outcome_value_target already exists: {examples_csv}")
        return examples_csv

    if "outcome_value_target_contract_version" in df.columns:
        observed = set(df["outcome_value_target_contract_version"].tolist())
        if observed != {OUTCOME_VALUE_TARGET_CONTRACT_VERSION}:
            raise ValueError(
                "Examples CSV has incompatible outcome target provenance. "
                "Regenerate episodes and targets instead of rewriting versions."
            )

    rows = df.to_dict(orient="records")

    add_outcome_value_targets_to_rows(
        rows=rows,
        gamma=gamma,
        group_keys=("scenario_id",),
    )

    updated = pd.DataFrame(rows)
    validate_example_contract_versions(updated, source_path=examples_csv)
    validate_example_outcome_contracts(updated, source_path=examples_csv)
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
    physics_config: PhysicsConfig | None = None,
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
        physics_config=physics_config,
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
    validation_examples_csv: str | Path,
    init_checkpoint: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    physics_config: PhysicsConfig,
    iteration: int,
    seed: int,
) -> Path:
    """
    Fine-tune model from previous best checkpoint.

    Uses the graph policy/value training Python API.
    """

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_csv = Path(examples_csv)
    validation_examples_csv = Path(validation_examples_csv)
    if not examples_csv.is_file():
        raise FileNotFoundError(f"Training examples CSV not found: {examples_csv}")
    if not validation_examples_csv.is_file():
        raise FileNotFoundError(f"Validation examples CSV not found: {validation_examples_csv}")

    candidate_checkpoint = output_dir / "candidate_checkpoint.pt"
    metrics_csv = output_dir / "train_metrics.csv"
    log_path = output_dir / "train.log"

    request = TrainingRequest(
        project_root=project_root.resolve(),
        examples_csv=examples_csv,
        output_path=candidate_checkpoint,
        config=config,
        init_checkpoint=Path(init_checkpoint),
        validation_examples_csv=validation_examples_csv,
        use_amp=False,
        normalize_features=True,
        save_best=True,
        tensorboard_log_dir=None,
        run_name=f"self_play_iter_{int(iteration):03d}",
        metrics_csv=metrics_csv,
        seed=int(seed),
        physics_config=physics_config,
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

    import torch

    checkpoint = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Candidate checkpoint payload must be a mapping.")
    require_checkpoint_contracts(
        checkpoint,
        source=str(candidate_checkpoint),
        expected_physics_config=physics_config,
    )
    if checkpoint.get("checkpoint_selection_metric") != "validation_loss":
        raise RuntimeError(
            "Self-play fine-tuning candidate must be selected by validation_loss; "
            f"observed {checkpoint.get('checkpoint_selection_metric')!r}."
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
    physics_config: PhysicsConfig | None = None,
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
        physics_config=physics_config,
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

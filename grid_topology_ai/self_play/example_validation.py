from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.outcome_contract import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.value_targets import (
    terminal_evidence_from_row,
    terminal_value_from_outcome,
)
from grid_topology_ai.self_play import _example_validation_core as _core
from grid_topology_ai.self_play._example_validation_core import *  # noqa: F401,F403
from grid_topology_ai.contracts import (
    require_outcome_objective_version,
)

_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "terminal_outcome_evidence_schema_version",
    "terminal_outcome_evidence_json",
)
_IDENTITY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "iteration",
    "episode_id",
)

REQUIRED_OUTCOME_COLUMNS: tuple[str, ...] = (
    _core.REQUIRED_OUTCOME_COLUMNS + _EVIDENCE_COLUMNS
)
REQUIRED_EXAMPLE_COLUMNS: tuple[str, ...] = (
    _core.REQUIRED_EXAMPLE_COLUMNS
    + _EVIDENCE_COLUMNS
    + _IDENTITY_COLUMNS
)


def load_and_validate_examples_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Examples CSV not found: {path}")
    try:
        examples = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Examples CSV has no readable columns: {path}"
        ) from exc
    validate_examples_dataframe(examples, source_path=path)
    return examples


def validate_examples_dataframe(
    examples: pd.DataFrame,
    *,
    source_path: str | Path,
) -> None:
    source = Path(source_path)
    if len(examples.columns) == 0:
        raise ValueError(
            f"Examples CSV has no readable columns: {source}"
        )

    missing = sorted(
        set(REQUIRED_EXAMPLE_COLUMNS) - set(examples.columns)
    )
    if missing:
        raise ValueError(
            "Examples CSV is missing required columns: "
            f"{missing}. File: {source}"
        )
    if examples.empty:
        raise ValueError(f"Examples CSV is empty: {source}")

    physics_config = _core.validate_example_contract_versions(
        examples,
        source_path=source,
    )
    (
        action_space_config,
        action_layout,
    ) = _core.validate_example_topology_action_contracts(
        examples,
        source_path=source,
    )

    for column in REQUIRED_EXAMPLE_COLUMNS:
        for index, value in examples[column].items():
            if _core._is_missing_required_value(value):
                raise ValueError(
                    f"Missing required value in column '{column}' "
                    f"at row {index}. File: {source}"
                )

    _validate_episode_identity(examples, source=source)
    validate_example_outcome_contracts(
        examples,
        source_path=source,
    )

    state_ids = examples["state_id"].map(
        lambda value: str(value).strip()
    )
    duplicated = state_ids[state_ids.duplicated()]
    if not duplicated.empty:
        raise ValueError(
            f"Duplicate state_id '{duplicated.iloc[0]}' in examples "
            f"CSV. File: {source}"
        )

    expected_dimensions = None
    for index, row in examples.iterrows():
        _core._require_integer(
            row["scenario_id"],
            column="scenario_id",
            index=index,
            source=source,
        )
        step = _core._require_integer(
            row["step"],
            column="step",
            index=index,
            source=source,
        )
        if step < 0:
            raise ValueError(
                f"step must be >= 0 at row {index}. File: {source}"
            )

        policy = _core._parse_policy(
            row["mcts_policy_json"],
            index=index,
            source=source,
        )
        state_path = Path(str(row["state_path"]).strip())
        if not state_path.exists():
            raise FileNotFoundError(
                f"State file not found: {state_path}. File: {source}"
            )
        if not state_path.is_file():
            raise ValueError(
                f"State path is not a file: {state_path}. "
                f"File: {source}"
            )

        dimensions, action_mask = _core._validate_npz_state(
            state_path,
            expected_physics_config=physics_config,
            expected_action_space_config=action_space_config,
            expected_action_layout=action_layout,
        )
        if expected_dimensions is None:
            expected_dimensions = dimensions
        elif dimensions != expected_dimensions:
            raise ValueError(
                f"Graph dimensions mismatch for {state_path}. "
                f"Expected {expected_dimensions._asdict()}, "
                f"observed {dimensions._asdict()}."
            )

        _core._validate_policy_against_mask(
            policy,
            action_mask=action_mask,
            index=index,
            source=source,
        )

        if (
            "selected_action_id" in examples.columns
            and not _core._is_missing_required_value(
                row["selected_action_id"]
            )
        ):
            selected = _core._require_integer(
                row["selected_action_id"],
                column="selected_action_id",
                index=index,
                source=source,
            )
            if (
                selected < 0
                or selected >= len(action_mask)
                or not bool(action_mask[selected])
            ):
                raise ValueError(
                    f"selected_action_id {selected} is invalid for "
                    f"action_mask at row {index}. File: {source}"
                )

        evidence = terminal_evidence_from_row(
            row.to_dict(),
            context=f"{source} row {index}",
        )
        _validate_state_terminal_evidence(
            state_path,
            expected_evidence=evidence,
            expected_identity={
                "run_id": row["run_id"],
                "iteration": int(row["iteration"]),
                "episode_id": row["episode_id"],
            },
            index=index,
            source=source,
        )


def _validate_episode_identity(
    examples: pd.DataFrame,
    *,
    source: Path,
) -> None:
    for index, row in examples.iterrows():
        for column in ("run_id", "episode_id"):
            value = row[column]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{column} must be a non-empty string at row "
                    f"{index}. File: {source}"
                )
        iteration = _core._require_integer(
            row["iteration"],
            column="iteration",
            index=index,
            source=source,
        )
        if iteration <= 0:
            raise ValueError(
                f"iteration must be > 0 at row {index}. File: {source}"
            )

    for episode_id, group in examples.groupby(
        "episode_id",
        sort=False,
    ):
        for column in ("run_id", "iteration", "scenario_id"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Episode {episode_id!r} contains mixed {column} "
                    f"values. File: {source}"
                )
        steps = sorted(
            _core._require_integer(
                value,
                column="step",
                index=index,
                source=source,
            )
            for index, value in group["step"].items()
        )
        if steps != list(range(len(steps))):
            raise ValueError(
                f"Episode {episode_id!r} must use contiguous steps "
                f"from 0. File: {source}"
            )


def validate_example_outcome_contracts(
    examples: pd.DataFrame,
    *,
    source_path: str | Path,
) -> None:
    source = Path(source_path)
    missing = sorted(
        set(REQUIRED_OUTCOME_COLUMNS) - set(examples.columns)
    )
    if missing:
        raise ValueError(
            "Examples CSV is missing required outcome columns: "
            f"{missing}. File: {source}"
        )

    for index, row in examples.iterrows():
        for column in REQUIRED_OUTCOME_COLUMNS:
            if _core._is_missing_required_value(row[column]):
                raise ValueError(
                    f"Missing required outcome value in column "
                    f"'{column}' at row {index}. File: {source}"
                )
        _core.require_exact_contract_version(
            row.get("outcome_value_target_contract_version"),
            expected=_core.OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
            name="outcome/value-target contract",
            source=f"{source} row {index}",
            regeneration_command=(
                "python -m " + "scripts" + ".self_play.generate ..."
            ),
        )
        require_outcome_objective_version(
            row.to_dict(),
            source=f"{source} row {index}",
        )
        _validate_outcome_contract(
            row,
            index=index,
            source=source,
        )


def _validate_outcome_contract(
    row: pd.Series,
    *,
    index: Any,
    source: Path,
) -> None:
    solved = _core._require_bool(
        row["solved"],
        column="solved",
        index=index,
        source=source,
    )
    done = _core._require_bool(
        row["done"],
        column="done",
        index=index,
        source=source,
    )
    if not done:
        raise ValueError(
            "Training example must carry a terminal episode outcome "
            f"at row {index}. File: {source}"
        )

    evidence = terminal_evidence_from_row(
        row.to_dict(),
        context=f"{source} row {index}",
    )
    reason = evidence.termination_reason
    if evidence.solved is not solved:
        raise ValueError(
            "terminal outcome evidence contradicts solved at row "
            f"{index}. File: {source}"
        )

    steps_to_terminal = _core._require_integer(
        row["outcome_steps_to_terminal"],
        column="outcome_steps_to_terminal",
        index=index,
        source=source,
    )
    if steps_to_terminal <= 0:
        raise ValueError(
            "outcome_steps_to_terminal must be > 0 at row "
            f"{index}. File: {source}"
        )

    gamma = _core._require_finite_number(
        row["outcome_gamma"],
        column="outcome_gamma",
        index=index,
        source=source,
    )
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(
            f"outcome_gamma must be in [0, 1] at row {index}. "
            f"File: {source}"
        )

    actual_target = _core._require_finite_number(
        row["outcome_value_target"],
        column="outcome_value_target",
        index=index,
        source=source,
    )
    if abs(actual_target) > 1.0 + 1e-6:
        raise ValueError(
            "outcome_value_target outside [-1, 1] at row "
            f"{index}. File: {source}"
        )

    terminal_value, expected_class = terminal_value_from_outcome(
        solved=solved,
        termination_reason=reason,
        evidence=evidence,
    )
    expected_target = float(
        terminal_value * gamma**steps_to_terminal
    )
    if not math.isclose(
        actual_target,
        expected_target,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise ValueError(
            "outcome_value_target contradicts the terminal outcome "
            f"at row {index}: expected {expected_target}, "
            f"observed {actual_target}. File: {source}"
        )

    actual_class = str(row["outcome_class"]).strip()
    if actual_class != expected_class:
        raise ValueError(
            "outcome_class contradicts the terminal outcome at row "
            f"{index}: expected {expected_class!r}, "
            f"observed {actual_class!r}. File: {source}"
        )

    mode = str(row["outcome_value_target_mode"]).strip()
    if mode != "alphazero_discounted":
        raise ValueError(
            f"Unsupported outcome_value_target_mode {mode!r} at "
            f"row {index}. File: {source}"
        )


def _validate_state_terminal_evidence(
    state_path: Path,
    *,
    expected_evidence: TerminalOutcomeEvidence,
    expected_identity: Mapping[str, object],
    index: Any,
    source: Path,
) -> None:
    metadata = _load_state_metadata(state_path)

    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"CSV {field} does not match state metadata at row "
                f"{index}. File: {source}. State: {state_path}"
            )

    version = metadata.get(
        "terminal_outcome_evidence_schema_version"
    )
    if (
        isinstance(version, (bool, np.bool_))
        or not isinstance(version, (int, np.integer))
        or int(version)
        != TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError(
            "State terminal outcome evidence schema version mismatch "
            f"at row {index}: {version!r}. File: {source}. "
            f"State: {state_path}"
        )

    raw_evidence = metadata.get("terminal_outcome_evidence")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError(
            "State metadata is missing terminal_outcome_evidence "
            f"at row {index}. File: {source}. State: {state_path}"
        )

    try:
        observed_evidence = TerminalOutcomeEvidence.from_mapping(
            raw_evidence
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid terminal_outcome_evidence in state metadata "
            f"at row {index}. File: {source}. State: {state_path}: "
            f"{exc}"
        ) from exc

    if observed_evidence != expected_evidence:
        raise ValueError(
            "CSV terminal outcome evidence does not match state "
            f"metadata at row {index}. File: {source}. "
            f"State: {state_path}"
        )


def _load_state_metadata(state_path: Path) -> Mapping[str, object]:
    try:
        with np.load(state_path, allow_pickle=False) as data:
            if "metadata_json" not in data.files:
                raise ValueError(
                    "State NPZ is missing required metadata_json: "
                    f"{state_path}"
                )
            raw_metadata = np.asarray(data["metadata_json"])
    except (OSError, EOFError) as exc:
        raise ValueError(
            f"Could not read NPZ state: {state_path}"
        ) from exc

    if raw_metadata.size != 1:
        raise ValueError(
            "State metadata_json must contain one JSON object: "
            f"{state_path}"
        )
    try:
        metadata = json.loads(str(raw_metadata.item()))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Invalid state metadata_json: {state_path}"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"State metadata_json must be an object: {state_path}"
        )
    return metadata

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from grid_topology_ai.contracts import (
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    require_exact_contract_version,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.termination import (
    parse_termination_reason,
    validate_outcome_invariants,
)
from grid_topology_ai.value_targets import terminal_value_from_outcome

REQUIRED_OUTCOME_COLUMNS: tuple[str, ...] = (
    "outcome_value_target",
    "outcome_value_target_contract_version",
    "solved",
    "done",
    "termination_reason",
    "outcome_class",
    "outcome_steps_to_terminal",
    "outcome_value_target_mode",
    "outcome_gamma",
)

REQUIRED_EXAMPLE_COLUMNS: tuple[str, ...] = (
    "state_path",
    "mcts_policy_json",
    "scenario_id",
    "step",
    "state_id",
    "physical_objective_schema_version",
) + REQUIRED_OUTCOME_COLUMNS

_REQUIRED_STATE_ARRAYS = ("bus_features", "branch_features", "edge_index", "action_mask")


class _GraphDimensions(NamedTuple):
    num_buses: int
    num_branches: int
    num_bus_features: int
    num_branch_features: int
    num_actions: int


def load_and_validate_examples_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Examples CSV not found: {path}")
    try:
        examples = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Examples CSV has no readable columns: {path}") from exc
    validate_examples_dataframe(examples, source_path=path)
    return examples


def validate_examples_dataframe(examples: pd.DataFrame, *, source_path: str | Path) -> None:
    source = Path(source_path)
    if len(examples.columns) == 0:
        raise ValueError(f"Examples CSV has no readable columns: {source}")
    missing = sorted(set(REQUIRED_EXAMPLE_COLUMNS) - set(examples.columns))
    if missing:
        raise ValueError(f"Examples CSV is missing required columns: {missing}. File: {source}")
    if examples.empty:
        raise ValueError(f"Examples CSV is empty: {source}")

    validate_example_contract_versions(examples, source_path=source)

    for column in REQUIRED_EXAMPLE_COLUMNS:
        for index, value in examples[column].items():
            if _is_missing_required_value(value):
                raise ValueError(f"Missing required value in column '{column}' at row {index}. File: {source}")

    # Outcome validation is deliberately independent per training row: a
    # scenario can legitimately appear in multiple replay iterations.
    validate_example_outcome_contracts(examples, source_path=source)

    state_ids = examples["state_id"].map(lambda value: str(value).strip())
    duplicated = state_ids[state_ids.duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Duplicate state_id '{duplicated.iloc[0]}' in examples CSV. File: {source}")

    expected: _GraphDimensions | None = None
    for index, row in examples.iterrows():
        scenario_id = _require_integer(row["scenario_id"], column="scenario_id", index=index, source=source)
        step = _require_integer(row["step"], column="step", index=index, source=source)
        if step < 0:
            raise ValueError(f"step must be >= 0 at row {index}. File: {source}")
        _ = scenario_id
        policy = _parse_policy(row["mcts_policy_json"], index=index, source=source)
        state_path = Path(str(row["state_path"]).strip())
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}. File: {source}")
        if not state_path.is_file():
            raise ValueError(f"State path is not a file: {state_path}. File: {source}")
        dims, action_mask = _validate_npz_state(state_path)
        if expected is None:
            expected = dims
        elif dims != expected:
            raise ValueError(f"Graph dimensions mismatch for {state_path}. Expected {expected._asdict()}, observed {dims._asdict()}.")
        _validate_policy_against_mask(policy, action_mask=action_mask, index=index, source=source)
        if "selected_action_id" in examples.columns and not _is_missing_required_value(row["selected_action_id"]):
            selected = _require_integer(row["selected_action_id"], column="selected_action_id", index=index, source=source)
            # The selected action is validated only against the environment action mask.
            # A continuation/safety gate may execute an action absent from the MCTS
            # policy target, so policy-support membership is intentionally not checked.
            if selected < 0 or selected >= len(action_mask) or not bool(action_mask[selected]):
                raise ValueError(f"selected_action_id {selected} is invalid for action_mask at row {index}. File: {source}")


def validate_example_contract_versions(
    examples: pd.DataFrame,
    *,
    source_path: str | Path,
) -> None:
    source = Path(source_path)
    for index, row in examples.iterrows():
        require_exact_contract_version(
            row.get("physical_objective_schema_version"),
            expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
            name="physical-objective contract",
            source=f"{source} row {index}",
            regeneration_command=(
                "python -m scripts" ".self_play.generate ..."
            ),
        )
        require_exact_contract_version(
            row.get("outcome_value_target_contract_version"),
            expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
            name="outcome/value-target contract",
            source=f"{source} row {index}",
            regeneration_command=(
                "python -m scripts" ".self_play.generate ..."
            ),
        )


def validate_example_outcome_contracts(
    examples: pd.DataFrame, *, source_path: str | Path
) -> None:
    """Validate the strict, terminal outcome contract of each training row."""
    source = Path(source_path)
    missing = sorted(set(REQUIRED_OUTCOME_COLUMNS) - set(examples.columns))
    if missing:
        raise ValueError(
            f"Examples CSV is missing required outcome columns: {missing}. File: {source}"
        )
    for index, row in examples.iterrows():
        for column in REQUIRED_OUTCOME_COLUMNS:
            if _is_missing_required_value(row[column]):
                raise ValueError(
                    f"Missing required outcome value in column '{column}' "
                    f"at row {index}. File: {source}"
                )
        require_exact_contract_version(
            row.get("outcome_value_target_contract_version"),
            expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
            name="outcome/value-target contract",
            source=f"{source} row {index}",
            regeneration_command=("python -m scripts" ".self_play.generate ..."),
        )
        _validate_outcome_contract(row, index=index, source=source)


def _is_missing_required_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _require_integer(value: Any, *, column: str, index: Any, source: Path) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{column} must be finite integer-valued at row {index}. File: {source}")
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)) or not float(number).is_integer():
        raise ValueError(f"{column} must be finite integer-valued at row {index}. File: {source}")
    return int(number)


def _require_bool(value: Any, *, column: str, index: Any, source: Path) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(
        f"{column} must be boolean at row {index}. File: {source}"
    )

def _require_finite_number(
    value: Any,
    *,
    column: str,
    index: Any,
    source: Path,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{column} must be finite numeric at row {index}. File: {source}"
        )
    number = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(number) or not math.isfinite(float(number)):
        raise ValueError(
            f"{column} must be finite numeric at row {index}. File: {source}"
        )

    return float(number)

def _validate_outcome_contract(
    row: pd.Series,
    *,
    index: Any,
    source: Path,
) -> None:
    solved = _require_bool(
        row["solved"],
        column="solved",
        index=index,
        source=source,
    )

    done = _require_bool(
        row["done"],
        column="done",
        index=index,
        source=source,
    )

    if not done:
        raise ValueError(
            f"Training example must carry a terminal episode outcome "
            f"at row {index}. File: {source}"
        )

    reason = parse_termination_reason(
        row["termination_reason"],
        allow_none=False,
    )
    validate_outcome_invariants(
        solved=solved,
        termination_reason=reason,
    )

    steps_to_terminal = _require_integer(
        row["outcome_steps_to_terminal"],
        column="outcome_steps_to_terminal",
        index=index,
        source=source,
    )

    if steps_to_terminal <= 0:
        raise ValueError(
            f"outcome_steps_to_terminal must be > 0 at row {index}. "
            f"File: {source}"
        )

    gamma = _require_finite_number(
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

    actual_target = _require_finite_number(
        row["outcome_value_target"],
        column="outcome_value_target",
        index=index,
        source=source,
    )

    if abs(actual_target) > 1.0 + 1e-6:
        raise ValueError(
            f"outcome_value_target outside [-1, 1] at row {index}. "
            f"File: {source}"
        )

    terminal_value, expected_class = terminal_value_from_outcome(
        solved=solved,
        termination_reason=reason,
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
            f"outcome_value_target contradicts the terminal outcome at "
            f"row {index}: expected {expected_target}, "
            f"observed {actual_target}. File: {source}"
        )

    actual_class = str(row["outcome_class"]).strip()

    if actual_class != expected_class:
        raise ValueError(
            f"outcome_class contradicts the terminal outcome at row {index}: "
            f"expected {expected_class!r}, observed {actual_class!r}. "
            f"File: {source}"
        )

    mode = str(row["outcome_value_target_mode"]).strip()

    if mode != "alphazero_discounted":
        raise ValueError(
            f"Unsupported outcome_value_target_mode {mode!r} at row {index}. "
            f"File: {source}"
        )

def _validate_outcome(value: Any, *, index: Any, source: Path) -> None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        raise ValueError(f"outcome_value_target must be finite numeric at row {index}. File: {source}")
    if abs(float(number)) > 1.0 + 1e-6:
        raise ValueError(f"outcome_value_target outside [-1, 1] at row {index}. File: {source}")


def _parse_policy(value: Any, *, index: Any, source: Path) -> dict[int, float]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid mcts_policy_json at row {index}. File: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"mcts_policy_json must be an object at row {index}. File: {source}")
    if not raw:
        raise ValueError(f"mcts_policy_json must not be empty at row {index}. File: {source}")
    policy: dict[int, float] = {}
    total = 0.0
    for key, probability in raw.items():
        try:
            action_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Policy action ID must be an integer at row {index}. File: {source}") from exc
        if action_id < 0:
            raise ValueError(f"Policy action ID must be >= 0 at row {index}. File: {source}")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"Policy probability must be numeric at row {index}. File: {source}")
        probability_float = float(probability)
        if not math.isfinite(probability_float):
            raise ValueError(f"Policy probability must be finite at row {index}. File: {source}")
        if probability_float < 0.0:
            raise ValueError(f"Policy probability must be >= 0 at row {index}. File: {source}")
        policy[action_id] = probability_float
        total += probability_float
    if total <= 0.0:
        raise ValueError(f"Policy probability mass must be > 0 at row {index}. File: {source}")
    return policy


def _validate_npz_state(state_path: Path) -> tuple[_GraphDimensions, np.ndarray]:
    try:
        with np.load(state_path, allow_pickle=False) as data:
            missing = [name for name in _REQUIRED_STATE_ARRAYS if name not in data.files]
            if missing:
                raise ValueError(f"State NPZ is missing required arrays {missing}: {state_path}")
            bus_features = np.asarray(data["bus_features"])
            branch_features = np.asarray(data["branch_features"])
            edge_index = np.asarray(data["edge_index"])
            action_mask = np.asarray(data["action_mask"], dtype=bool)
    except (OSError, ValueError, EOFError) as exc:
        if isinstance(exc, ValueError) and "missing required arrays" in str(exc):
            raise
        raise ValueError(f"Could not read NPZ state: {state_path}") from exc

    if bus_features.ndim != 2 or bus_features.shape[0] <= 0:
        raise ValueError(f"{state_path}: bus_features must be non-empty 2D, got {bus_features.shape}")
    if branch_features.ndim != 2 or branch_features.shape[0] <= 0:
        raise ValueError(f"{state_path}: branch_features must be non-empty 2D, got {branch_features.shape}")
    if edge_index.shape != (2, branch_features.shape[0]):
        raise ValueError(f"{state_path}: edge_index must have shape (2, num_branches), got {edge_index.shape}")
    if action_mask.ndim != 1 or action_mask.shape[0] != branch_features.shape[0] + 1 or action_mask.shape[0] <= 0:
        raise ValueError(f"{state_path}: action_mask must be 1D with num_branches + 1 entries, got {action_mask.shape}")
    if not bool(action_mask.any()):
        raise ValueError(f"{state_path}: action_mask must contain at least one valid action")
    if not np.isfinite(bus_features).all() or not np.isfinite(branch_features).all() or not np.isfinite(edge_index).all():
        raise ValueError(f"{state_path}: graph arrays must contain only finite values")
    if not np.equal(edge_index, np.rint(edge_index)).all():
        raise ValueError(f"{state_path}: edge_index must be integer-valued")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(bus_features.shape[0]):
        raise ValueError(f"{state_path}: edge_index values out of bounds")
    return (
        _GraphDimensions(
            num_buses=int(bus_features.shape[0]),
            num_branches=int(branch_features.shape[0]),
            num_bus_features=int(bus_features.shape[1]),
            num_branch_features=int(branch_features.shape[1]),
            num_actions=int(action_mask.shape[0]),
        ),
        action_mask,
    )


def _validate_policy_against_mask(policy: dict[int, float], *, action_mask: np.ndarray, index: Any, source: Path) -> None:
    masked_mass = 0.0
    for action_id, probability in policy.items():
        if action_id >= len(action_mask):
            raise ValueError(f"Policy action ID {action_id} is out of range at row {index}. File: {source}")
        if probability > 0.0 and not bool(action_mask[action_id]):
            raise ValueError(f"Policy action ID {action_id} is masked at row {index}. File: {source}")
        if bool(action_mask[action_id]):
            masked_mass += probability
    if masked_mass <= 0.0:
        raise ValueError(f"Policy probability mass after action_mask must be > 0 at row {index}. File: {source}")

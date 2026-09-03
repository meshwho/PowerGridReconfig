from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.physics.objective import TerminalOutcomeEvidence
from grid_topology_ai.value_targets import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
    teacher_outcome_from_row,
    terminal_evidence_from_row,
    topology_utility_from_evidence,
)
from grid_topology_ai.state import validate_state_npz_schema_arrays
from grid_topology_ai.actions import (
    ActionSlot,
    ActionSpaceConfig,
    action_layout_fingerprint,
    action_layout_from_value,
    build_branch_action_slots,
    require_branch_status_policy_layout,
)


_BASE_OUTCOME_COLUMNS: tuple[str, ...] = (
    "outcome_value_target",
    "solved",
    "done",
    "teacher_outcome",
    "outcome_class",
    "outcome_steps_to_terminal",
    "outcome_value_target_mode",
    "outcome_gamma",
)
_BASE_EXAMPLE_COLUMNS: tuple[str, ...] = (
    "state_path",
    "mcts_policy_json",
    "scenario_id",
    "step",
    "state_id",
    "physics_config",
    "topology_action_config",
    "action_layout",
    "action_layout_fingerprint",
) + _BASE_OUTCOME_COLUMNS
_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "terminal_outcome_evidence_json",
)
_IDENTITY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "iteration",
    "episode_id",
)

REQUIRED_OUTCOME_COLUMNS: tuple[str, ...] = (
    _BASE_OUTCOME_COLUMNS + _EVIDENCE_COLUMNS
)
REQUIRED_EXAMPLE_COLUMNS: tuple[str, ...] = (
    _BASE_EXAMPLE_COLUMNS + _EVIDENCE_COLUMNS + _IDENTITY_COLUMNS
)

_REQUIRED_STATE_ARRAYS = (
    "bus_features",
    "branch_features",
    "edge_index",
    "action_mask",
    "branch_ids",
)
_REQUIRED_STATE_METADATA = (
    "physics_config",
    "topology_action_config",
    "action_layout",
    "run_id",
    "iteration",
    "episode_id",
    "terminal_outcome_evidence",
)


class _GraphDimensions(NamedTuple):
    num_buses: int
    num_branches: int
    num_bus_features: int
    num_branch_features: int
    num_actions: int


def _json_value(value: object, *, name: str, source: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {name} JSON for {source}.") from exc


def _physics_config_from_value(
    value: object,
    *,
    source: str,
) -> PhysicsConfig:
    raw = _json_value(value, name="physics_config", source=source)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"physics_config must be an object for {source}."
        )
    try:
        return PhysicsConfig.from_mapping(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid physics_config for {source}: {exc}"
        ) from exc


def _action_config_from_value(
    value: object,
    *,
    source: str,
) -> ActionSpaceConfig:
    raw = _json_value(
        value,
        name="topology_action_config",
        source=source,
    )
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"topology_action_config must be an object for {source}."
        )
    try:
        return ActionSpaceConfig.from_contract_mapping(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid topology_action_config for {source}: {exc}"
        ) from exc


def _action_layout_from_value(
    value: object,
    *,
    source: str,
) -> tuple[ActionSlot, ...]:
    try:
        layout = action_layout_from_value(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid action_layout for {source}: {exc}"
        ) from exc
    require_branch_status_policy_layout(layout)
    return layout


def _same_action_config(
    left: ActionSpaceConfig,
    right: ActionSpaceConfig,
) -> bool:
    return left.to_contract_dict() == right.to_contract_dict()


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


def validate_example_topology_action_contracts(
    examples: pd.DataFrame,
    *,
    source_path: str | Path,
) -> tuple[ActionSpaceConfig, tuple[ActionSlot, ...]]:
    """Read the current action config/layout without provenance hashes."""

    source = Path(source_path)
    observed_config: ActionSpaceConfig | None = None
    representative_layout: tuple[ActionSlot, ...] | None = None
    observed_policy_layout: str | None = None

    for index, row in examples.iterrows():
        row_source = f"{source} row {index}"
        row_config = _action_config_from_value(
            row["topology_action_config"],
            source=row_source,
        )
        row_layout = _action_layout_from_value(
            row["action_layout"],
            source=row_source,
        )
        row_fingerprint = str(row["action_layout_fingerprint"]).strip()
        expected_fingerprint = action_layout_fingerprint(row_layout)
        if row_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Action layout identity mismatch for {row_source}."
            )

        if observed_config is None:
            observed_config = row_config
        elif not _same_action_config(observed_config, row_config):
            raise ValueError(
                f"Topology action config mismatch for {row_source}."
            )

        if representative_layout is None:
            representative_layout = row_layout

        row_policy_layout = require_branch_status_policy_layout(row_layout)
        if observed_policy_layout is None:
            observed_policy_layout = row_policy_layout
        elif row_policy_layout != observed_policy_layout:
            raise ValueError(
                "Examples contain incompatible policy layouts. "
                f"File: {source}"
            )

    if observed_config is None or representative_layout is None:
        raise ValueError(f"Examples CSV is empty: {source}")
    return observed_config, representative_layout


def validate_example_contract_versions(
    examples: pd.DataFrame,
    *,
    source_path: str | Path,
    expected_physics_config: PhysicsConfig | None = None,
) -> PhysicsConfig:
    """Return the single semantic PhysicsConfig used by the examples."""

    source = Path(source_path)
    observed: PhysicsConfig | None = None
    for index, row in examples.iterrows():
        row_config = _physics_config_from_value(
            row["physics_config"],
            source=f"{source} row {index}",
        )
        expected = expected_physics_config or observed
        if expected is not None and row_config != expected:
            raise ValueError(
                f"PhysicsConfig mismatch for {source} row {index}."
            )
        if observed is None:
            observed = row_config

    if observed is None:
        raise ValueError(f"Examples CSV is empty: {source}")
    return observed


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

    for column in REQUIRED_EXAMPLE_COLUMNS:
        for index, value in examples[column].items():
            if _is_missing_required_value(value):
                raise ValueError(
                    f"Missing required value in column '{column}' "
                    f"at row {index}. File: {source}"
                )

    physics_config = validate_example_contract_versions(
        examples,
        source_path=source,
    )
    action_space_config, _ = validate_example_topology_action_contracts(
        examples,
        source_path=source,
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

    expected_feature_dimensions: tuple[int, int] | None = None
    for index, row in examples.iterrows():
        _require_integer(
            row["scenario_id"],
            column="scenario_id",
            index=index,
            source=source,
        )
        step = _require_integer(
            row["step"],
            column="step",
            index=index,
            source=source,
        )
        if step < 0:
            raise ValueError(
                f"step must be >= 0 at row {index}. File: {source}"
            )

        policy = _parse_policy(
            row["mcts_policy_json"],
            index=index,
            source=source,
        )
        state_path = Path(str(row["state_path"]).strip())

        row_layout = _action_layout_from_value(
            row["action_layout"],
            source=f"{source} row {index}",
        )
        dimensions, action_mask = _validate_npz_state(
            state_path,
            expected_physics_config=physics_config,
            expected_action_space_config=action_space_config,
            expected_action_layout=row_layout,
        )
        feature_dimensions = (
            dimensions.num_bus_features,
            dimensions.num_branch_features,
        )
        if expected_feature_dimensions is None:
            expected_feature_dimensions = feature_dimensions
        elif feature_dimensions != expected_feature_dimensions:
            raise ValueError(
                "Graph feature dimensions mismatch for "
                f"{state_path}. Expected {expected_feature_dimensions}, "
                f"observed {feature_dimensions}."
            )

        _validate_policy_against_mask(
            policy,
            action_mask=action_mask,
            index=index,
            source=source,
        )

        if (
            "selected_action_id" in examples.columns
            and not _is_missing_required_value(row["selected_action_id"])
        ):
            selected = _require_integer(
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


def policy_vector_from_json(
    value: object,
    *,
    action_mask: np.ndarray,
    index: Any,
    source_path: str | Path,
) -> np.ndarray:
    source = Path(source_path)
    mask = np.asarray(action_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(
            f"action_mask must be 1D at row {index}. File: {source}"
        )

    policy = _parse_policy(
        value,
        index=index,
        source=source,
    )
    _validate_policy_against_mask(
        policy,
        action_mask=mask,
        index=index,
        source=source,
    )

    vector = np.zeros(mask.shape[0], dtype=np.float32)
    for action_id, probability in policy.items():
        vector[action_id] = probability
    return vector


def _validate_policy_against_mask(
    policy: dict[int, float],
    *,
    action_mask: np.ndarray,
    index: Any,
    source: Path,
) -> None:
    masked_mass = 0.0
    for action_id, probability in policy.items():
        if action_id >= len(action_mask):
            raise ValueError(
                f"Policy action ID {action_id} is out of range at row "
                f"{index}. File: {source}"
            )
        if probability > 0.0 and not bool(action_mask[action_id]):
            raise ValueError(
                f"Policy action ID {action_id} is masked at row {index}. "
                f"File: {source}"
            )
        if bool(action_mask[action_id]):
            masked_mass += probability

    if masked_mass <= 0.0:
        raise ValueError(
            f"Policy probability mass after action_mask must be > 0 at row "
            f"{index}. File: {source}"
        )
    if not math.isclose(
        masked_mass,
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Policy probability mass after action_mask must equal 1.0 "
            f"at row {index}; observed {masked_mass:.9g}. File: {source}"
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
        iteration = _require_integer(
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
            _require_integer(
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


def _validate_episode_outcome_consistency(
    examples: pd.DataFrame,
    *,
    source: Path,
) -> None:
    if "episode_id" not in examples.columns:
        return

    for episode_id, group in examples.groupby(
        "episode_id",
        sort=False,
    ):
        for column in ("run_id", "iteration", "scenario_id"):
            if (
                column in group.columns
                and group[column].nunique(dropna=False) != 1
            ):
                raise ValueError(
                    f"Episode {episode_id!r} contains mixed {column} "
                    f"values. File: {source}"
                )

        expected: tuple[object, ...] | None = None
        for index, row in group.iterrows():
            evidence = terminal_evidence_from_row(
                row.to_dict(),
                context=f"{source} row {index}",
            )
            signature = (
                evidence,
                _require_finite_number(
                    row["outcome_value_target"],
                    column="outcome_value_target",
                    index=index,
                    source=source,
                ),
                str(row["outcome_class"]).strip(),
                str(row["outcome_value_target_mode"]).strip(),
                _require_finite_number(
                    row["outcome_gamma"],
                    column="outcome_gamma",
                    index=index,
                    source=source,
                ),
            )
            if expected is None:
                expected = signature
            elif signature != expected:
                raise ValueError(
                    f"Episode {episode_id!r} contains mixed terminal "
                    f"outcomes or evidence. File: {source}"
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
            if _is_missing_required_value(row[column]):
                raise ValueError(
                    "Missing required outcome value in column "
                    f"'{column}' at row {index}. File: {source}"
                )
        _validate_outcome(
            row,
            index=index,
            source=source,
        )

    _validate_episode_outcome_consistency(
        examples,
        source=source,
    )


def _validate_outcome(
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
            "Training example must carry a terminal episode outcome "
            f"at row {index}. File: {source}"
        )

    row_mapping = row.to_dict()
    evidence = terminal_evidence_from_row(
        row_mapping,
        context=f"{source} row {index}",
    )
    if evidence.solved is not solved:
        raise ValueError(
            "terminal outcome evidence contradicts solved at row "
            f"{index}. File: {source}"
        )

    steps_to_terminal = _require_integer(
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

    gamma = _require_finite_number(
        row["outcome_gamma"],
        column="outcome_gamma",
        index=index,
        source=source,
    )
    if gamma != TERMINAL_UTILITY_GAMMA:
        raise ValueError(
            f"outcome_gamma must be exactly 1.0 at row {index}. "
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
            "outcome_value_target outside [-1, 1] at row "
            f"{index}. File: {source}"
        )

    terminal_value = topology_utility_from_evidence(evidence)
    expected_class = teacher_outcome_from_row(
        row_mapping,
        context=f"{source} row {index}",
    ).value
    if not math.isclose(
        actual_target,
        terminal_value,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise ValueError(
            "outcome_value_target contradicts the terminal outcome "
            f"at row {index}: expected {terminal_value}, "
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
    if mode != VALUE_TARGET_MODE:
        raise ValueError(
            f"Unsupported outcome_value_target_mode {mode!r} at "
            f"row {index}. File: {source}"
        )


def _is_missing_required_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _require_integer(
    value: Any,
    *,
    column: str,
    index: Any,
    source: Path,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{column} must be finite integer-valued at row {index}. "
            f"File: {source}"
        )
    number = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]
    if (
        pd.isna(number)
        or not math.isfinite(float(number))
        or not float(number).is_integer()
    ):
        raise ValueError(
            f"{column} must be finite integer-valued at row {index}. "
            f"File: {source}"
        )
    return int(number)


def _require_bool(
    value: Any,
    *,
    column: str,
    index: Any,
    source: Path,
) -> bool:
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
            f"{column} must be finite numeric at row {index}. "
            f"File: {source}"
        )
    number = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        raise ValueError(
            f"{column} must be finite numeric at row {index}. "
            f"File: {source}"
        )
    return float(number)


def _parse_policy(
    value: Any,
    *,
    index: Any,
    source: Path,
) -> dict[int, float]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid mcts_policy_json at row {index}. File: {source}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"mcts_policy_json must be an object at row {index}. "
            f"File: {source}"
        )
    if not raw:
        raise ValueError(
            f"mcts_policy_json must not be empty at row {index}. "
            f"File: {source}"
        )

    policy: dict[int, float] = {}
    total = 0.0
    for key, probability in raw.items():
        try:
            action_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Policy action ID must be an integer at row {index}. "
                f"File: {source}"
            ) from exc
        if action_id < 0:
            raise ValueError(
                f"Policy action ID must be >= 0 at row {index}. "
                f"File: {source}"
            )
        if isinstance(probability, bool) or not isinstance(
            probability,
            (int, float),
        ):
            raise ValueError(
                f"Policy probability must be numeric at row {index}. "
                f"File: {source}"
            )
        probability_float = float(probability)
        if not math.isfinite(probability_float):
            raise ValueError(
                f"Policy probability must be finite at row {index}. "
                f"File: {source}"
            )
        if probability_float < 0.0:
            raise ValueError(
                f"Policy probability must be >= 0 at row {index}. "
                f"File: {source}"
            )
        policy[action_id] = probability_float
        total += probability_float

    if total <= 0.0:
        raise ValueError(
            f"Policy probability mass must be > 0 at row {index}. "
            f"File: {source}"
        )
    return policy


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
    missing = sorted(set(_REQUIRED_STATE_METADATA) - set(metadata))
    if missing:
        raise ValueError(
            f"State metadata is missing required fields {missing}: {state_path}"
        )
    return metadata


def validate_state_physics_config_payload(
    state_path: str | Path,
    *,
    expected_physics_config: PhysicsConfig,
) -> None:
    """Validate current physics semantics stored with one state."""

    state_path = Path(state_path)
    metadata = _load_state_metadata(state_path)
    observed = _physics_config_from_value(
        metadata["physics_config"],
        source=str(state_path),
    )
    if observed != expected_physics_config:
        raise ValueError(
            f"PhysicsConfig mismatch for {state_path}."
        )
    validate_state_npz_schema_arrays(state_path)


def validate_state_topology_action_payload(
    state_path: str | Path,
    *,
    expected_action_space_config: ActionSpaceConfig,
    expected_action_layout: tuple[ActionSlot, ...],
) -> tuple[ActionSpaceConfig, tuple[ActionSlot, ...]]:
    """Validate current action semantics stored with one state."""

    state_path = Path(state_path)
    metadata = _load_state_metadata(state_path)
    observed_config = _action_config_from_value(
        metadata["topology_action_config"],
        source=str(state_path),
    )
    observed_layout = _action_layout_from_value(
        metadata["action_layout"],
        source=str(state_path),
    )
    if not _same_action_config(
        observed_config,
        expected_action_space_config,
    ):
        raise ValueError(
            f"Topology action config mismatch for {state_path}."
        )
    if observed_layout != expected_action_layout:
        raise ValueError(
            f"Action layout mismatch for {state_path}."
        )

    try:
        with np.load(state_path, allow_pickle=False) as data:
            if "branch_ids" not in data.files:
                raise ValueError(
                    f"State NPZ is missing branch_ids: {state_path}"
                )
            branch_ids = np.asarray(data["branch_ids"])
    except (OSError, EOFError) as exc:
        raise ValueError(
            f"Could not read NPZ state: {state_path}"
        ) from exc

    if branch_ids.ndim != 1 or branch_ids.size == 0:
        raise ValueError(
            f"{state_path}: branch_ids must be non-empty 1D, "
            f"got {branch_ids.shape}"
        )
    try:
        branch_values = np.asarray(branch_ids, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{state_path}: branch_ids must be numeric"
        ) from exc
    if not np.isfinite(branch_values).all():
        raise ValueError(
            f"{state_path}: branch_ids must be finite"
        )
    if not np.equal(branch_values, np.rint(branch_values)).all():
        raise ValueError(
            f"{state_path}: branch_ids must be integer-valued"
        )
    if bool((branch_values < 0).any()):
        raise ValueError(
            f"{state_path}: branch_ids must be non-negative"
        )

    branch_layout = build_branch_action_slots(
        int(branch_id)
        for branch_id in branch_values.tolist()
    )
    if branch_layout != observed_layout:
        raise ValueError(
            f"{state_path}: branch_ids do not match action_layout metadata"
        )
    return observed_config, observed_layout


def _validate_npz_state(
    state_path: Path,
    *,
    expected_physics_config: PhysicsConfig,
    expected_action_space_config: ActionSpaceConfig,
    expected_action_layout: tuple[ActionSlot, ...],
) -> tuple[_GraphDimensions, np.ndarray]:
    try:
        with np.load(state_path, allow_pickle=False) as data:
            missing = [
                name
                for name in _REQUIRED_STATE_ARRAYS
                if name not in data.files
            ]
            if missing:
                raise ValueError(
                    f"State NPZ is missing required arrays {missing}: "
                    f"{state_path}"
                )
            bus_features = np.asarray(data["bus_features"])
            branch_features = np.asarray(data["branch_features"])
            edge_index = np.asarray(data["edge_index"])
            action_mask = np.asarray(
                data["action_mask"],
                dtype=bool,
            )
    except (OSError, ValueError, EOFError) as exc:
        if (
            isinstance(exc, ValueError)
            and "missing required arrays" in str(exc)
        ):
            raise
        raise ValueError(
            f"Could not read NPZ state: {state_path}"
        ) from exc

    validate_state_physics_config_payload(
        state_path,
        expected_physics_config=expected_physics_config,
    )
    _, state_action_layout = validate_state_topology_action_payload(
        state_path,
        expected_action_space_config=expected_action_space_config,
        expected_action_layout=expected_action_layout,
    )

    if bus_features.ndim != 2 or bus_features.shape[0] <= 0:
        raise ValueError(
            f"{state_path}: bus_features must be non-empty 2D, "
            f"got {bus_features.shape}"
        )
    if branch_features.ndim != 2 or branch_features.shape[0] <= 0:
        raise ValueError(
            f"{state_path}: branch_features must be non-empty 2D, "
            f"got {branch_features.shape}"
        )
    if edge_index.shape != (2, branch_features.shape[0]):
        raise ValueError(
            f"{state_path}: edge_index must have shape (2, num_branches), "
            f"got {edge_index.shape}"
        )

    expected_num_actions = len(state_action_layout)
    if (
        action_mask.ndim != 1
        or action_mask.shape[0] != expected_num_actions
        or action_mask.shape[0] <= 0
    ):
        raise ValueError(
            f"{state_path}: action_mask must be 1D "
            f"with {expected_num_actions} entries, "
            f"got {action_mask.shape}"
        )
    if not bool(action_mask.any()):
        raise ValueError(
            f"{state_path}: action_mask must contain at least one valid action"
        )
    if (
        not np.isfinite(bus_features).all()
        or not np.isfinite(branch_features).all()
        or not np.isfinite(edge_index).all()
    ):
        raise ValueError(
            f"{state_path}: graph arrays must contain only finite values"
        )
    if not np.equal(edge_index, np.rint(edge_index)).all():
        raise ValueError(
            f"{state_path}: edge_index must be integer-valued"
        )
    if (
        int(edge_index.min()) < 0
        or int(edge_index.max()) >= int(bus_features.shape[0])
    ):
        raise ValueError(
            f"{state_path}: edge_index values out of bounds"
        )

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
        if metadata[field] != expected:
            raise ValueError(
                f"CSV {field} does not match state metadata at row "
                f"{index}. File: {source}. State: {state_path}"
            )

    raw_evidence = metadata["terminal_outcome_evidence"]
    if not isinstance(raw_evidence, Mapping):
        raise ValueError(
            "State metadata has invalid terminal_outcome_evidence "
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

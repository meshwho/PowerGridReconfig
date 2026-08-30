from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd

from grid_topology_ai.physics.objective import (
    RedispatchStatus,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.state import GridFMState
from grid_topology_ai.physics.utility import (
    DEFAULT_STATE_UTILITY_SCALE,
    require_reward_discount_factor,
    state_utility,
)
from grid_topology_ai.termination import (
    TeacherOutcome,
    classify_teacher_outcome,
)


VALUE_TARGET_MODE = "final_topology_state_utility"
TERMINAL_UTILITY_GAMMA = 1.0
DEFAULT_HEURISTIC_UTILITY_SCALE = DEFAULT_STATE_UTILITY_SCALE
_UTILITY_TOLERANCE = 1e-7


def require_discount_factor(value: object) -> float:
    """Validate a caller gamma and return the fixed value-target gamma."""
    require_reward_discount_factor(value)
    return TERMINAL_UTILITY_GAMMA


def require_bounded_utility(value: object, *, context: str) -> float:
    """Validate and safely normalize a value-head utility to ``[-1, 1]``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite real utility, got {value!r}")
    utility = float(value)
    if not math.isfinite(utility):
        raise ValueError(f"{context} must be finite, got {value!r}")
    if utility < -1.0 - _UTILITY_TOLERANCE or utility > 1.0 + _UTILITY_TOLERANCE:
        raise ValueError(f"{context} must be in [-1, 1], got {utility!r}")
    return float(min(1.0, max(-1.0, utility)))


def topology_utility_from_evidence(
    evidence: TerminalOutcomeEvidence,
) -> float:
    """Return the stored utility of the final pre-redispatch topology state."""
    if not isinstance(evidence, TerminalOutcomeEvidence):
        raise TypeError("evidence must be TerminalOutcomeEvidence")
    return require_bounded_utility(
        evidence.topology_utility,
        context="final topology state utility",
    )


def heuristic_terminal_utility_estimate(
    state: GridFMState,
    *,
    physics_config: PhysicsConfig | None = None,
    utility_scale: float = DEFAULT_HEURISTIC_UTILITY_SCALE,
) -> float:
    """Return the canonical bounded physical-state utility as a fallback."""
    estimate = state_utility(
        state,
        physics_config=physics_config,
        utility_scale=utility_scale,
    )
    return require_bounded_utility(
        estimate,
        context="heuristic terminal utility estimate",
    )


_IDENTITY_FIELDS = (
    "run_id",
    "iteration",
    "episode_id",
    "scenario_id",
)


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{field} must be a boolean, got {value!r}"
        )
    return bool(value)


def _require_gamma(value: object) -> float:
    gamma = require_discount_factor(value)

    if gamma != TERMINAL_UTILITY_GAMMA:
        raise ValueError(
            "outcome gamma must be exactly 1.0 for "
            "undiscounted terminal utility"
        )

    return gamma


def teacher_outcome_from_row(
    row: Mapping[str, object],
    *,
    context: str,
) -> TeacherOutcome:
    value = row.get("teacher_outcome")
    if isinstance(value, TeacherOutcome):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"{context}: teacher_outcome must be a string, got {value!r}"
        )
    try:
        return TeacherOutcome(value.strip())
    except ValueError as exc:
        allowed = ", ".join(outcome.value for outcome in TeacherOutcome)
        raise ValueError(
            f"{context}: unknown teacher_outcome {value!r}; expected one of: {allowed}"
        ) from exc


def terminal_evidence_from_row(
    row: Mapping[str, object],
    *,
    context: str,
) -> TerminalOutcomeEvidence:
    """Parse and validate terminal evidence stored on one example row."""

    solved = _require_bool(
        row.get("solved"),
        field=f"{context}.solved",
    )
    outcome = teacher_outcome_from_row(row, context=context)

    try:
        evidence = TerminalOutcomeEvidence.from_json(
            row.get("terminal_outcome_evidence_json")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: invalid terminal outcome evidence: {exc}"
        ) from exc

    if evidence.solved is not solved:
        raise ValueError(
            f"{context}: terminal outcome evidence contradicts solved."
        )

    expected_outcome = classify_teacher_outcome(
        topology_solved=evidence.solved,
        redispatch_validated=(
            evidence.redispatch_status is RedispatchStatus.VALIDATED
        ),
    )
    if outcome is not expected_outcome:
        raise ValueError(
            f"{context}: teacher_outcome={outcome.value!r} contradicts "
            f"terminal evidence; expected {expected_outcome.value!r}."
        )

    return evidence


def _require_group_key(
    row: Mapping[str, object],
    key: str,
) -> object:
    if key not in row:
        raise ValueError(f"Missing required group key {key!r}")
    value = row[key]
    if value is None or (
        isinstance(value, str)
        and not value.strip()
    ):
        raise ValueError(f"Invalid group key {key!r}: {value!r}")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid group key {key!r}: {value!r}")

    missing = pd.isna(value)
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError(
            f"Group key {key!r} must be a hashable scalar, "
            f"got {value!r}"
        )
    if bool(missing) or (
        isinstance(value, Real)
        and not math.isfinite(float(value))
    ):
        raise ValueError(f"Invalid group key {key!r}: {value!r}")

    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(
            f"Group key {key!r} must be hashable, got {value!r}"
        ) from exc

    if key in {"run_id", "episode_id"}:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{key} must be a non-empty string, got {value!r}"
            )
        return value.strip()

    if key == "iteration" and (
        not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(
            "iteration must be a positive integer, "
            f"got {value!r}"
        )

    if key == "scenario_id" and (
        not isinstance(value, Integral)
        or int(value) < 0
    ):
        raise ValueError(
            "scenario_id must be a non-negative integer, "
            f"got {value!r}"
        )
    return value


def _require_step(row: Mapping[str, object]) -> int:
    if "step" not in row:
        raise ValueError("Missing required step")
    value = row["step"]
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            f"step must be a non-negative integer, got {value!r}"
        )
    step = int(value)
    if step < 0:
        raise ValueError(
            f"step must be a non-negative integer, got {value!r}"
        )
    return step


def add_outcome_value_targets_to_rows(
    rows: list[dict[str, object]],
    gamma: float,
    group_keys: tuple[str, ...] = ("episode_id",),
) -> None:
    """Atomically derive undiscounted terminal-utility targets."""

    normalized_gamma = _require_gamma(gamma)
    if (
        not isinstance(group_keys, tuple)
        or not group_keys
        or any(
            not isinstance(key, str)
            or not key.strip()
            for key in group_keys
        )
        or len(set(group_keys)) != len(group_keys)
    ):
        raise ValueError(
            "group_keys must be a non-empty tuple of unique field names"
        )

    if (
        group_keys == ("scenario_id",)
        and rows
        and all("episode_id" in row for row in rows)
    ):
        group_keys = ("episode_id",)

    groups: dict[
        tuple[object, ...],
        list[tuple[int, dict[str, object]]],
    ] = {}
    for row in rows:
        for field in _IDENTITY_FIELDS:
            _require_group_key(row, field)

        key = tuple(
            _require_group_key(row, name)
            for name in group_keys
        )
        groups.setdefault(key, []).append(
            (_require_step(row), row)
        )

    pending_updates: list[
        tuple[dict[str, object], dict[str, object]]
    ] = []

    for key, indexed_rows in groups.items():
        steps = [step for step, _ in indexed_rows]
        if len(steps) != len(set(steps)):
            raise ValueError(
                f"Duplicate step in episode group {key!r}"
            )
        indexed_rows.sort(key=lambda item: item[0])
        if steps and sorted(steps) != list(range(len(steps))):
            raise ValueError(
                f"Episode group {key!r} must use contiguous steps from 0"
            )

        for field in _IDENTITY_FIELDS:
            values = {
                _require_group_key(row, field)
                for _, row in indexed_rows
            }
            if len(values) != 1:
                raise ValueError(
                    f"Mixed {field} values in episode group {key!r}"
                )

        expected_solved: bool | None = None
        expected_outcome: TeacherOutcome | None = None
        expected_evidence: TerminalOutcomeEvidence | None = None

        for _, row in indexed_rows:
            done = _require_bool(
                row.get("done"),
                field="done",
            )
            if not done:
                raise ValueError(
                    "done must be True; cannot derive outcome target "
                    "from an unfinished episode."
                )

            evidence = terminal_evidence_from_row(
                row,
                context=f"episode group {key!r}",
            )
            solved = evidence.solved
            outcome = teacher_outcome_from_row(
                row,
                context=f"episode group {key!r}",
            )

            if expected_solved is None:
                expected_solved = solved
                expected_outcome = outcome
                expected_evidence = evidence
            elif (
                solved != expected_solved
                or outcome is not expected_outcome
                or evidence != expected_evidence
            ):
                raise ValueError(
                    "Cannot derive targets from mixed terminal evidence "
                    f"in episode group {key!r}"
                )

        if (
            expected_solved is None
            or expected_outcome is None
            or expected_evidence is None
        ):
            raise RuntimeError(
                f"Episode group {key!r} unexpectedly has no rows"
            )

        terminal_utility = topology_utility_from_evidence(expected_evidence)
        outcome_class = expected_outcome.value
        total = len(indexed_rows)

        for position, (_, row) in enumerate(indexed_rows):
            steps_to_terminal = total - position
            pending_updates.append(
                (
                    row,
                    {
                        "outcome_value_target": terminal_utility,
                        "outcome_class": outcome_class,
                        "outcome_steps_to_terminal": steps_to_terminal,
                        "outcome_value_target_mode": VALUE_TARGET_MODE,
                        "outcome_gamma": normalized_gamma,
                    },
                )
            )

    for row, updates in pending_updates:
        row.update(updates)

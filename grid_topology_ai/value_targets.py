from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd

from grid_topology_ai.physics.objective import (
    TerminalOutcomeEvidence,
    parse_terminal_outcome_fields,
)
from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.state import GridFMState
from grid_topology_ai.physics.utility import (
    DEFAULT_STATE_UTILITY_SCALE,
    state_utility,
)
from grid_topology_ai.termination import (
    TerminationReason,
    validate_outcome_invariants,
)


VALUE_TARGET_MODE = "final_topology_state_utility"
TERMINAL_UTILITY_GAMMA = 1.0
DEFAULT_HEURISTIC_UTILITY_SCALE = DEFAULT_STATE_UTILITY_SCALE
_UTILITY_TOLERANCE = 1e-7


def require_discount_factor(value: object) -> float:
    """Validate a caller gamma and return the fixed value-target gamma."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"gamma must be a finite real number in [0, 1], got {value!r}"
        )
    gamma = float(value)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(
            f"gamma must be a finite real number in [0, 1], got {value!r}"
        )
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


def terminal_utility_from_outcome(
    solved: bool,
    termination_reason: TerminationReason | str | None,
    *,
    evidence: TerminalOutcomeEvidence | None = None,
) -> tuple[float, str]:
    """Map one terminal episode outcome to the primary topology utility.

    Current training artifacts carry terminal evidence. Their target is the
    canonical utility of the final pre-redispatch topology state. Redispatch
    success, failure, and magnitude do not change that primary value.
    """
    if not isinstance(solved, bool):
        raise ValueError(f"solved must be a boolean, got {solved!r}")
    reason = validate_outcome_invariants(
        solved=solved,
        termination_reason=termination_reason,
    )

    if evidence is not None:
        if evidence.solved is not solved:
            raise ValueError(
                "Terminal outcome evidence contradicts solved."
            )
        if evidence.termination_reason is not reason:
            raise ValueError(
                "Terminal outcome evidence contradicts termination_reason."
            )
        return (
            topology_utility_from_evidence(evidence),
            evidence.termination_reason.value,
        )

    if reason is TerminationReason.SOLVED:
        return 1.0, TerminationReason.SOLVED.value

    if reason is TerminationReason.REDISPATCH_VALIDATED:
        raise ValueError(
            "redispatch_validated requires terminal outcome evidence."
        )

    return -1.0, "unsolved_terminal" if reason is None else reason.value


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


def terminal_evidence_from_row(
    row: Mapping[str, object],
    *,
    context: str,
) -> TerminalOutcomeEvidence:
    """Parse and validate terminal evidence stored on one example row."""

    try:
        solved, reason = parse_terminal_outcome_fields(
            solved=row.get("solved"),
            termination_reason=row.get("termination_reason"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: invalid terminal outcome: {exc}"
        ) from exc

    try:
        evidence = TerminalOutcomeEvidence.from_json(
            row.get("terminal_outcome_evidence_json")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: invalid terminal outcome evidence: {exc}"
        ) from exc

    if (
        evidence.solved is not solved
        or evidence.termination_reason is not reason
    ):
        raise ValueError(
            f"{context}: terminal outcome evidence contradicts "
            "solved or termination_reason."
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
        expected_reason: TerminationReason | None = None
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
            reason = evidence.termination_reason

            if expected_solved is None:
                expected_solved = solved
                expected_reason = reason
                expected_evidence = evidence
            elif (
                solved != expected_solved
                or reason is not expected_reason
                or evidence != expected_evidence
            ):
                raise ValueError(
                    "Cannot derive targets from mixed terminal evidence "
                    f"in episode group {key!r}"
                )

        if (
            expected_solved is None
            or expected_reason is None
            or expected_evidence is None
        ):
            raise RuntimeError(
                f"Episode group {key!r} unexpectedly has no rows"
            )

        terminal_utility, outcome_class = (
            terminal_utility_from_outcome(
                expected_solved,
                expected_reason,
                evidence=expected_evidence,
            )
        )
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

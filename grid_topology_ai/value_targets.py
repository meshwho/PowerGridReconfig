from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd

from grid_topology_ai.contracts import OUTCOME_VALUE_TARGET_CONTRACT_VERSION
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
    validate_outcome_invariants,
)


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a boolean, got {value!r}")
    return bool(value)


def _require_gamma(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"gamma must be a finite real number in [0, 1], got {value!r}")
    gamma = float(value)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be a finite real number in [0, 1], got {value!r}")
    return gamma


def terminal_value_from_outcome(
    solved: bool,
    termination_reason: TerminationReason | str | None,
) -> tuple[float, str]:
    """Convert a terminal outcome into a normalized value and class.

    Solved episodes return ``+1.0``, redispatch handoffs return ``0.0``,
    and failed terminal outcomes return ``-1.0``.  The accompanying class
    is normalized for diagnostics and outcome-contract validation.
    """
    strict_solved = _require_bool(solved, field="solved")
    reason = validate_outcome_invariants(
        solved=strict_solved,
        termination_reason=termination_reason,
    )
    if reason is TerminationReason.SOLVED:
        return 1.0, TerminationReason.SOLVED.value
    if reason in {
        TerminationReason.HANDOFF_TO_REDISPATCH,
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
    }:
        return 0.0, TerminationReason.HANDOFF_TO_REDISPATCH.value
    return -1.0, "unsolved_terminal" if reason is None else reason.value


def _require_group_key(row: Mapping[str, object], key: str) -> object:
    if key not in row:
        raise ValueError(f"Missing required group key {key!r}")
    value = row[key]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Invalid group key {key!r}: {value!r}")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid group key {key!r}: {value!r}")
    missing = pd.isna(value)
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError(f"Group key {key!r} must be a hashable scalar, got {value!r}")
    if bool(missing) or (
        isinstance(value, Real) and not math.isfinite(float(value))
    ):
        raise ValueError(f"Invalid group key {key!r}: {value!r}")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"Group key {key!r} must be hashable, got {value!r}") from exc
    if key == "scenario_id" and (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or int(value) < 0
    ):
        raise ValueError(f"scenario_id must be a non-negative integer, got {value!r}")
    return value


def _require_step(row: Mapping[str, object]) -> int:
    if "step" not in row:
        raise ValueError("Missing required step")
    value = row["step"]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"step must be a non-negative integer, got {value!r}")
    step = int(value)
    if step < 0:
        raise ValueError(f"step must be a non-negative integer, got {value!r}")
    return step


def add_outcome_value_targets_to_rows(
    rows: list[dict[str, object]],
    gamma: float,
    group_keys: tuple[str, ...] = ("scenario_id",),
) -> None:
    """Atomically derive strict discounted terminal targets for episode rows."""
    normalized_gamma = _require_gamma(gamma)
    if (
        not isinstance(group_keys, tuple)
        or not group_keys
        or any(not isinstance(key, str) or not key.strip() for key in group_keys)
        or len(set(group_keys)) != len(group_keys)
    ):
        raise ValueError("group_keys must be a non-empty tuple of unique field names")
    groups: dict[
        tuple[object, ...], list[tuple[int, dict[str, object]]]
    ] = {}
    for row in rows:
        if row.get("physical_objective_schema_version") != PHYSICAL_OBJECTIVE_SCHEMA_VERSION:
            raise ValueError("Cannot derive current outcome value targets from legacy solved labels.")
        key = tuple(_require_group_key(row, name) for name in group_keys)
        groups.setdefault(key, []).append((_require_step(row), row))

    pending_updates: list[tuple[dict[str, object], dict[str, object]]] = []
    for key, indexed_rows in groups.items():
        steps = [step for step, _ in indexed_rows]
        if len(steps) != len(set(steps)):
            raise ValueError(f"Duplicate step in episode group {key!r}")
        indexed_rows.sort(key=lambda item: item[0])
        expected_solved: bool | None = None
        expected_reason: TerminationReason | None = None
        for _, row in indexed_rows:
            solved = _require_bool(row.get("solved"), field="solved")
            done = _require_bool(row.get("done"), field="done")
            if not done:
                raise ValueError("Cannot derive outcome target from an unfinished episode.")
            try:
                reason = parse_termination_reason(
                    row.get("termination_reason"), allow_none=False
                )
                validate_outcome_invariants(
                    solved=solved, termination_reason=reason
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid terminal outcome in episode group {key!r}: {exc}") from exc
            if expected_solved is None:
                expected_solved, expected_reason = solved, reason
            elif solved != expected_solved or reason != expected_reason:
                raise ValueError(f"Cannot derive targets from mixed episode outcomes in group {key!r}")

        if expected_solved is None or expected_reason is None:
            raise RuntimeError(f"Episode group {key!r} unexpectedly has no rows")
        terminal_value, outcome_class = terminal_value_from_outcome(
            expected_solved, expected_reason
        )
        total = len(indexed_rows)
        for position, (_, row) in enumerate(indexed_rows):
            pending_updates.append(
                (
                    row,
                    {
                        "outcome_value_target": (
                            terminal_value * normalized_gamma ** (total - position)
                        ),
                        "outcome_class": outcome_class,
                        "outcome_steps_to_terminal": total - position,
                        "outcome_value_target_mode": "alphazero_discounted",
                        "outcome_gamma": normalized_gamma,
                        "outcome_value_target_contract_version": (
                            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                        ),
                    },
                )
            )
    for row, updates in pending_updates:
        row.update(updates)

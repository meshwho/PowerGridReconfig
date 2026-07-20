from __future__ import annotations

import math
from numbers import Real

from grid_topology_ai.contracts import OUTCOME_VALUE_TARGET_CONTRACT_VERSION
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
    validate_outcome_invariants,
)


def terminal_value_from_outcome(
    solved: bool,
    termination_reason: TerminationReason | str | None,
) -> tuple[float, str]:
    """
    Convert terminal episode outcome into a bounded AlphaZero-style value.

    Returns
    -------
    tuple[float, str]
        terminal_value:
            +1.0 for solved episodes
             0.0 for redispatch handoff
            -1.0 for failed / max_steps / unsafe terminal outcomes

        outcome_class:
            Normalized textual outcome class used for diagnostics.
    """

    reason = validate_outcome_invariants(
        solved=bool(solved),
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


def add_outcome_value_targets_to_rows(
    rows: list[dict],
    gamma: float,
    group_keys: tuple[str, ...] = ("scenario_id",),
) -> None:
    """
    Add strict AlphaZero-like outcome value targets to generated rows.

    Every row receives:

    - outcome_value_target
    - outcome_class
    - outcome_steps_to_terminal
    - outcome_value_target_mode
    - outcome_gamma

    The target is based only on the episode outcome already recorded on every
    row. The function rejects incomplete or inconsistent groups instead of
    rewriting their source outcome fields:

        solved  -> +1.0 * gamma^k
        handoff ->  0.0 * gamma^k
        failed  -> -1.0 * gamma^k
    """

    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, Real)
        or not math.isfinite(gamma)
    ):
        raise ValueError(f"gamma must be a finite number in [0, 1], got {gamma!r}")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    groups: dict[tuple, list[dict]] = {}

    for row in rows:
        physical_version = row.get("physical_objective_schema_version")
        if physical_version != PHYSICAL_OBJECTIVE_SCHEMA_VERSION:
            raise ValueError(
                "Cannot derive current outcome value targets from legacy "
                "solved labels. Regenerate episodes with "
                "python -m scripts.self_play.generate before computing targets. "
                f"Expected physical_objective_schema_version="
                f"{PHYSICAL_OBJECTIVE_SCHEMA_VERSION}, observed "
                f"{physical_version!r}."
            )
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    for group_key, group_rows in groups.items():
        group_rows.sort(key=lambda r: int(r.get("step", 0)))

        if not group_rows:
            continue

        terminal_solved, terminal_reason = _validate_group_outcome(
            group_rows,
            group_key=group_key,
        )
        terminal_value, outcome_class = terminal_value_from_outcome(
            solved=terminal_solved,
            termination_reason=terminal_reason,
        )

        n = len(group_rows)

        for position, row in enumerate(group_rows):
            steps_to_terminal = n - position

            row["outcome_value_target"] = float(
                terminal_value * (float(gamma) ** steps_to_terminal)
            )
            row["outcome_class"] = outcome_class
            row["outcome_steps_to_terminal"] = int(steps_to_terminal)
            row["outcome_value_target_mode"] = "alphazero_discounted"
            row["outcome_gamma"] = float(gamma)
            row["outcome_value_target_contract_version"] = (
                OUTCOME_VALUE_TARGET_CONTRACT_VERSION
            )


def _validate_group_outcome(
    group_rows: list[dict],
    *,
    group_key: tuple,
) -> tuple[bool, TerminationReason]:
    """Return the pre-existing, consistent terminal outcome for one episode."""

    group_label = f"group {group_key!r}"
    expected: tuple[bool, bool, TerminationReason] | None = None

    for row_index, row in enumerate(group_rows):
        solved = _require_bool(
            row.get("solved"),
            field="solved",
            group_label=group_label,
            row_index=row_index,
        )
        done = _require_bool(
            row.get("done"),
            field="done",
            group_label=group_label,
            row_index=row_index,
        )
        try:
            reason = parse_termination_reason(
                row.get("termination_reason"),
                allow_none=False,
            )
            validate_outcome_invariants(
                solved=solved,
                termination_reason=reason,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot derive outcome targets for {group_label}: invalid "
                f"episode outcome at row {row_index}: {exc}"
            ) from exc

        outcome = (solved, done, reason)
        if expected is None:
            expected = outcome
        elif outcome != expected:
            raise ValueError(
                f"Cannot derive outcome targets for {group_label}: "
                "episode-level outcomes differ between rows."
            )

    assert expected is not None
    solved, done, reason = expected
    if not done:
        raise ValueError(
            f"Cannot derive outcome targets for {group_label}: episode is not done."
        )
    return solved, reason


def _require_bool(
    value: object,
    *,
    field: str,
    group_label: str,
    row_index: int,
) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"Cannot derive outcome targets for {group_label}: {field} at row "
        f"{row_index} must be a boolean, got {value!r}."
    )

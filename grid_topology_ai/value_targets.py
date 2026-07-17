from __future__ import annotations

from grid_topology_ai.outcome import TerminationReason, parse_termination_reason
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION

VALUE_TARGET_SCHEMA_VERSION = 2


def terminal_value_from_outcome(
    solved: bool,
    termination_reason: str | None,
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

    reason = parse_termination_reason(termination_reason)
    if bool(solved) and reason is not TerminationReason.SOLVED:
        raise ValueError("solved=True is only valid with termination_reason='solved'; regenerate old examples with schema version 2")
    if reason is TerminationReason.SOLVED:
        if not bool(solved):
            raise ValueError("termination_reason='solved' requires solved=True; regenerate old examples with schema version 2")
        return 1.0, reason.value
    if reason in {TerminationReason.HANDOFF_TO_REDISPATCH, TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER}:
        return 0.0, reason.value
    if reason is None:
        return -1.0, "unsolved_terminal"
    return -1.0, reason.value


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

    The target is based only on final episode outcome:

        solved  -> +1.0 * gamma^k
        handoff ->  0.0 * gamma^k
        failed  -> -1.0 * gamma^k
    """

    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    groups: dict[tuple, list[dict]] = {}

    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    for _, group_rows in groups.items():
        group_rows.sort(key=lambda r: int(r.get("step", 0)))

        if not group_rows:
            continue

        terminal_row = group_rows[-1]

        terminal_value, outcome_class = terminal_value_from_outcome(
            solved=bool(terminal_row.get("solved", False)),
            termination_reason=terminal_row.get("termination_reason"),
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
            row["value_target_schema_version"] = VALUE_TARGET_SCHEMA_VERSION
            row["physical_objective_schema_version"] = PHYSICAL_OBJECTIVE_SCHEMA_VERSION
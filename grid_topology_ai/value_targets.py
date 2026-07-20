from __future__ import annotations

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

    The target is based only on final episode outcome:

        solved  -> +1.0 * gamma^k
        handoff ->  0.0 * gamma^k
        failed  -> -1.0 * gamma^k
    """

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

    # Derive every update before mutating any source row.  A bad later episode
    # must not leave earlier episodes partially rewritten.
    pending_updates: list[tuple[dict, dict[str, object]]] = []
    for _, group_rows in groups.items():
        group_rows.sort(key=lambda r: int(r.get("step", 0)))

        if not group_rows:
            continue

        terminal_row = group_rows[-1]

        if terminal_row.get("done") is not True:
            raise ValueError("Cannot derive outcome target from an unfinished episode.")

        # A generated episode has exactly one terminal classification.  Earlier
        # transition rows may be unfinished or may carry the propagated final
        # outcome, but may not claim a different one.
        terminal_reason = terminal_row.get("termination_reason")
        for row in group_rows[:-1]:
            reason = row.get("termination_reason")
            if row.get("done") is True or reason not in (None, ""):
                if row.get("done") is not True or (
                    bool(row.get("solved", False)) != bool(terminal_row.get("solved", False))
                    or parse_termination_reason(reason) != parse_termination_reason(terminal_reason)
                ):
                    raise ValueError("Cannot derive targets from mixed episode outcomes.")

        terminal_value, outcome_class = terminal_value_from_outcome(
            solved=bool(terminal_row.get("solved", False)),
            termination_reason=parse_termination_reason(
                terminal_reason,
            ),
        )

        n = len(group_rows)

        for position, row in enumerate(group_rows):
            steps_to_terminal = n - position
            pending_updates.append((row, {
                "outcome_value_target": float(
                    terminal_value * (float(gamma) ** steps_to_terminal)
                ),
                "outcome_class": outcome_class,
                "outcome_steps_to_terminal": int(steps_to_terminal),
                "outcome_value_target_mode": "alphazero_discounted",
                "outcome_gamma": float(gamma),
                "outcome_value_target_contract_version": (
                    OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                ),
            }))

    for row, updates in pending_updates:
        row.update(updates)

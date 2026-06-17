from __future__ import annotations


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

    reason = "" if termination_reason is None else str(termination_reason)

    if bool(solved) or reason == "solved":
        return 1.0, "solved"

    if reason in {
        "handoff_to_redispatch",
        "handoff_to_redispatch_teacher",
        "handoff_to_redispatch_with_hard_overload",
    }:
        return 0.0, "handoff_to_redispatch_teacher"

    return -1.0, reason or "unsolved_terminal"


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
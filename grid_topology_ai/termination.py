from __future__ import annotations

from enum import StrEnum


class TerminationReason(StrEnum):
    SOLVED = "solved"
    HANDOFF_TO_REDISPATCH = "handoff_to_redispatch"
    HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD = (
        "handoff_to_redispatch_with_hard_overload"
    )
    UNSAFE_STOP_WITH_HARD_OVERLOAD = "unsafe_stop_with_hard_overload"
    POWER_FLOW_FAILED = "power_flow_failed"
    MAX_STEPS_REACHED = "max_steps_reached"
    HANDOFF_TO_REDISPATCH_TEACHER = "handoff_to_redispatch_teacher"
    TEACHER_DEPTH_LIMIT = "teacher_depth_limit"


def parse_termination_reason(
    value: object,
    *,
    allow_none: bool = True,
) -> TerminationReason | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_none:
            return None
        raise ValueError("termination_reason is required.")
    if isinstance(value, TerminationReason):
        return value
    if not isinstance(value, str):
        raise TypeError("termination_reason must be a string or TerminationReason.")
    try:
        return TerminationReason(value.strip())
    except ValueError:
        allowed = ", ".join(reason.value for reason in TerminationReason)
        raise ValueError(
            f"Unknown termination_reason {value!r}. Expected one of: {allowed}."
        ) from None


def termination_reason_value(reason: TerminationReason | None) -> str | None:
    return None if reason is None else reason.value


def validate_outcome_invariants(
    *,
    solved: bool,
    termination_reason: TerminationReason | str | None,
    physically_secure: bool | None = None,
) -> TerminationReason | None:
    reason = parse_termination_reason(termination_reason)
    if bool(solved) != (reason is TerminationReason.SOLVED):
        raise ValueError(
            "Contradictory outcome: solved=True is compatible only with "
            "TerminationReason.SOLVED, and SOLVED requires solved=True."
        )
    if physically_secure is not None and bool(solved) != bool(physically_secure):
        raise ValueError(
            "Contradictory outcome: solved must equal physically_secure for "
            "classified episode outcomes."
        )
    return reason

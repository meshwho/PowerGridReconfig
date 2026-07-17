from __future__ import annotations

from enum import StrEnum


class TerminationReason(StrEnum):
    SOLVED = "solved"
    HANDOFF_TO_REDISPATCH = "handoff_to_redispatch"
    HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD = "handoff_to_redispatch_with_hard_overload"
    UNSAFE_STOP = "unsafe_stop"
    POWER_FLOW_FAILED = "power_flow_failed"
    MAX_STEPS_REACHED = "max_steps_reached"
    HANDOFF_TO_REDISPATCH_TEACHER = "handoff_to_redispatch_teacher"
    TEACHER_DEPTH_LIMIT = "teacher_depth_limit"


def parse_termination_reason(value: str | TerminationReason | None) -> TerminationReason | None:
    if value is None:
        return None
    if isinstance(value, TerminationReason):
        return value
    try:
        return TerminationReason(str(value))
    except ValueError as exc:
        raise ValueError(f"Unknown termination_reason: {value!r}") from exc

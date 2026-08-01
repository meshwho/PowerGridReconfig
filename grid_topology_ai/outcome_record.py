from __future__ import annotations

import numpy as np

from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
    validate_outcome_invariants,
)


def parse_terminal_outcome_fields(
    *,
    solved: object,
    termination_reason: object,
) -> tuple[bool, TerminationReason]:
    """Parse the two scalar fields that identify a terminal outcome."""

    if not isinstance(solved, (bool, np.bool_)):
        raise ValueError(
            f"solved must be a boolean, got {solved!r}"
        )

    parsed_solved = bool(solved)
    reason = parse_termination_reason(
        termination_reason,
        allow_none=False,
    )
    assert reason is not None

    validate_outcome_invariants(
        solved=parsed_solved,
        termination_reason=reason,
    )
    return parsed_solved, reason

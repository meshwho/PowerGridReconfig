from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from grid_topology_ai.outcome_contract import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
    TerminalOutcomeEvidence,
)
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


def terminal_evidence_from_metadata(
    metadata: Mapping[str, object],
    *,
    source: str,
    solved: object,
    termination_reason: object,
) -> TerminalOutcomeEvidence:
    """Read terminal evidence from versioned artifact metadata."""

    parsed_solved, reason = parse_terminal_outcome_fields(
        solved=solved,
        termination_reason=termination_reason,
    )

    version = metadata.get(
        "terminal_outcome_evidence_schema_version"
    )
    if (
        isinstance(version, bool)
        or not isinstance(version, Integral)
        or int(version)
        != TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{source} has unsupported terminal outcome evidence "
            f"schema version {version!r}."
        )

    raw_evidence = metadata.get("terminal_outcome_evidence")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError(
            f"{source} is missing terminal_outcome_evidence metadata."
        )

    try:
        evidence = TerminalOutcomeEvidence.from_mapping(
            raw_evidence
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} contains invalid terminal_outcome_evidence: "
            f"{exc}"
        ) from exc

    if (
        evidence.solved is not parsed_solved
        or evidence.termination_reason is not reason
    ):
        raise ValueError(
            f"{source} terminal_outcome_evidence contradicts "
            "episode_solved or episode_termination_reason."
        )

    return evidence

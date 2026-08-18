"""Unified value-return contract for topology control."""

from __future__ import annotations

import math
from numbers import Real

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.grid_utility import (
    DEFAULT_STATE_UTILITY_SCALE,
    state_utility,
)
from grid_topology_ai.outcome_contract import TerminalOutcomeEvidence
from grid_topology_ai.termination import (
    TerminationReason,
    validate_outcome_invariants,
)


VALUE_TARGET_MODE = "final_topology_state_utility"
TERMINAL_UTILITY_GAMMA = 1.0
DEFAULT_HEURISTIC_UTILITY_SCALE = DEFAULT_STATE_UTILITY_SCALE
_UTILITY_TOLERANCE = 1e-7


def require_reward_discount_factor(value: object) -> float:
    """Validate the discount used by diagnostic reward accumulation."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"gamma must be a finite real number in [0, 1], got {value!r}"
        )
    gamma = float(value)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(
            f"gamma must be a finite real number in [0, 1], got {value!r}"
        )
    return gamma


def require_discount_factor(value: object) -> float:
    """Validate a caller gamma and return the fixed value-target gamma."""
    require_reward_discount_factor(value)
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

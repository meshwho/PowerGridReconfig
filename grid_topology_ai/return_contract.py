"""Unified AlphaZero-style terminal-utility contract."""

from __future__ import annotations

import math
from numbers import Integral, Real

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.grid_utility import state_security_penalty
from grid_topology_ai.outcome_contract import (
    RedispatchStatus,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.termination import (
    TerminationReason,
    validate_outcome_invariants,
)

VALUE_TARGET_MODE = "alphazero_terminal_utility"
TERMINAL_UTILITY_GAMMA = 1.0
DEFAULT_HEURISTIC_UTILITY_SCALE = 500.0
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
    """Validate a caller gamma and return the fixed terminal-utility gamma."""
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


def terminal_utility_from_outcome(
    solved: bool,
    termination_reason: TerminationReason | str | None,
    *,
    evidence: TerminalOutcomeEvidence | None = None,
) -> tuple[float, str]:
    """Map one terminal episode outcome to the canonical utility and class.

    ``+1`` means the topology controller physically solved the grid. ``0`` is
    reserved for a redispatch that was actually executed and validated against
    the physical contract. A redispatch request or handoff alone is an unsolved
    terminal outcome and therefore receives ``-1``.
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

    if reason is TerminationReason.SOLVED:
        return 1.0, TerminationReason.SOLVED.value

    if reason is TerminationReason.REDISPATCH_VALIDATED:
        if evidence is None:
            raise ValueError(
                "redispatch_validated requires terminal outcome evidence."
            )
        if (
            evidence.redispatch_status is not RedispatchStatus.VALIDATED
            or evidence.redispatch_assessment is None
            or not evidence.redispatch_assessment.physically_secure
        ):
            raise ValueError(
                "redispatch_validated requires a physically secure validated "
                "redispatch assessment."
            )
        return 0.0, TerminationReason.REDISPATCH_VALIDATED.value

    return -1.0, "unsolved_terminal" if reason is None else reason.value


def discounted_terminal_utility(
    terminal_utility: object,
    *,
    steps_to_terminal: object,
    gamma: object,
) -> float:
    """Discount terminal utility over an exact number of transitions."""
    utility = require_bounded_utility(
        terminal_utility,
        context="terminal_utility",
    )
    if (
        isinstance(steps_to_terminal, bool)
        or not isinstance(steps_to_terminal, Integral)
        or int(steps_to_terminal) < 0
    ):
        raise ValueError(
            "steps_to_terminal must be a non-negative integer, "
            f"got {steps_to_terminal!r}"
        )
    discount = require_reward_discount_factor(gamma)
    return float(utility * discount ** int(steps_to_terminal))


def heuristic_terminal_utility_estimate(
    state: GridFMState,
    *,
    physics_config: PhysicsConfig | None = None,
    utility_scale: float = DEFAULT_HEURISTIC_UTILITY_SCALE,
) -> float:
    """Return a bounded fallback estimate with terminal-utility semantics.

    This is used only when no neural value is available. It monotonically maps
    the shared physical penalty to ``[-1, 1]``: a zero-penalty state estimates
    ``+1`` and increasingly unsafe states approach ``-1``. It is a search
    heuristic, never a training target.
    """
    scale = float(utility_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("utility_scale must be finite and > 0")
    penalty = state_security_penalty(
        state,
        physics_config=physics_config,
    )
    estimate = 1.0 - 2.0 * penalty / (penalty + scale)
    return require_bounded_utility(
        estimate,
        context="heuristic terminal utility estimate",
    )
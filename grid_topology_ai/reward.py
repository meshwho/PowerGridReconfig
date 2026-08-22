from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.physics.utility import (
    GridUtilityBreakdown,
    GridUtilityWeights,
    grid_utility_breakdown,
    potential_shaping_reward,
    DEFAULT_STATE_UTILITY_SCALE,
    state_utility,
)
from grid_topology_ai.outcome_record import TerminalOutcomeEvidence
from grid_topology_ai.termination import (
    TerminationReason,
    validate_outcome_invariants,
)
from grid_topology_ai.physics.objective import (
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
    assess_physical_state,
)

@dataclass(frozen=True)
class GridFMRewardBreakdown:
    """Detailed potential-shaping diagnostics for one transition."""

    # Stable fields retained for environment and transition-table consumers.
    reward: float
    before_penalty: float
    after_penalty: float
    improvement: float
    switching_penalty: float
    before_max_loading: float
    after_max_loading: float
    before_total_overload: float
    after_total_overload: float
    before_num_overloaded: int
    after_num_overloaded: int
    before_num_hard_overloaded: int
    after_num_hard_overloaded: int
    before_voltage_penalty: float
    after_voltage_penalty: float
    done: bool
    success: bool
    message: str

    # Explicit provenance for the diagnostic reward contract. Defaults preserve
    # compatibility with lightweight test doubles built against the old schema.
    potential_shaping: float = 0.0
    discount_factor: float = 1.0
    before_potential: float = 0.0
    after_potential: float | None = None
    reward_role: str = "diagnostic_potential_shaping"


class GridFMReward:
    """Policy-invariant potential shaping used only for diagnostics.

    The optimized return is defined in :mod:`grid_topology_ai.reward`.
    This class must not add switching costs, solved bonuses, or terminal failure
    penalties because those terms are not potential based and would define a
    second objective.
    """

    CONTRACT = "potential_shaping_v1"

    def __init__(
        self,
        *,
        physics_config: PhysicsConfig | None = None,
        discount_factor: float = 0.95,
        overload_limit_percent: float = OVERLOAD_LIMIT_PERCENT,
        hard_overload_limit_percent: float = HARD_OVERLOAD_LIMIT_PERCENT,
        total_overload_weight: float = 2.0,
        hard_overload_weight: float = 5.0,
        num_overloaded_weight: float = 10.0,
        num_hard_overloaded_weight: float = 30.0,
        voltage_violation_weight: float = 500.0,
    ):
        if physics_config is not None:
            if (
                overload_limit_percent != OVERLOAD_LIMIT_PERCENT
                or hard_overload_limit_percent != HARD_OVERLOAD_LIMIT_PERCENT
            ):
                raise ValueError(
                    "PhysicsConfig cannot be combined with explicit overload thresholds."
                )
            overload_limit_percent = physics_config.overload_limit_percent
            hard_overload_limit_percent = physics_config.hard_overload_limit_percent

        self.physics_config = physics_config
        self.discount_factor = require_reward_discount_factor(
            discount_factor
        )
        self.overload_limit_percent = float(overload_limit_percent)
        self.hard_overload_limit_percent = float(hard_overload_limit_percent)
        self.utility_weights = GridUtilityWeights(
            total_overload=total_overload_weight,
            hard_overload=hard_overload_weight,
            num_overloaded=num_overloaded_weight,
            num_hard_overloaded=num_hard_overloaded_weight,
            voltage_violation=voltage_violation_weight,
        )

        self.total_overload_weight = self.utility_weights.total_overload
        self.hard_overload_weight = self.utility_weights.hard_overload
        self.num_overloaded_weight = self.utility_weights.num_overloaded
        self.num_hard_overloaded_weight = self.utility_weights.num_hard_overloaded
        self.voltage_violation_weight = self.utility_weights.voltage_violation

    def config_dict(self) -> dict[str, Any]:
        """Return reproducible shaping provenance."""

        return {
            "reward_contract": self.CONTRACT,
            "reward_role": "diagnostic_only",
            "discount_factor": self.discount_factor,
            "overload_limit_percent": self.overload_limit_percent,
            "hard_overload_limit_percent": self.hard_overload_limit_percent,
            "total_overload_weight": self.total_overload_weight,
            "hard_overload_weight": self.hard_overload_weight,
            "num_overloaded_weight": self.num_overloaded_weight,
            "num_hard_overloaded_weight": self.num_hard_overloaded_weight,
            "voltage_violation_weight": self.voltage_violation_weight,
        }

    def compute(
        self,
        before_state: GridFMState,
        after_state: GridFMState | None,
        power_flow_success: bool,
    ) -> GridFMRewardBreakdown:
        """Compute ``gamma*Phi(after) - Phi(before)`` and diagnostics."""

        before = self._utility_breakdown(before_state)
        before_potential = -before.penalty

        if not power_flow_success or after_state is None:
            return GridFMRewardBreakdown(
                reward=0.0,
                potential_shaping=0.0,
                discount_factor=self.discount_factor,
                before_potential=before_potential,
                after_potential=None,
                before_penalty=before.penalty,
                after_penalty=float("inf"),
                improvement=float("-inf"),
                switching_penalty=0.0,
                before_max_loading=float(
                    before_state.metrics["max_loading_percent"]
                ),
                after_max_loading=float("inf"),
                before_total_overload=before.total_overload,
                after_total_overload=float("inf"),
                before_num_overloaded=before.num_overloaded,
                after_num_overloaded=10**9,
                before_num_hard_overloaded=before.num_hard_overloaded,
                after_num_hard_overloaded=10**9,
                before_voltage_penalty=before.voltage_violation,
                after_voltage_penalty=float("inf"),
                done=True,
                success=False,
                message=(
                    "Power flow failed; no non-potential failure reward was added."
                ),
            )

        after = self._utility_breakdown(after_state)
        after_potential = -after.penalty
        improvement = before.penalty - after.penalty
        shaping = potential_shaping_reward(
            before_state,
            after_state,
            discount_factor=self.discount_factor,
            physics_config=self.physics_config,
            overload_limit_percent=(
                None
                if self.physics_config is not None
                else self.overload_limit_percent
            ),
            hard_overload_limit_percent=(
                None
                if self.physics_config is not None
                else self.hard_overload_limit_percent
            ),
            weights=self.utility_weights,
        )
        assessment = assess_physical_state(after_state.metrics)

        return GridFMRewardBreakdown(
            reward=shaping,
            potential_shaping=shaping,
            discount_factor=self.discount_factor,
            before_potential=before_potential,
            after_potential=after_potential,
            before_penalty=before.penalty,
            after_penalty=after.penalty,
            improvement=float(improvement),
            switching_penalty=0.0,
            before_max_loading=float(before_state.metrics["max_loading_percent"]),
            after_max_loading=float(after_state.metrics["max_loading_percent"]),
            before_total_overload=before.total_overload,
            after_total_overload=after.total_overload,
            before_num_overloaded=before.num_overloaded,
            after_num_overloaded=after.num_overloaded,
            before_num_hard_overloaded=before.num_hard_overloaded,
            after_num_hard_overloaded=after.num_hard_overloaded,
            before_voltage_penalty=before.voltage_violation,
            after_voltage_penalty=after.voltage_violation,
            done=assessment.physically_secure,
            success=True,
            message="Diagnostic potential shaping computed successfully.",
        )

    def _utility_breakdown(self, state: GridFMState) -> GridUtilityBreakdown:
        return grid_utility_breakdown(
            state,
            physics_config=self.physics_config,
            overload_limit_percent=(
                None
                if self.physics_config is not None
                else self.overload_limit_percent
            ),
            hard_overload_limit_percent=(
                None
                if self.physics_config is not None
                else self.hard_overload_limit_percent
            ),
            weights=self.utility_weights,
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

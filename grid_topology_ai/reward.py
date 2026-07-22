from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.grid_utility import (
    GridUtilityBreakdown,
    GridUtilityWeights,
    grid_utility_breakdown,
    potential_shaping_reward,
)
from grid_topology_ai.physical_objective import (
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
    assess_physical_state,
)
from grid_topology_ai.return_contract import require_discount_factor


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

    The optimized return is defined in :mod:`grid_topology_ai.return_contract`.
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
        switching_penalty: float = 0.0,
        non_convergence_penalty: float = 0.0,
        solved_bonus: float = 0.0,
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

        non_potential_terms = {
            "switching_penalty": switching_penalty,
            "non_convergence_penalty": non_convergence_penalty,
            "solved_bonus": solved_bonus,
        }
        enabled = {
            name: float(value)
            for name, value in non_potential_terms.items()
            if float(value) != 0.0
        }
        if enabled:
            raise ValueError(
                "GridFMReward accepts only potential-based shaping. "
                f"Remove non-potential terms: {enabled}."
            )

        self.physics_config = physics_config
        self.discount_factor = require_discount_factor(discount_factor)
        self.overload_limit_percent = float(overload_limit_percent)
        self.hard_overload_limit_percent = float(hard_overload_limit_percent)
        self.utility_weights = GridUtilityWeights(
            total_overload=total_overload_weight,
            hard_overload=hard_overload_weight,
            num_overloaded=num_overloaded_weight,
            num_hard_overloaded=num_hard_overloaded_weight,
            voltage_violation=voltage_violation_weight,
        )

        # Compatibility attributes remain zero by contract.
        self.switching_penalty = 0.0
        self.non_convergence_penalty = 0.0
        self.solved_bonus = 0.0
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
        action_is_switching: bool,
        power_flow_success: bool,
    ) -> GridFMRewardBreakdown:
        """Compute ``gamma*Phi(after) - Phi(before)`` and diagnostics."""

        _ = action_is_switching  # Switching costs are outside this contract.
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

    def _state_penalty(self, state: GridFMState) -> float:
        """Compatibility wrapper around the canonical grid utility."""

        return self._utility_breakdown(state).penalty

    def _total_overload(self, state: GridFMState) -> float:
        return self._utility_breakdown(state).total_overload

    def _total_hard_overload(self, state: GridFMState) -> float:
        return self._utility_breakdown(state).total_hard_overload

    def _voltage_penalty(self, state: GridFMState) -> float:
        return self._utility_breakdown(state).voltage_violation

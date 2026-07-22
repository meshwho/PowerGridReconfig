from __future__ import annotations

from dataclasses import dataclass

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.grid_utility import (
    GridUtilityBreakdown,
    GridUtilityWeights,
    grid_utility_breakdown,
)
from grid_topology_ai.physical_objective import (
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
    assess_physical_state,
)


@dataclass(frozen=True)
class GridFMRewardBreakdown:
    """Detailed and auditable reward explanation."""

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


class GridFMReward:
    """Dense diagnostic reward based on shared physical grid utility."""

    def __init__(
        self,
        *,
        physics_config: PhysicsConfig | None = None,
        overload_limit_percent: float = OVERLOAD_LIMIT_PERCENT,
        hard_overload_limit_percent: float = HARD_OVERLOAD_LIMIT_PERCENT,
        switching_penalty: float = 1.0,
        non_convergence_penalty: float = 1000.0,
        solved_bonus: float = 50.0,
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
        self.overload_limit_percent = float(overload_limit_percent)
        self.hard_overload_limit_percent = float(hard_overload_limit_percent)
        self.switching_penalty = float(switching_penalty)
        self.non_convergence_penalty = float(non_convergence_penalty)
        self.solved_bonus = float(solved_bonus)
        self.utility_weights = GridUtilityWeights(
            total_overload=total_overload_weight,
            hard_overload=hard_overload_weight,
            num_overloaded=num_overloaded_weight,
            num_hard_overloaded=num_hard_overloaded_weight,
            voltage_violation=voltage_violation_weight,
        )

        # Public compatibility attributes used in metadata and tests.
        self.total_overload_weight = self.utility_weights.total_overload
        self.hard_overload_weight = self.utility_weights.hard_overload
        self.num_overloaded_weight = self.utility_weights.num_overloaded
        self.num_hard_overloaded_weight = self.utility_weights.num_hard_overloaded
        self.voltage_violation_weight = self.utility_weights.voltage_violation

    def config_dict(self) -> dict[str, float]:
        """Return reproducible reward and utility configuration."""
        return {
            "overload_limit_percent": self.overload_limit_percent,
            "hard_overload_limit_percent": self.hard_overload_limit_percent,
            "switching_penalty": self.switching_penalty,
            "non_convergence_penalty": self.non_convergence_penalty,
            "solved_bonus": self.solved_bonus,
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
        """Compute one dense transition reward and its physical breakdown."""
        before = self._utility_breakdown(before_state)

        if not power_flow_success or after_state is None:
            topology_cost = self.switching_penalty if action_is_switching else 0.0
            return GridFMRewardBreakdown(
                reward=-self.non_convergence_penalty,
                before_penalty=before.penalty,
                after_penalty=self.non_convergence_penalty,
                improvement=-self.non_convergence_penalty,
                switching_penalty=topology_cost,
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
                message="Power flow failed after action.",
            )

        after = self._utility_breakdown(after_state)
        improvement = before.penalty - after.penalty
        topology_cost = self.switching_penalty if action_is_switching else 0.0
        reward = improvement - topology_cost

        assessment = assess_physical_state(after_state.metrics)
        if assessment.physically_secure:
            reward += self.solved_bonus

        return GridFMRewardBreakdown(
            reward=float(reward),
            before_penalty=before.penalty,
            after_penalty=after.penalty,
            improvement=float(improvement),
            switching_penalty=topology_cost,
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
            message="Reward computed successfully.",
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

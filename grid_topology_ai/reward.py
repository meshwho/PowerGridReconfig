from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_topology_ai.physical_objective import (
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
    assess_physical_state,
)
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)


@dataclass(frozen=True)
class GridFMRewardBreakdown:
    """
    Detailed reward explanation.

    This is important for debugging:
    we do not want a black-box reward where we cannot understand why
    an action was considered good or bad.
    """

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
    """
    Reward function for topology switching.

    Main idea:
        reward = improvement in grid security - switching cost

    We compare:
        before_state -> after_state

    If the action improves the grid, reward is positive.
    If it makes the grid worse, reward is negative.
    """

    def __init__(
        self,
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
        self.overload_limit_percent = overload_limit_percent
        self.hard_overload_limit_percent = hard_overload_limit_percent
        self.switching_penalty = switching_penalty
        self.non_convergence_penalty = non_convergence_penalty
        self.solved_bonus = solved_bonus
        self.total_overload_weight = float(total_overload_weight)
        self.hard_overload_weight = float(hard_overload_weight)
        self.num_overloaded_weight = float(num_overloaded_weight)
        self.num_hard_overloaded_weight = float(num_hard_overloaded_weight)
        self.voltage_violation_weight = float(voltage_violation_weight)
        self.loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
        self.status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")

        self.vm_idx = BUS_FEATURE_COLUMNS.index("Vm")

    def config_dict(self) -> dict[str, float]:
        """
        Return reward configuration for experiment metadata.

        This makes reward settings reproducible across dataset generation,
        training, evaluation, and checkpoints.
        """

        return {
            "overload_limit_percent": float(self.overload_limit_percent),
            "hard_overload_limit_percent": float(self.hard_overload_limit_percent),
            "switching_penalty": float(self.switching_penalty),
            "non_convergence_penalty": float(self.non_convergence_penalty),
            "solved_bonus": float(self.solved_bonus),
            "total_overload_weight": float(self.total_overload_weight),
            "hard_overload_weight": float(self.hard_overload_weight),
            "num_overloaded_weight": float(self.num_overloaded_weight),
            "num_hard_overloaded_weight": float(self.num_hard_overloaded_weight),
            "voltage_violation_weight": float(self.voltage_violation_weight),
        }

    def compute(
        self,
        before_state: GridFMState,
        after_state: GridFMState | None,
        action_is_switching: bool,
        power_flow_success: bool,
    ) -> GridFMRewardBreakdown:
        """
        Compute reward for one transition.

        Parameters
        ----------
        before_state:
            State before action.

        after_state:
            State after action. Can be None if power flow failed.

        action_is_switching:
            True if the action changed topology.
            False for do_nothing.

        power_flow_success:
            Whether AC power flow converged after action.
        """

        before_penalty = self._state_penalty(before_state)

        if not power_flow_success or after_state is None:
            reward = -self.non_convergence_penalty

            return GridFMRewardBreakdown(
                reward=reward,
                before_penalty=before_penalty,
                after_penalty=self.non_convergence_penalty,
                improvement=-self.non_convergence_penalty,
                switching_penalty=self.switching_penalty if action_is_switching else 0.0,
                before_max_loading=before_state.metrics["max_loading_percent"],
                after_max_loading=float("inf"),
                before_total_overload=self._total_overload(before_state),
                after_total_overload=float("inf"),
                before_num_overloaded=before_state.metrics["num_overloaded_branches"],
                after_num_overloaded=10**9,
                before_num_hard_overloaded=before_state.metrics[
                    "num_hard_overloaded_branches"
                ],
                after_num_hard_overloaded=10**9,
                before_voltage_penalty=self._voltage_penalty(before_state),
                after_voltage_penalty=float("inf"),
                done=True,
                success=False,
                message="Power flow failed after action.",
            )

        after_penalty = self._state_penalty(after_state)

        improvement = before_penalty - after_penalty

        topology_cost = self.switching_penalty if action_is_switching else 0.0

        reward = improvement - topology_cost

        # Reward magnitude follows the authoritative physical assessment.
        assessment = assess_physical_state(after_state.metrics)
        done = assessment.physically_secure

        if done:
            reward += self.solved_bonus

        return GridFMRewardBreakdown(
            reward=float(reward),
            before_penalty=float(before_penalty),
            after_penalty=float(after_penalty),
            improvement=float(improvement),
            switching_penalty=float(topology_cost),
            before_max_loading=float(before_state.metrics["max_loading_percent"]),
            after_max_loading=float(after_state.metrics["max_loading_percent"]),
            before_total_overload=float(self._total_overload(before_state)),
            after_total_overload=float(self._total_overload(after_state)),
            before_num_overloaded=int(before_state.metrics["num_overloaded_branches"]),
            after_num_overloaded=int(after_state.metrics["num_overloaded_branches"]),
            before_num_hard_overloaded=int(
                before_state.metrics["num_hard_overloaded_branches"]
            ),
            after_num_hard_overloaded=int(
                after_state.metrics["num_hard_overloaded_branches"]
            ),
            before_voltage_penalty=float(self._voltage_penalty(before_state)),
            after_voltage_penalty=float(self._voltage_penalty(after_state)),
            done=bool(done),
            success=True,
            message="Reward computed successfully.",
        )

    def _state_penalty(self, state: GridFMState) -> float:
        """
        Convert one grid state into a scalar penalty.

        Lower is better.

        Penalty components:
        - total overload above 100%;
        - hard overload above 120%;
        - number of overloaded branches;
        - voltage violations.
        """

        total_overload = self._total_overload(state)
        hard_overload = self._total_hard_overload(state)
        voltage_penalty = self._voltage_penalty(state)

        num_overloaded = state.metrics["num_overloaded_branches"]
        num_hard_overloaded = state.metrics["num_hard_overloaded_branches"]

        penalty = (
            self.total_overload_weight * total_overload
            + self.hard_overload_weight * hard_overload
            + self.num_overloaded_weight * num_overloaded
            + self.num_hard_overloaded_weight * num_hard_overloaded
            + self.voltage_violation_weight * voltage_penalty
        )

        return float(penalty)

    def _active_loadings(self, state: GridFMState) -> np.ndarray:
        """
        Return loading_percent only for active branches.
        """

        status = state.branch_features[:, self.status_idx]
        loading = state.branch_features[:, self.loading_idx]

        return loading[status > 0]

    def _total_overload(self, state: GridFMState) -> float:
        """
        Sum of overloads above 100%.

        Example:
            loading = [80, 105, 130]
            total_overload = 0 + 5 + 30 = 35
        """

        loading = self._active_loadings(state)

        overload = np.maximum(loading - self.overload_limit_percent, 0.0)

        return float(np.sum(overload))

    def _total_hard_overload(self, state: GridFMState) -> float:
        """
        Sum of overloads above hard threshold, for example 120%.
        """

        loading = self._active_loadings(state)

        hard_overload = np.maximum(loading - self.hard_overload_limit_percent, 0.0)

        return float(np.sum(hard_overload))

    def _voltage_penalty(self, state: GridFMState) -> float:
        """
        Voltage penalty based on violation magnitude, not only count.

        Example:
            Vm = 1.0601 with max limit 1.06 should produce a tiny penalty.
            Vm = 1.10 with max limit 1.06 should produce a much larger penalty.

        This is much better than counting violation buses.
        """

        if "total_voltage_violation" in state.metrics:
            return float(state.metrics["total_voltage_violation"])

        # Fallback for older states.
        num_low = state.metrics["num_low_voltage_buses"]
        num_high = state.metrics["num_high_voltage_buses"]

        return float(num_low + num_high)

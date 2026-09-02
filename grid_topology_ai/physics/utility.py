"""Canonical physical grid scoring and potential shaping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.physics.objective import assess_physical_state
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState


DEFAULT_STATE_UTILITY_SCALE = 500.0


@dataclass(frozen=True, slots=True)
class GridUtilityWeights:
    """Weights for one lower-is-better physical security penalty."""

    total_overload: float = 2.0
    hard_overload: float = 5.0
    num_overloaded: float = 10.0
    num_hard_overloaded: float = 30.0
    voltage_violation: float = 500.0
    max_loading_excess: float = 0.0
    generator_p_violation: float = 1.0
    generator_q_violation: float = 1.0
    angle_difference_violation: float = 1.0
    invalid_physical_state: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("total_overload", self.total_overload),
            ("hard_overload", self.hard_overload),
            ("num_overloaded", self.num_overloaded),
            ("num_hard_overloaded", self.num_hard_overloaded),
            ("voltage_violation", self.voltage_violation),
            ("max_loading_excess", self.max_loading_excess),
            ("generator_p_violation", self.generator_p_violation),
            ("generator_q_violation", self.generator_q_violation),
            ("angle_difference_violation", self.angle_difference_violation),
            ("invalid_physical_state", self.invalid_physical_state),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative.")


DEFAULT_GRID_UTILITY_WEIGHTS = GridUtilityWeights()
CONTINUATION_GRID_UTILITY_WEIGHTS = GridUtilityWeights(
    total_overload=4.0,
    hard_overload=30.0,
    num_overloaded=80.0,
    num_hard_overloaded=1000.0,
    voltage_violation=500.0,
    max_loading_excess=5.0,
)
CONTINUATION_SWITCH_PENALTY = 8.0


@dataclass(frozen=True, slots=True)
class GridUtilityBreakdown:
    """Auditable components of the physical security penalty."""

    total_overload: float
    total_hard_overload: float
    num_overloaded: int
    num_hard_overloaded: int
    voltage_violation: float
    max_loading_excess: float
    penalty: float
    generator_p_violation_mw: float
    num_generator_p_violations: int
    generator_q_violation_mvar: float
    num_generator_q_violations: int
    angle_difference_violation_degrees: float
    num_angle_difference_violations: int
    invalid_physical_state_flags: int


def _require_physics_config(physics_config: PhysicsConfig) -> PhysicsConfig:
    if not isinstance(physics_config, PhysicsConfig):
        raise TypeError("physics_config must be a PhysicsConfig.")
    return physics_config


def _require_discount_factor(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"discount_factor must be a finite real number in [0, 1], got {value!r}"
        )
    discount = float(value)
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError(
            f"discount_factor must be a finite real number in [0, 1], got {value!r}"
        )
    return discount


def _nonnegative_metric(state: GridFMState, key: str) -> float:
    value = float(state.metrics[key])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative.")
    return value


def _nonnegative_count(state: GridFMState, key: str) -> int:
    raw = state.metrics[key]
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be a non-negative integer.")
    value = int(raw)
    if value < 0 or float(raw) != float(value):
        raise ValueError(f"{key} must be a non-negative integer.")
    return value


def _invalid_physical_state_flags(state: GridFMState) -> int:
    count = 0
    for key in (
        "power_flow_converged",
        "all_values_finite",
        "topology_connected",
    ):
        value = state.metrics[key]
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be a bool.")
        count += int(not bool(value))
    return count


def active_branch_loadings(state: GridFMState) -> np.ndarray:
    """Return finite loading percentages for active branches only."""

    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
    status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")
    loading = np.asarray(state.branch_features[:, loading_idx], dtype=np.float64)
    status = np.asarray(state.branch_features[:, status_idx], dtype=np.float64)
    active = loading[status > 0.0]
    if not np.isfinite(active).all():
        raise ValueError("Active branch loadings must be finite.")
    return active


def grid_utility_breakdown(
    state: GridFMState,
    *,
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> GridUtilityBreakdown:
    """Build the canonical physical penalty and all of its components."""

    config = _require_physics_config(physics_config)
    overload_limit = config.overload_limit_percent
    hard_limit = config.hard_overload_limit_percent
    tolerance = config.thermal_tolerance_percent
    loading = active_branch_loadings(state)

    total_overload = float(
        np.sum(
            np.where(
                loading > overload_limit + tolerance,
                loading - overload_limit,
                0.0,
            )
        )
    )
    total_hard_overload = float(
        np.sum(
            np.where(
                loading > hard_limit + tolerance,
                loading - hard_limit,
                0.0,
            )
        )
    )
    num_overloaded = _nonnegative_count(state, "num_overloaded_branches")
    num_hard_overloaded = _nonnegative_count(
        state, "num_hard_overloaded_branches"
    )
    if num_hard_overloaded > num_overloaded:
        raise ValueError("Invalid overloaded-branch counts in state metrics.")

    voltage_violation = _nonnegative_metric(state, "total_voltage_violation")
    max_loading = _nonnegative_metric(state, "max_loading_percent")
    max_loading_excess = (
        max_loading - overload_limit
        if max_loading > overload_limit + tolerance
        else 0.0
    )

    generator_p_violation_mw = _nonnegative_metric(
        state, "total_generator_p_violation_mw"
    )
    num_generator_p_violations = _nonnegative_count(
        state, "num_generator_p_violations"
    )
    generator_q_violation_mvar = _nonnegative_metric(
        state, "total_generator_q_violation_mvar"
    )
    num_generator_q_violations = _nonnegative_count(
        state, "num_generator_q_violations"
    )
    angle_difference_violation_degrees = _nonnegative_metric(
        state, "total_angle_difference_violation_degrees"
    )
    num_angle_difference_violations = _nonnegative_count(
        state, "num_angle_difference_violations"
    )
    invalid_physical_state_flags = _invalid_physical_state_flags(state)

    generator_p_violation = (
        generator_p_violation_mw + float(num_generator_p_violations)
    )
    generator_q_violation = (
        generator_q_violation_mvar + float(num_generator_q_violations)
    )
    angle_difference_violation = (
        angle_difference_violation_degrees + float(num_angle_difference_violations)
    )

    penalty = (
        weights.total_overload * total_overload
        + weights.hard_overload * total_hard_overload
        + weights.num_overloaded * num_overloaded
        + weights.num_hard_overloaded * num_hard_overloaded
        + weights.voltage_violation * voltage_violation
        + weights.max_loading_excess * max_loading_excess
        + weights.generator_p_violation * generator_p_violation
        + weights.generator_q_violation * generator_q_violation
        + weights.angle_difference_violation * angle_difference_violation
        + weights.invalid_physical_state * invalid_physical_state_flags
    )
    if not math.isfinite(penalty):
        raise ValueError("Grid utility penalty must be finite.")

    return GridUtilityBreakdown(
        total_overload=total_overload,
        total_hard_overload=total_hard_overload,
        num_overloaded=num_overloaded,
        num_hard_overloaded=num_hard_overloaded,
        voltage_violation=voltage_violation,
        max_loading_excess=float(max_loading_excess),
        penalty=float(penalty),
        generator_p_violation_mw=generator_p_violation_mw,
        num_generator_p_violations=num_generator_p_violations,
        generator_q_violation_mvar=generator_q_violation_mvar,
        num_generator_q_violations=num_generator_q_violations,
        angle_difference_violation_degrees=angle_difference_violation_degrees,
        num_angle_difference_violations=num_angle_difference_violations,
        invalid_physical_state_flags=invalid_physical_state_flags,
    )


def state_security_penalty(
    state: GridFMState,
    *,
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> float:
    """Return the canonical lower-is-better grid security penalty."""

    return grid_utility_breakdown(
        state,
        physics_config=physics_config,
        weights=weights,
    ).penalty


def state_utility(
    state: GridFMState,
    *,
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    utility_scale: float = DEFAULT_STATE_UTILITY_SCALE,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> float:
    """Map the canonical physical penalty monotonically into ``[-1, 1]``."""

    scale = float(utility_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("utility_scale must be finite and > 0")
    penalty = state_security_penalty(
        state,
        physics_config=physics_config,
        weights=weights,
    )
    return float(1.0 - 2.0 * penalty / (penalty + scale))


def state_potential(
    state: GridFMState,
    *,
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> float:
    """Return the higher-is-better potential ``Phi(s) = -penalty(s)``."""

    return -state_security_penalty(
        state,
        physics_config=physics_config,
        weights=weights,
    )


def potential_shaping_reward(
    before_state: GridFMState,
    after_state: GridFMState,
    *,
    discount_factor: float,
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> float:
    """Return policy-invariant potential shaping ``gamma*Phi(s') - Phi(s)``."""

    gamma = _require_discount_factor(discount_factor)
    before_potential = state_potential(
        before_state,
        physics_config=physics_config,
        weights=weights,
    )
    after_potential = state_potential(
        after_state,
        physics_config=physics_config,
        weights=weights,
    )
    shaping = gamma * after_potential - before_potential
    if not math.isfinite(shaping):
        raise ValueError("Potential shaping reward must be finite.")
    return float(shaping)


@dataclass(frozen=True)
class GridFMRewardBreakdown:
    """Detailed potential-shaping diagnostics for one transition."""

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
    potential_shaping: float
    discount_factor: float
    before_potential: float
    after_potential: float | None
    reward_role: str


class GridFMReward:
    """Policy-invariant potential shaping used only for diagnostics."""

    CONTRACT = "potential_shaping_v1"

    def __init__(
        self,
        *,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        discount_factor: float = 0.95,
        total_overload_weight: float = 2.0,
        hard_overload_weight: float = 5.0,
        num_overloaded_weight: float = 10.0,
        num_hard_overloaded_weight: float = 30.0,
        voltage_violation_weight: float = 500.0,
    ):
        self.physics_config = _require_physics_config(physics_config)
        self.discount_factor = require_reward_discount_factor(discount_factor)
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

    @property
    def overload_limit_percent(self) -> float:
        return self.physics_config.overload_limit_percent

    @property
    def hard_overload_limit_percent(self) -> float:
        return self.physics_config.hard_overload_limit_percent

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
                reward_role="diagnostic_potential_shaping",
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
            weights=self.utility_weights,
        )
        assessment = assess_physical_state(after_state.metrics)

        return GridFMRewardBreakdown(
            reward=shaping,
            potential_shaping=shaping,
            discount_factor=self.discount_factor,
            before_potential=before_potential,
            after_potential=after_potential,
            reward_role="diagnostic_potential_shaping",
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
            weights=self.utility_weights,
        )


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

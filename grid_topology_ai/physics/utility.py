"""Canonical physical grid scoring and potential shaping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.physics.objective import (
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
    assess_physical_state,
)


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

# Preserve the established continuation-analysis ranking while keeping the
# scoring implementation and its coefficients in one authoritative module.
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
    generator_p_violation_mw: float = 0.0
    num_generator_p_violations: int = 0
    generator_q_violation_mvar: float = 0.0
    num_generator_q_violations: int = 0
    angle_difference_violation_degrees: float = 0.0
    num_angle_difference_violations: int = 0
    invalid_physical_state_flags: int = 0


def _resolved_limits(
    *,
    physics_config: PhysicsConfig | None,
    overload_limit_percent: float | None,
    hard_overload_limit_percent: float | None,
    thermal_tolerance_percent: float | None,
) -> tuple[float, float, float]:
    config = physics_config or DEFAULT_PHYSICS_CONFIG
    overload_limit = (
        config.overload_limit_percent
        if overload_limit_percent is None
        else float(overload_limit_percent)
    )
    hard_overload_limit = (
        config.hard_overload_limit_percent
        if hard_overload_limit_percent is None
        else float(hard_overload_limit_percent)
    )
    tolerance = (
        config.thermal_tolerance_percent
        if thermal_tolerance_percent is None
        else float(thermal_tolerance_percent)
    )
    if not all(
        math.isfinite(value)
        for value in (overload_limit, hard_overload_limit, tolerance)
    ):
        raise ValueError("Grid utility limits and tolerance must be finite.")
    if overload_limit < 0.0 or hard_overload_limit < overload_limit:
        raise ValueError(
            "Expected 0 <= overload_limit_percent <= hard_overload_limit_percent."
        )
    if tolerance < 0.0:
        raise ValueError("thermal_tolerance_percent must be non-negative.")
    return overload_limit, hard_overload_limit, tolerance


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
    value = float(state.metrics.get(key, 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative.")
    return value


def _nonnegative_count(state: GridFMState, key: str) -> int:
    raw = state.metrics.get(key, 0)
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
        if key not in state.metrics:
            continue
        value = state.metrics[key]
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be a bool.")
        if not bool(value):
            count += 1
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
    physics_config: PhysicsConfig | None = None,
    overload_limit_percent: float | None = None,
    hard_overload_limit_percent: float | None = None,
    thermal_tolerance_percent: float | None = None,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> GridUtilityBreakdown:
    """Build the canonical physical penalty and all of its components."""

    overload_limit, hard_limit, tolerance = _resolved_limits(
        physics_config=physics_config,
        overload_limit_percent=overload_limit_percent,
        hard_overload_limit_percent=hard_overload_limit_percent,
        thermal_tolerance_percent=thermal_tolerance_percent,
    )
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
    num_overloaded = int(state.metrics["num_overloaded_branches"])
    num_hard_overloaded = int(state.metrics["num_hard_overloaded_branches"])
    if num_overloaded < 0 or not 0 <= num_hard_overloaded <= num_overloaded:
        raise ValueError("Invalid overloaded-branch counts in state metrics.")

    voltage_violation = float(
        state.metrics.get(
            "total_voltage_violation",
            int(state.metrics.get("num_low_voltage_buses", 0))
            + int(state.metrics.get("num_high_voltage_buses", 0)),
        )
    )
    if not math.isfinite(voltage_violation) or voltage_violation < 0.0:
        raise ValueError("total_voltage_violation must be finite and non-negative.")

    max_loading = float(
        state.metrics.get(
            "max_loading_percent",
            np.max(loading) if loading.size else 0.0,
        )
    )
    if not math.isfinite(max_loading) or max_loading < 0.0:
        raise ValueError("max_loading_percent must be finite and non-negative.")
    max_loading_excess = (
        max_loading - overload_limit
        if max_loading > overload_limit + tolerance
        else 0.0
    )

    generator_p_violation_mw = _nonnegative_metric(
        state,
        "total_generator_p_violation_mw",
    )
    num_generator_p_violations = _nonnegative_count(
        state,
        "num_generator_p_violations",
    )
    generator_q_violation_mvar = _nonnegative_metric(
        state,
        "total_generator_q_violation_mvar",
    )
    num_generator_q_violations = _nonnegative_count(
        state,
        "num_generator_q_violations",
    )
    angle_difference_violation_degrees = _nonnegative_metric(
        state,
        "total_angle_difference_violation_degrees",
    )
    num_angle_difference_violations = _nonnegative_count(
        state,
        "num_angle_difference_violations",
    )
    invalid_physical_state_flags = _invalid_physical_state_flags(state)

    generator_p_violation = (
        generator_p_violation_mw + float(num_generator_p_violations)
    )
    generator_q_violation = (
        generator_q_violation_mvar + float(num_generator_q_violations)
    )
    angle_difference_violation = (
        angle_difference_violation_degrees
        + float(num_angle_difference_violations)
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
    physics_config: PhysicsConfig | None = None,
    overload_limit_percent: float | None = None,
    hard_overload_limit_percent: float | None = None,
    thermal_tolerance_percent: float | None = None,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> float:
    """Return the canonical lower-is-better grid security penalty."""

    return grid_utility_breakdown(
        state,
        physics_config=physics_config,
        overload_limit_percent=overload_limit_percent,
        hard_overload_limit_percent=hard_overload_limit_percent,
        thermal_tolerance_percent=thermal_tolerance_percent,
        weights=weights,
    ).penalty


def state_utility(
    state: GridFMState,
    *,
    physics_config: PhysicsConfig | None = None,
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
    physics_config: PhysicsConfig | None = None,
    overload_limit_percent: float | None = None,
    hard_overload_limit_percent: float | None = None,
    thermal_tolerance_percent: float | None = None,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> float:
    """Return the higher-is-better potential ``Phi(s) = -penalty(s)``."""

    return -state_security_penalty(
        state,
        physics_config=physics_config,
        overload_limit_percent=overload_limit_percent,
        hard_overload_limit_percent=hard_overload_limit_percent,
        thermal_tolerance_percent=thermal_tolerance_percent,
        weights=weights,
    )


def potential_shaping_reward(
    before_state: GridFMState,
    after_state: GridFMState,
    *,
    discount_factor: float,
    physics_config: PhysicsConfig | None = None,
    overload_limit_percent: float | None = None,
    hard_overload_limit_percent: float | None = None,
    thermal_tolerance_percent: float | None = None,
    weights: GridUtilityWeights = DEFAULT_GRID_UTILITY_WEIGHTS,
) -> float:
    """Return policy-invariant potential shaping ``gamma*Phi(s') - Phi(s)``."""

    gamma = _require_discount_factor(discount_factor)
    before_potential = state_potential(
        before_state,
        physics_config=physics_config,
        overload_limit_percent=overload_limit_percent,
        hard_overload_limit_percent=hard_overload_limit_percent,
        thermal_tolerance_percent=thermal_tolerance_percent,
        weights=weights,
    )
    after_potential = state_potential(
        after_state,
        physics_config=physics_config,
        overload_limit_percent=overload_limit_percent,
        hard_overload_limit_percent=hard_overload_limit_percent,
        thermal_tolerance_percent=thermal_tolerance_percent,
        weights=weights,
    )
    shaping = gamma * after_potential - before_potential
    if not math.isfinite(shaping):
        raise ValueError("Potential shaping reward must be finite.")
    return float(shaping)


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

    The optimized return is defined in :mod:`grid_topology_ai.value_targets`.
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

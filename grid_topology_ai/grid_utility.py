"""Canonical physical grid scoring and potential shaping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState


@dataclass(frozen=True, slots=True)
class GridUtilityWeights:
    """Weights for one lower-is-better physical security penalty."""

    total_overload: float = 2.0
    hard_overload: float = 5.0
    num_overloaded: float = 10.0
    num_hard_overloaded: float = 30.0
    voltage_violation: float = 500.0
    max_loading_excess: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("total_overload", self.total_overload),
            ("hard_overload", self.hard_overload),
            ("num_overloaded", self.num_overloaded),
            ("num_hard_overloaded", self.num_hard_overloaded),
            ("voltage_violation", self.voltage_violation),
            ("max_loading_excess", self.max_loading_excess),
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

    penalty = (
        weights.total_overload * total_overload
        + weights.hard_overload * total_hard_overload
        + weights.num_overloaded * num_overloaded
        + weights.num_hard_overloaded * num_hard_overloaded
        + weights.voltage_violation * voltage_violation
        + weights.max_loading_excess * max_loading_excess
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

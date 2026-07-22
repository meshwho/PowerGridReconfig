from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.grid_utility import (
    CONTINUATION_GRID_UTILITY_WEIGHTS,
    CONTINUATION_SWITCH_PENALTY,
    GridUtilityWeights,
    grid_utility_breakdown,
    potential_shaping_reward,
    state_potential,
    state_security_penalty,
)
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.continuation_gate import topology_penalty


def _state(
    loadings: list[float],
    statuses: list[float],
    *,
    num_overloaded: int,
    num_hard: int,
    voltage_violation: float,
    max_loading: float | None = None,
) -> GridFMState:
    branch_features = np.zeros(
        (len(loadings), len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = loadings
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = statuses
    active = [
        loading
        for loading, status in zip(loadings, statuses, strict=True)
        if status > 0.0
    ]
    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=np.zeros((2, 1), dtype=np.float32),
        branch_features=branch_features,
        edge_index=np.zeros((2, len(loadings)), dtype=np.int64),
        branch_ids=np.arange(len(loadings), dtype=np.int64),
        branch_status=np.asarray(statuses, dtype=np.float32),
        metrics={
            "num_overloaded_branches": num_overloaded,
            "num_hard_overloaded_branches": num_hard,
            "max_loading_percent": (
                max(active, default=0.0)
                if max_loading is None
                else max_loading
            ),
            "total_voltage_violation": voltage_violation,
        },
        outaged_branch_ids=[],
    )


def test_default_grid_utility_has_auditable_components() -> None:
    state = _state(
        [80.0, 110.0, 130.0],
        [1.0, 1.0, 1.0],
        num_overloaded=2,
        num_hard=1,
        voltage_violation=0.1,
    )
    breakdown = grid_utility_breakdown(state)

    assert breakdown.total_overload == pytest.approx(40.0)
    assert breakdown.total_hard_overload == pytest.approx(10.0)
    assert breakdown.num_overloaded == 2
    assert breakdown.num_hard_overloaded == 1
    assert breakdown.voltage_violation == pytest.approx(0.1)
    assert breakdown.penalty == pytest.approx(230.0)
    assert state_security_penalty(state) == pytest.approx(230.0)
    assert state_potential(state) == pytest.approx(-230.0)


def test_reward_uses_the_default_grid_utility() -> None:
    physics = PhysicsConfig(
        overload_limit_percent=115.0,
        hard_overload_limit_percent=135.0,
        thermal_tolerance_percent=0.0,
    )
    state = _state(
        [130.0, 105.0, 200.0],
        [1.0, 1.0, 0.0],
        num_overloaded=1,
        num_hard=0,
        voltage_violation=0.2,
        max_loading=130.0,
    )
    canonical = state_security_penalty(state, physics_config=physics)

    assert canonical == pytest.approx(140.0)
    assert GridFMReward(physics_config=physics)._state_penalty(state) == pytest.approx(
        canonical
    )


def test_potential_shaping_uses_exact_discounted_difference() -> None:
    before = _state(
        [130.0],
        [1.0],
        num_overloaded=1,
        num_hard=1,
        voltage_violation=0.0,
    )
    after = _state(
        [110.0],
        [1.0],
        num_overloaded=1,
        num_hard=0,
        voltage_violation=0.0,
    )

    assert potential_shaping_reward(
        before,
        after,
        discount_factor=0.95,
    ) == pytest.approx(0.95 * state_potential(after) - state_potential(before))


def test_discounted_potential_shaping_telescopes() -> None:
    states = [
        _state([130.0], [1.0], num_overloaded=1, num_hard=1, voltage_violation=0.0),
        _state([110.0], [1.0], num_overloaded=1, num_hard=0, voltage_violation=0.0),
        _state([90.0], [1.0], num_overloaded=0, num_hard=0, voltage_violation=0.0),
    ]
    gamma = 0.9
    rewards = [
        potential_shaping_reward(
            states[index],
            states[index + 1],
            discount_factor=gamma,
        )
        for index in range(len(states) - 1)
    ]
    discounted_sum = sum(
        gamma**index * value
        for index, value in enumerate(rewards)
    )

    assert discounted_sum == pytest.approx(
        gamma ** (len(states) - 1) * state_potential(states[-1])
        - state_potential(states[0])
    )


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("nan"), True])
def test_potential_shaping_rejects_invalid_discount(gamma: object) -> None:
    state = _state(
        [100.0],
        [1.0],
        num_overloaded=0,
        num_hard=0,
        voltage_violation=0.0,
    )
    with pytest.raises(ValueError, match="discount_factor"):
        potential_shaping_reward(state, state, discount_factor=gamma)  # type: ignore[arg-type]


def test_continuation_scoring_uses_named_shared_weights() -> None:
    physics = PhysicsConfig(
        overload_limit_percent=115.0,
        hard_overload_limit_percent=135.0,
        thermal_tolerance_percent=0.0,
    )
    state = _state(
        [130.0, 105.0, 200.0],
        [1.0, 1.0, 0.0],
        num_overloaded=1,
        num_hard=0,
        voltage_violation=0.2,
        max_loading=130.0,
    )
    base = state_security_penalty(
        state,
        physics_config=physics,
        weights=CONTINUATION_GRID_UTILITY_WEIGHTS,
    )

    assert base == pytest.approx(315.0)
    assert topology_penalty(state, physics_config=physics) == pytest.approx(base)
    assert topology_penalty(
        state,
        depth=2,
        physics_config=physics,
    ) == pytest.approx(base + 2 * CONTINUATION_SWITCH_PENALTY)


def test_tolerance_and_branch_status_are_applied_once() -> None:
    physics = PhysicsConfig(
        overload_limit_percent=100.0,
        hard_overload_limit_percent=120.0,
        thermal_tolerance_percent=0.01,
    )
    state = _state(
        [100.005, 150.0],
        [1.0, 0.0],
        num_overloaded=0,
        num_hard=0,
        voltage_violation=0.0,
        max_loading=100.005,
    )
    breakdown = grid_utility_breakdown(state, physics_config=physics)

    assert breakdown.total_overload == 0.0
    assert breakdown.total_hard_overload == 0.0
    assert breakdown.max_loading_excess == 0.0
    assert breakdown.penalty == 0.0


def test_grid_utility_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        GridUtilityWeights(total_overload=-1.0)


def test_shaping_is_separate_from_mcts_backup() -> None:
    root = Path(__file__).resolve().parents[1] / "grid_topology_ai"
    reward = (root / "reward.py").read_text(encoding="utf-8")
    mcts = (root / "search/mcts.py").read_text(encoding="utf-8")
    backup = mcts.split("def _backup", 1)[1].split("def _leaf_value", 1)[0]

    assert "potential_shaping_reward" in reward
    assert "scaled_reward" not in backup
    assert "_scale_value(" not in backup
    assert "node.total_value += value" in backup

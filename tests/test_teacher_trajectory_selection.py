from __future__ import annotations

from dataclasses import dataclass

import pytest

from grid_topology_ai.search.teacher import (
    ImpactBeamSearchConfig,
    pareto_front,
    select_epsilon_optimal_trajectory,
    switch_count,
    update_pareto_archive,
)


@dataclass
class FakeNode:
    safety_score: float
    branch_ids: list[int | None]
    action_ids: list[int]
    solved: bool = False
    discounted_score: float = 0.0
    num_hard_overloaded: int = 0
    history: tuple[float, ...] = ()


def node(
    safety: float,
    switches: int,
    *,
    solved: bool = False,
    hard: int = 0,
    history: tuple[float, ...] = (),
) -> FakeNode:
    return FakeNode(
        safety_score=float(safety),
        branch_ids=list(range(1, switches + 1)),
        action_ids=list(range(1, switches + 1)),
        solved=solved,
        num_hard_overloaded=hard,
        history=history,
    )


def test_scenario_61_selects_three_switch_prefix_at_one_percent_epsilon() -> None:
    root = node(181.276, 0)
    candidates = [
        root,
        node(104.293, 1),
        node(47.254, 2),
        node(36.511, 3),
        node(35.796, 4),
        node(35.563, 5),
    ]

    result = select_epsilon_optimal_trajectory(
        root,
        candidates,
        relative_physical_epsilon=0.01,
        max_hard_overloaded=0,
    )

    assert result.selected_switch_count == 3
    assert result.selected_safety == pytest.approx(36.511)
    assert result.best_physical_safety == pytest.approx(35.563)
    assert result.retained_improvement_fraction == pytest.approx(0.99349, abs=1e-5)


def test_zero_epsilon_recovers_exact_physical_minimum() -> None:
    root = node(181.276, 0)
    candidates = [root, node(36.511, 3), node(35.796, 4), node(35.563, 5)]

    result = select_epsilon_optimal_trajectory(
        root,
        candidates,
        relative_physical_epsilon=0.0,
        max_hard_overloaded=0,
    )

    assert result.selected_switch_count == 5
    assert result.selected_safety == pytest.approx(35.563)


def test_selection_is_invariant_to_positive_rescaling_of_physical_penalty() -> None:
    root = node(181.276, 0)
    candidates = [
        root,
        node(104.293, 1),
        node(47.254, 2),
        node(36.511, 3),
        node(35.796, 4),
        node(35.563, 5),
    ]
    scaled_root = node(1812.76, 0)
    scaled_candidates = [
        scaled_root,
        node(1042.93, 1),
        node(472.54, 2),
        node(365.11, 3),
        node(357.96, 4),
        node(355.63, 5),
    ]

    original = select_epsilon_optimal_trajectory(
        root,
        candidates,
        relative_physical_epsilon=0.01,
        max_hard_overloaded=0,
    )
    scaled = select_epsilon_optimal_trajectory(
        scaled_root,
        scaled_candidates,
        relative_physical_epsilon=0.01,
        max_hard_overloaded=0,
    )

    assert original.selected_switch_count == scaled.selected_switch_count == 3


def test_shortest_strictly_solved_trajectory_has_priority() -> None:
    root = node(100.0, 0)
    candidates = [
        root,
        node(4.0, 2),
        node(0.0, 3, solved=True),
        node(0.0, 5, solved=True),
    ]

    result = select_epsilon_optimal_trajectory(
        root,
        candidates,
        relative_physical_epsilon=0.01,
        max_hard_overloaded=0,
    )

    assert result.node.solved is True
    assert result.selected_switch_count == 3
    assert result.selected_safety == 0.0


def test_temporary_intermediate_worsening_does_not_disqualify_final_trajectory() -> None:
    root = node(300.0, 0)
    temporary_worsening = node(
        100.0,
        3,
        history=(300.0, 350.0, 170.0, 100.0),
    )
    candidates = [root, node(190.0, 1), temporary_worsening, node(95.0, 5)]

    result = select_epsilon_optimal_trajectory(
        root,
        candidates,
        relative_physical_epsilon=0.03,
        max_hard_overloaded=0,
    )

    assert result.node is temporary_worsening
    assert result.node.history[1] > result.node.history[0]


def test_pareto_archive_preserves_useful_shallower_prefixes() -> None:
    root = node(181.276, 0)
    archive = [root]

    for candidate in (
        node(104.293, 1),
        node(47.254, 2),
        node(36.511, 3),
        node(35.796, 4),
        node(35.563, 5),
    ):
        archive = update_pareto_archive(
            archive,
            [candidate],
            max_hard_overloaded=0,
        )

    assert [switch_count(item) for item in archive] == [0, 1, 2, 3, 4, 5]
    assert any(
        switch_count(item) == 3 and item.safety_score == pytest.approx(36.511)
        for item in archive
    )


def test_pareto_front_removes_dominated_trajectory() -> None:
    candidates = [
        node(100.0, 0),
        node(80.0, 2),
        node(90.0, 3),
        node(50.0, 3),
    ]

    front = pareto_front(candidates, max_hard_overloaded=0)
    objective_pairs = {(switch_count(item), item.safety_score) for item in front}

    assert (3, 90.0) not in objective_pairs
    assert (2, 80.0) in objective_pairs
    assert (3, 50.0) in objective_pairs


def test_stop_action_is_not_counted_as_physical_switch() -> None:
    item = FakeNode(
        safety_score=10.0,
        branch_ids=[10, 20, None],
        action_ids=[11, 21, 0],
    )
    assert switch_count(item) == 2


def test_final_selection_excludes_hard_overload_regression() -> None:
    root = node(100.0, 0, hard=1)
    candidates = [root, node(40.0, 2, hard=1), node(1.0, 3, hard=2)]

    result = select_epsilon_optimal_trajectory(
        root,
        candidates,
        relative_physical_epsilon=0.01,
        max_hard_overloaded=1,
    )

    assert result.selected_safety == pytest.approx(40.0)
    assert result.selected_switch_count == 2


def test_config_rejects_invalid_relative_physical_epsilon() -> None:
    with pytest.raises(ValueError, match="relative_physical_epsilon"):
        ImpactBeamSearchConfig(relative_physical_epsilon=1.0)

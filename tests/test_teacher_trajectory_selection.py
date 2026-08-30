from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from grid_topology_ai.search.teacher import (
    ImpactBeamSearchConfig,
    ImpactBeamSearchPlanner,
    switch_count,
    update_top_j_candidate_archive,
)


@dataclass
class FakeNode:
    safety_score: float
    branch_ids: list[int | None]
    action_ids: list[int]
    solved: bool = False
    discounted_score: float = 0.0
    num_hard_overloaded: int = 0


def node(
    safety: float,
    action_ids: list[int],
    *,
    discounted_score: float = 0.0,
) -> FakeNode:
    return FakeNode(
        safety_score=float(safety),
        branch_ids=[int(action_id) for action_id in action_ids],
        action_ids=[int(action_id) for action_id in action_ids],
        discounted_score=float(discounted_score),
    )


def test_archive_keeps_top_j_independently_per_switch_count() -> None:
    root = node(100.0, [])
    one_switch_best = node(20.0, [11])
    one_switch_second = node(25.0, [12])
    one_switch_pruned = node(30.0, [13])
    two_switch_best = node(10.0, [21, 22])
    two_switch_second = node(15.0, [23, 24])
    two_switch_pruned = node(18.0, [25, 26])

    archive = update_top_j_candidate_archive(
        {},
        [
            root,
            one_switch_pruned,
            two_switch_pruned,
            one_switch_second,
            two_switch_second,
            one_switch_best,
            two_switch_best,
        ],
        per_switch_count=2,
    )

    assert [item.action_ids for item in archive[0]] == [[]]
    assert [item.action_ids for item in archive[1]] == [[11], [12]]
    assert [item.action_ids for item in archive[2]] == [[21, 22], [23, 24]]


def test_archive_deduplicates_by_action_sequence_and_keeps_better_j() -> None:
    worse_duplicate = node(40.0, [11, 12], discounted_score=5.0)
    better_duplicate = node(30.0, [11, 12], discounted_score=1.0)
    other = node(35.0, [13, 14])

    archive = update_top_j_candidate_archive(
        {},
        [worse_duplicate, other, better_duplicate],
        per_switch_count=5,
    )

    bucket = archive[2]
    assert len(bucket) == 2
    assert bucket[0] is better_duplicate
    assert bucket[1] is other


def test_archive_does_not_apply_cross_switch_topology_pareto_pruning() -> None:
    root = node(50.0, [])
    one_switch = node(40.0, [11])
    two_switch_worse_j = node(45.0, [21, 22])

    archive = update_top_j_candidate_archive(
        {},
        [root, one_switch, two_switch_worse_j],
        per_switch_count=5,
    )

    assert archive[0] == [root]
    assert archive[1] == [one_switch]
    assert archive[2] == [two_switch_worse_j]


def test_switch_count_ignores_terminal_handoff_action() -> None:
    item = FakeNode(
        safety_score=10.0,
        branch_ids=[10, 20, None],
        action_ids=[11, 21, 0],
    )

    assert switch_count(item) == 2


def test_beam_candidate_generation_excludes_stop_action() -> None:
    stop = SimpleNamespace(kind="stop", action_id=0)
    first = SimpleNamespace(kind="set_branch_status", action_id=1)
    second = SimpleNamespace(kind="set_branch_status", action_id=2)

    class FakeActionSpace:
        @staticmethod
        def loading_priority(state, action):
            del state
            return {1: 80.0, 2: 90.0}[int(action.action_id)]

    env = SimpleNamespace(
        current_state=object(),
        valid_actions=lambda: [stop, first, second],
        action_space=FakeActionSpace(),
    )
    planner = ImpactBeamSearchPlanner(
        ImpactBeamSearchConfig(candidate_pool_size=10)
    )

    candidates = planner._candidate_actions(env)

    assert [int(action.action_id) for action in candidates] == [2, 1]
    assert all(action.kind != "stop" for action in candidates)


def test_beam_config_no_longer_exposes_stop_decision_flag() -> None:
    with pytest.raises(TypeError, match="include_stop_action"):
        ImpactBeamSearchConfig(include_stop_action=True)

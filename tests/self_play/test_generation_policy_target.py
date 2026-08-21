from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.self_play.generation import _select_generation_action


def _result(policy: dict[int, float]) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        root=SimpleNamespace(actions_by_id={
            1: SimpleNamespace(branch_id=11),
            2: SimpleNamespace(branch_id=22),
        }),
    )


def test_greedy_action_is_in_saved_policy_support() -> None:
    decision = _select_generation_action(
        search_result=_result({1: 0.7, 2: 0.3}),
        temperature=0.0,
        rng=np.random.default_rng(1),
        scenario_id=8,
        step=0,
    )
    assert decision.selected_action_id == 1
    assert decision.selected_branch_id == 11
    assert decision.policy_target == {1: 1.0}
    assert decision.policy_target[decision.selected_action_id] > 0.0


def test_sampled_action_is_in_saved_policy_support() -> None:
    decision = _select_generation_action(
        search_result=_result({1: 0.25, 2: 0.75}),
        temperature=1.0,
        rng=np.random.default_rng(7),
    )
    assert decision.policy_target == pytest.approx({1: 0.25, 2: 0.75})
    assert decision.policy_target[decision.selected_action_id] > 0.0
    assert decision.selected_branch_id in {11, 22}


def test_selected_action_must_exist_in_root_actions() -> None:
    result = _result({2: 1.0})
    result.root.actions_by_id.pop(2)
    with pytest.raises(RuntimeError, match="missing from root.actions_by_id"):
        _select_generation_action(
            search_result=result,
            temperature=0.0,
            rng=np.random.default_rng(1),
        )


def test_stop_action_has_no_branch() -> None:
    decision = _select_generation_action(
        search_result=SimpleNamespace(
            policy={0: 1.0}, root=SimpleNamespace(actions_by_id={})
        ),
        temperature=0.0,
        rng=np.random.default_rng(1),
    )
    assert decision.selected_action_id == 0
    assert decision.selected_branch_id is None

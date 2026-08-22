from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.search.screening import (
    DCActionScore,
    DCActionScreener,
)
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner


_LOADING_INDEX = BRANCH_FEATURE_COLUMNS.index("loading_percent")


class _FakeEnv:
    def __init__(
        self,
        actions: list[GridFMAction],
        loadings: list[float],
    ) -> None:
        branch_features = np.zeros(
            (len(loadings), len(BRANCH_FEATURE_COLUMNS)),
            dtype=float,
        )
        branch_features[:, _LOADING_INDEX] = np.asarray(
            loadings,
            dtype=float,
        )

        self.current_state = SimpleNamespace(
            branch_features=branch_features,
            metrics={},
        )
        self.backend = object()
        self.done = False
        self.solved = False
        self._actions = list(actions)

    def valid_actions(self) -> list[GridFMAction]:
        return list(self._actions)

    def operational_action_mask(self) -> np.ndarray:
        size = max(action.action_id for action in self._actions) + 1
        mask = np.zeros(size, dtype=bool)

        for action in self._actions:
            mask[action.action_id] = True

        return mask


class _Evaluator:
    def __init__(self, policy: np.ndarray) -> None:
        self.policy = np.asarray(policy, dtype=float)

    def evaluate(
        self,
        *,
        state: object,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        del state, action_mask
        return self.policy.copy(), 0.0


class _DCRanker:
    def __init__(self) -> None:
        self.seen_action_ids: list[int] = []

    def rank_actions(
        self,
        *,
        state: object,
        actions: list[GridFMAction],
        backend: object,
        neural_policy: np.ndarray | None = None,
    ) -> list[GridFMAction]:
        del state, backend, neural_policy
        self.seen_action_ids = [action.action_id for action in actions]
        return list(reversed(actions))


def _switch_actions(count: int) -> list[GridFMAction]:
    return [
        GridFMAction(
            action_id=index + 1,
            action_type="switch_off_branch",
            branch_id=100 + index,
            branch_pos=index,
        )
        for index in range(count)
    ]


def _action_ids(actions: list[GridFMAction]) -> list[int]:
    return [int(action.action_id) for action in actions]


def _expand(
    *,
    actions: list[GridFMAction],
    loadings: list[float],
    config: MCTSConfig,
    policy: np.ndarray | None = None,
    include_stop: bool = False,
    dc_ranker: _DCRanker | None = None,
) -> tuple[MCTSPlanner, MCTSNode]:
    evaluator = None if policy is None else _Evaluator(policy)

    # These tests cover ranking and initial shortlist construction only.
    config = replace(config, exploration_quota=0)
    planner = MCTSPlanner(config, evaluator=evaluator)  # type: ignore[arg-type]
    planner._should_include_stop_action = (  # type: ignore[method-assign]
        lambda state: include_stop
    )

    if dc_ranker is not None:
        planner.dc_screener = dc_ranker  # type: ignore[assignment]

    node = MCTSNode(
        env=_FakeEnv(actions, loadings),  # type: ignore[arg-type]
        depth=0,
    )
    planner._expand_node(node)
    return planner, node


def test_loading_ranking_keeps_every_legal_switch() -> None:
    actions = _switch_actions(10)
    loadings = [10, 90, 20, 80, 30, 70, 40, 60, 50, 100]

    _, node = _expand(
        actions=actions,
        loadings=loadings,
        config=MCTSConfig(top_k_actions=3),
    )

    expected = [10, 2, 4, 6, 8, 9, 7, 5, 3, 1]

    assert _action_ids(node.ranked_actions) == expected
    assert list(node.actions_by_id) == expected[:3]
    assert list(node.action_priors) == expected[:3]
    assert set(node.action_scores) == set(expected)
    assert sum(node.action_priors.values()) == pytest.approx(1.0)


def test_neural_ranking_retains_low_prior_tail() -> None:
    actions = _switch_actions(10)
    policy = np.zeros(11, dtype=float)
    policy[1:] = np.linspace(1.0, 0.1, 10)

    _, node = _expand(
        actions=actions,
        loadings=[100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        config=MCTSConfig(top_k_actions=3),
        policy=policy,
    )

    # The legacy loading backup extends the active neural top-3 to five
    # actions. The remaining low-prior actions stay queued for widening.
    assert _action_ids(node.ranked_actions) == list(range(1, 11))
    assert list(node.actions_by_id) == [1, 2, 3, 4, 5]
    assert 10 in node.action_scores
    assert 10 not in node.action_priors
    assert len(node.ranked_actions) == len(
        {action.action_id for action in node.ranked_actions}
    )
    assert sum(node.action_priors.values()) == pytest.approx(1.0)


def test_loading_backup_does_not_duplicate_neural_actions() -> None:
    actions = _switch_actions(8)
    policy = np.zeros(9, dtype=float)
    policy[1:] = np.linspace(1.0, 0.2, 8)

    _, node = _expand(
        actions=actions,
        loadings=[80, 90, 100, 70, 60, 50, 40, 30],
        config=MCTSConfig(top_k_actions=3),
        policy=policy,
    )

    assert list(node.actions_by_id) == [1, 2, 3, 4, 5]
    assert len(node.actions_by_id) == len(set(node.actions_by_id))
    assert len(node.ranked_actions) == 8


def test_dc_candidate_pool_does_not_prune_the_legal_tail() -> None:
    actions = _switch_actions(10)
    policy = np.zeros(11, dtype=float)
    policy[1:] = np.linspace(1.0, 0.1, 10)
    ranker = _DCRanker()

    _, node = _expand(
        actions=actions,
        loadings=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        config=MCTSConfig(
            top_k_actions=3,
            use_dc_screening=True,
            dc_top_k_actions=2,
            dc_candidate_pool=4,
            dc_keep_policy_actions=1,
            dc_keep_loading_actions=1,
        ),
        policy=policy,
        dc_ranker=ranker,
    )

    assert ranker.seen_action_ids == [1, 2, 3, 4, 10]
    assert list(node.actions_by_id) == [10, 4, 1]
    assert set(_action_ids(node.ranked_actions)) == set(range(1, 11))
    assert 9 in node.action_scores
    assert 9 not in node.action_priors


def test_dc_ranker_returns_full_order_and_screen_keeps_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _switch_actions(3)
    actions.append(
        GridFMAction(
            action_id=0,
            action_type="do_nothing",
        )
    )
    penalties = {1: 3.0, 2: 1.0, 3: 2.0}
    failures = {3}
    screener = DCActionScreener(top_k=2)

    def fake_score_action(
        *,
        state: object,
        action: GridFMAction,
        backend: object,
        neural_policy: np.ndarray | None = None,
    ) -> DCActionScore:
        del state, backend, neural_policy
        failed = action.action_id in failures
        return DCActionScore(
            action=action,
            success=not failed,
            penalty=penalties[action.action_id],
            max_loading_percent=100.0,
            num_overloaded=0,
            num_hard_overloaded=0,
            total_overload=0.0,
            hard_overload=0.0,
            policy_prior=0.0,
        )

    monkeypatch.setattr(
        screener,
        "score_action",
        fake_score_action,
    )

    ranked = screener.rank_actions(
        state=object(),  # type: ignore[arg-type]
        actions=actions,
        backend=object(),  # type: ignore[arg-type]
    )
    screened = screener.screen_actions(
        state=object(),  # type: ignore[arg-type]
        actions=actions,
        backend=object(),  # type: ignore[arg-type]
    )
    unbounded = screener.screen_actions(
        state=object(),  # type: ignore[arg-type]
        actions=actions,
        backend=object(),  # type: ignore[arg-type]
        top_k=0,
    )

    assert _action_ids(ranked) == [2, 1, 3]
    assert _action_ids(screened) == [2, 1]
    assert _action_ids(unbounded) == [2, 1, 3]


def test_stop_action_follows_search_stop_policy() -> None:
    stop = GridFMAction(
        action_id=0,
        action_type="do_nothing",
    )
    switches = _switch_actions(3)
    actions = [stop, *switches]

    _, allowed = _expand(
        actions=actions,
        loadings=[30, 20, 10],
        config=MCTSConfig(top_k_actions=2),
        include_stop=True,
    )
    _, blocked = _expand(
        actions=actions,
        loadings=[30, 20, 10],
        config=MCTSConfig(top_k_actions=2),
        include_stop=False,
    )

    assert _action_ids(allowed.ranked_actions) == [0, 1, 2, 3]
    assert list(allowed.actions_by_id) == [0, 1, 2]
    assert 0 not in blocked.action_scores
    assert 0 not in blocked.action_priors


def test_discarded_action_is_removed_from_active_and_future_sets() -> None:
    actions = _switch_actions(3)
    child = MCTSNode(
        env=SimpleNamespace(done=False, solved=False),  # type: ignore[arg-type]
        depth=1,
    )
    node = MCTSNode(
        env=SimpleNamespace(done=False, solved=False),  # type: ignore[arg-type]
        depth=0,
        ranked_actions=list(actions),
        action_scores={1: 1.0, 2: 0.5, 3: 0.25},
        action_priors={1: 0.5, 2: 0.3, 3: 0.2},
        actions_by_id={action.action_id: action for action in actions},
        children={2: child},
    )

    MCTSPlanner._discard_action(node, 2)

    assert _action_ids(node.ranked_actions) == [1, 3]
    assert 2 not in node.action_scores
    assert 2 not in node.action_priors
    assert 2 not in node.actions_by_id
    assert 2 not in node.children

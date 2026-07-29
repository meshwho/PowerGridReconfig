from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.search.mcts import (
    MCTSConfig,
    MCTSNode,
    MCTSPlanner,
)


_LOADING_INDEX = BRANCH_FEATURE_COLUMNS.index(
    "loading_percent"
)


class _RecordingEvaluator:
    def __init__(
        self,
        policy: np.ndarray,
    ) -> None:
        self.policy = np.asarray(
            policy,
            dtype=float,
        )
        self.seen_masks: list[np.ndarray] = []

    def evaluate(
        self,
        *,
        state: object,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        del state
        self.seen_masks.append(
            np.asarray(
                action_mask,
                dtype=bool,
            ).copy()
        )
        return self.policy.copy(), 0.0


class _FakeEnv:
    def __init__(
        self,
        *,
        actions: list[GridFMAction],
        operational_mask: np.ndarray,
        loadings: list[float],
    ) -> None:
        branch_features = np.zeros(
            (
                len(loadings),
                len(BRANCH_FEATURE_COLUMNS),
            ),
            dtype=float,
        )
        branch_features[
            :,
            _LOADING_INDEX,
        ] = np.asarray(
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
        self._operational_mask = np.asarray(
            operational_mask,
            dtype=bool,
        ).copy()

    def valid_actions(self) -> list[GridFMAction]:
        return list(self._actions)

    def operational_action_mask(self) -> np.ndarray:
        return self._operational_mask.copy()


def _actions() -> list[GridFMAction]:
    return [
        GridFMAction(
            action_id=0,
            action_type="do_nothing",
        ),
        GridFMAction(
            action_id=1,
            action_type="switch_off_branch",
            branch_id=101,
            branch_pos=0,
        ),
        GridFMAction(
            action_id=2,
            action_type="switch_off_branch",
            branch_id=102,
            branch_pos=1,
        ),
    ]


def _expand(
    *,
    include_stop: bool,
    operational_mask: list[bool],
) -> tuple[
    MCTSPlanner,
    MCTSNode,
    _RecordingEvaluator,
    _FakeEnv,
]:
    evaluator = _RecordingEvaluator(
        np.asarray(
            [0.8, 0.15, 0.05],
            dtype=float,
        )
    )
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=2,
            exploration_quota=0,
        ),
        evaluator=evaluator,  # type: ignore[arg-type]
    )
    planner._should_include_stop_action = (  # type: ignore[method-assign]
        lambda state: include_stop
    )

    env = _FakeEnv(
        actions=_actions(),
        operational_mask=np.asarray(
            operational_mask,
            dtype=bool,
        ),
        loadings=[90.0, 80.0],
    )
    node = MCTSNode(
        env=env,  # type: ignore[arg-type]
        depth=0,
    )

    planner._expand_node(node)

    return planner, node, evaluator, env


def test_mcts_action_mask_applies_stop_policy_to_a_copy() -> None:
    planner = MCTSPlanner(MCTSConfig())
    planner._should_include_stop_action = (  # type: ignore[method-assign]
        lambda state: False
    )
    original = np.asarray(
        [True, True, False],
        dtype=bool,
    )

    result = planner._mcts_action_mask(
        state=SimpleNamespace(),  # type: ignore[arg-type]
        operational_mask=original,
    )

    assert original.tolist() == [
        True,
        True,
        False,
    ]
    assert result.tolist() == [
        False,
        True,
        False,
    ]
    assert result is not original


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (
            np.asarray([], dtype=bool),
            "must contain the stop action",
        ),
        (
            np.asarray([[True, True]], dtype=bool),
            "must be one-dimensional",
        ),
    ],
)
def test_mcts_action_mask_rejects_invalid_shape(
    mask: np.ndarray,
    message: str,
) -> None:
    planner = MCTSPlanner(MCTSConfig())

    with pytest.raises(
        ValueError,
        match=message,
    ):
        planner._mcts_action_mask(
            state=SimpleNamespace(),  # type: ignore[arg-type]
            operational_mask=mask,
        )


def test_blocked_stop_is_removed_before_neural_evaluation() -> None:
    _, node, evaluator, env = _expand(
        include_stop=False,
        operational_mask=[True, True, True],
    )

    assert env.operational_action_mask().tolist() == [
        True,
        True,
        True,
    ]
    assert len(evaluator.seen_masks) == 1
    assert evaluator.seen_masks[0].tolist() == [
        False,
        True,
        True,
    ]
    assert [
        action.action_id
        for action in node.ranked_actions
    ] == [1, 2]
    assert list(node.actions_by_id) == [1, 2]
    assert 0 not in node.action_scores
    assert 0 not in node.action_priors


def test_allowed_stop_is_visible_to_evaluator_and_search() -> None:
    _, node, evaluator, _ = _expand(
        include_stop=True,
        operational_mask=[True, True, True],
    )

    assert len(evaluator.seen_masks) == 1
    assert evaluator.seen_masks[0].tolist() == [
        True,
        True,
        True,
    ]
    assert [
        action.action_id
        for action in node.ranked_actions
    ] == [0, 1, 2]
    assert list(node.actions_by_id) == [0, 1, 2]
    assert 0 in node.action_scores
    assert 0 in node.action_priors


def test_final_mask_preserves_operational_switch_filter() -> None:
    _, node, evaluator, _ = _expand(
        include_stop=False,
        operational_mask=[True, True, False],
    )

    assert evaluator.seen_masks[0].tolist() == [
        False,
        True,
        False,
    ]
    assert [
        action.action_id
        for action in node.ranked_actions
    ] == [1]
    assert list(node.actions_by_id) == [1]
    assert 2 not in node.action_scores


def test_expand_rejects_action_id_outside_final_mask() -> None:
    invalid_action = GridFMAction(
        action_id=3,
        action_type="switch_off_branch",
        branch_id=103,
        branch_pos=0,
    )
    env = _FakeEnv(
        actions=[invalid_action],
        operational_mask=np.asarray(
            [True, True, True],
            dtype=bool,
        ),
        loadings=[90.0],
    )
    planner = MCTSPlanner(
        MCTSConfig(
            exploration_quota=0,
        )
    )
    planner._should_include_stop_action = (  # type: ignore[method-assign]
        lambda state: False
    )
    node = MCTSNode(
        env=env,  # type: ignore[arg-type]
        depth=0,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "action_id 3 outside action mask of size 3"
        ),
    ):
        planner._expand_node(node)

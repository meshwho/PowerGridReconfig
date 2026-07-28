from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.evaluation.checkpoint import EvaluationRequest
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner


_LOADING_INDEX = BRANCH_FEATURE_COLUMNS.index("loading_percent")


class _FakeEnv:
    def __init__(
        self,
        *,
        actions: list[GridFMAction],
        depth: int,
    ) -> None:
        branch_features = np.zeros(
            (len(actions), len(BRANCH_FEATURE_COLUMNS)),
            dtype=float,
        )
        branch_features[:, _LOADING_INDEX] = np.linspace(
            100.0,
            80.0,
            len(actions),
        )

        self.current_state = SimpleNamespace(
            branch_features=branch_features,
            metrics={},
        )
        self.backend = SimpleNamespace(node_depth=depth)
        self.done = False
        self.solved = False
        self._actions = list(actions)

    def valid_actions(self) -> list[GridFMAction]:
        return list(self._actions)

    def valid_action_mask(self) -> np.ndarray:
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


class _RecordingDCRanker:
    def __init__(self) -> None:
        self.called_depths: list[int] = []

    def rank_actions(
        self,
        *,
        state: object,
        actions: list[GridFMAction],
        backend: object,
        neural_policy: np.ndarray | None = None,
    ) -> list[GridFMAction]:
        del state, neural_policy
        self.called_depths.append(int(backend.node_depth))
        return list(reversed(actions))


def _switch_actions(count: int = 5) -> list[GridFMAction]:
    return [
        GridFMAction(
            action_id=index + 1,
            action_type="switch_off_branch",
            branch_id=100 + index,
            branch_pos=index,
        )
        for index in range(count)
    ]


def _dc_calls(
    *,
    dc_max_depth: int,
    node_depths: tuple[int, ...],
    use_dc_screening: bool = True,
) -> list[int]:
    actions = _switch_actions()
    policy = np.zeros(len(actions) + 1, dtype=float)
    policy[1:] = np.linspace(1.0, 0.2, len(actions))

    planner = MCTSPlanner(
        MCTSConfig(
            max_depth=5,
            top_k_actions=2,
            exploration_quota=0,
            use_dc_screening=use_dc_screening,
            dc_max_depth=dc_max_depth,
            dc_top_k_actions=2,
            dc_candidate_pool=0,
            dc_keep_policy_actions=0,
            dc_keep_loading_actions=0,
        ),
        evaluator=_Evaluator(policy),  # type: ignore[arg-type]
    )
    planner._should_include_stop_action = (  # type: ignore[method-assign]
        lambda state: False
    )

    ranker = _RecordingDCRanker()
    planner.dc_screener = ranker  # type: ignore[assignment]

    for depth in node_depths:
        node = MCTSNode(
            env=_FakeEnv(  # type: ignore[arg-type]
                actions=actions,
                depth=depth,
            ),
            depth=depth,
        )
        planner._expand_node(node)

    return ranker.called_depths


def test_dc_max_depth_zero_screens_only_root() -> None:
    assert _dc_calls(
        dc_max_depth=0,
        node_depths=(0, 1, 2),
    ) == [0]


def test_positive_dc_max_depth_is_inclusive() -> None:
    assert _dc_calls(
        dc_max_depth=1,
        node_depths=(0, 1, 2),
    ) == [0, 1]


def test_negative_one_screens_every_expanded_depth() -> None:
    assert _dc_calls(
        dc_max_depth=-1,
        node_depths=(0, 1, 4),
    ) == [0, 1, 4]


def test_disabled_dc_screening_never_calls_ranker() -> None:
    assert _dc_calls(
        dc_max_depth=-1,
        node_depths=(0, 1, 2),
        use_dc_screening=False,
    ) == []


@pytest.mark.parametrize(
    "value",
    [True, -2, 1.5, "1", None],
)
def test_mcts_config_rejects_invalid_dc_max_depth(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="dc_max_depth",
    ):
        MCTSConfig(dc_max_depth=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [True, -2, 1.5, "1", None],
)
def test_evaluation_request_rejects_invalid_dc_max_depth(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="dc_max_depth",
    ):
        EvaluationRequest(
            raw_dir=Path("raw"),
            transitions_csv=Path("transitions.csv"),
            checkpoint=Path("checkpoint.pt"),
            config=EvaluationConfig(),
            dc_max_depth=value,  # type: ignore[arg-type]
        )


def test_numpy_integer_depth_is_normalized() -> None:
    config = MCTSConfig(
        dc_max_depth=np.int64(2),
    )
    request = EvaluationRequest(
        raw_dir=Path("raw"),
        transitions_csv=Path("transitions.csv"),
        checkpoint=Path("checkpoint.pt"),
        config=EvaluationConfig(),
        dc_max_depth=np.int64(-1),
    )

    assert config.dc_max_depth == 2
    assert type(config.dc_max_depth) is int
    assert request.dc_max_depth == -1
    assert type(request.dc_max_depth) is int

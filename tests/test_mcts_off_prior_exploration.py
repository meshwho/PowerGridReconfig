from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS
import grid_topology_ai.evaluation as evaluation
from grid_topology_ai.evaluation import EvaluationRequest
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner
from tests.topology_contract_helpers import topology_metadata


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


class _FakeReward:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def config_dict(self) -> dict[str, object]:
        return {"reward": "fake"}


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


def _expand(
    *,
    quota: int,
    seed: int = 17,
    top_k: int = 3,
    switch_count: int = 10,
) -> tuple[MCTSPlanner, MCTSNode]:
    actions = _switch_actions(switch_count)
    policy = np.zeros(switch_count + 1, dtype=float)
    policy[1:] = np.linspace(1.0, 0.1, switch_count)

    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=top_k,
            widening_coefficient=0.0,
            exploration_quota=quota,
            random_seed=seed,
        ),
        evaluator=_Evaluator(policy),  # type: ignore[arg-type]
    )
    planner._should_include_stop_action = (  # type: ignore[method-assign]
        lambda state: False
    )
    node = MCTSNode(
        env=_FakeEnv(
            actions,
            list(reversed(range(1, switch_count + 1))),
        ),  # type: ignore[arg-type]
        depth=0,
    )
    planner._expand_node(node)
    return planner, node


def test_seeded_quota_activates_reproducible_tail_actions() -> None:
    _, first = _expand(quota=2, seed=23)
    _, second = _expand(quota=2, seed=23)

    assert first.forced_exploration_action_ids == (
        second.forced_exploration_action_ids
    )
    assert len(first.forced_exploration_action_ids) == 2
    assert set(first.forced_exploration_action_ids).issubset(
        set(range(6, 11))
    )
    assert list(first.actions_by_id) == [
        1,
        2,
        3,
        4,
        5,
        *first.forced_exploration_action_ids,
    ]
    assert sum(first.action_priors.values()) == pytest.approx(1.0)


def test_zero_quota_keeps_only_the_initial_shortlist() -> None:
    _, node = _expand(quota=0)

    assert node.forced_exploration_action_ids == []
    assert list(node.actions_by_id) == [1, 2, 3, 4, 5]


def test_quota_is_limited_by_the_available_tail() -> None:
    _, node = _expand(
        quota=5,
        top_k=8,
        switch_count=10,
    )

    assert set(node.forced_exploration_action_ids) == {9, 10}
    assert set(node.actions_by_id) == set(range(1, 11))
    assert len(node.actions_by_id) == 10


def test_forced_actions_are_tried_before_puct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switches = _switch_actions(5)
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=1,
            widening_coefficient=0.0,
            exploration_quota=2,
        )
    )
    root = MCTSNode(
        env=SimpleNamespace(done=False, solved=False),  # type: ignore[arg-type]
        depth=0,
        is_expanded=True,
        ranked_actions=switches,
        action_scores={
            action.action_id: 1.0
            for action in switches
        },
        forced_exploration_action_ids=[4, 5],
        actions_by_id={
            action.action_id: action
            for action in (switches[0], switches[3], switches[4])
        },
        action_priors={1: 0.8, 4: 0.1, 5: 0.1},
    )
    selected: list[int] = []

    monkeypatch.setattr(
        planner,
        "_widen_node",
        lambda node: False,
    )
    monkeypatch.setattr(
        planner,
        "_leaf_value",
        lambda node: 0.0,
    )

    def fake_create_child(
        parent: MCTSNode,
        action_id: int,
    ) -> MCTSNode:
        del parent
        selected.append(action_id)
        return MCTSNode(
            env=SimpleNamespace(done=True, solved=False),  # type: ignore[arg-type]
            depth=1,
        )

    monkeypatch.setattr(
        planner,
        "_create_child",
        fake_create_child,
    )
    monkeypatch.setattr(
        planner,
        "_select_action_id",
        lambda node: pytest.fail(
            "PUCT must wait until quota actions are visited"
        ),
    )

    planner._run_one_simulation(root)
    planner._run_one_simulation(root)

    assert selected == [4, 5]
    assert root.children[4].visit_count == 1
    assert root.children[5].visit_count == 1
    assert planner._next_forced_exploration_action(root) is None

    monkeypatch.setattr(
        planner,
        "_select_action_id",
        lambda node: 1,
    )
    planner._run_one_simulation(root)

    assert selected == [4, 5, 1]


def test_widening_preserves_forced_tail_actions() -> None:
    switches = _switch_actions(8)
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=2,
            widening_coefficient=1.0,
            widening_exponent=0.5,
            exploration_quota=2,
        )
    )
    node = MCTSNode(
        env=SimpleNamespace(done=False, solved=False),  # type: ignore[arg-type]
        depth=0,
        visit_count=9,
        is_expanded=True,
        ranked_actions=switches,
        action_scores={
            action.action_id: float(9 - action.action_id)
            for action in switches
        },
        forced_exploration_action_ids=[7, 8],
    )
    planner._set_active_actions(
        node,
        [switches[0], switches[1], switches[6], switches[7]],
    )

    assert planner._target_switch_width(node) == 5
    assert planner._widen_node(node) is True
    assert set(node.actions_by_id) == {1, 2, 3, 4, 5, 7, 8}
    assert node.forced_exploration_action_ids == [7, 8]


def test_discard_removes_action_from_forced_queue() -> None:
    switches = _switch_actions(3)
    node = MCTSNode(
        env=SimpleNamespace(done=False, solved=False),  # type: ignore[arg-type]
        depth=0,
        ranked_actions=switches,
        action_scores={1: 1.0, 2: 0.5, 3: 0.25},
        forced_exploration_action_ids=[2, 3],
        actions_by_id={
            action.action_id: action
            for action in switches
        },
        action_priors={1: 0.5, 2: 0.3, 3: 0.2},
    )

    MCTSPlanner._discard_action(node, 2)

    assert node.forced_exploration_action_ids == [3]
    assert 2 not in node.actions_by_id


@pytest.mark.parametrize(
    "value",
    [-1, 1.5, True, "2"],
)
def test_mcts_config_rejects_invalid_exploration_quota(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="exploration_quota"):
        MCTSConfig(exploration_quota=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config_type",
    [GenerationConfig, EvaluationConfig],
)
def test_typed_configs_normalize_exploration_quota(
    config_type: type[GenerationConfig] | type[EvaluationConfig],
) -> None:
    config = config_type(exploration_quota=3.0)  # type: ignore[arg-type]

    assert config.exploration_quota == 3


@pytest.mark.parametrize(
    "config_type",
    [GenerationConfig, EvaluationConfig],
)
@pytest.mark.parametrize(
    "value",
    [-1, 1.5, True, "1.5"],
)
def test_typed_configs_reject_invalid_exploration_quota(
    config_type: type[GenerationConfig] | type[EvaluationConfig],
    value: object,
) -> None:
    with pytest.raises(ValueError, match="exploration_quota"):
        config_type(exploration_quota=value)  # type: ignore[arg-type]


def test_evaluation_task_config_records_exploration_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation,
        "GridFMReward",
        _FakeReward,
    )
    monkeypatch.setattr(
        evaluation,
        "_load_checkpoint_topology_action_payload",
        lambda checkpoint_path: topology_metadata(),
    )

    request = EvaluationRequest(
        raw_dir=tmp_path / "raw",
        transitions_csv=tmp_path / "transitions.csv",
        checkpoint=tmp_path / "checkpoint.pt",
        config=EvaluationConfig(
            exploration_quota=4,
            policy_mode="ungated",
        ),
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )

    task = evaluation._make_task_config(request)

    assert task["exploration_quota"] == 4

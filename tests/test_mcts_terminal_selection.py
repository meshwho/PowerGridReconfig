from types import SimpleNamespace

import pytest

import grid_topology_ai.search.mcts as mcts
from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner
from tests.test_reward_logic import _state


def _terminal_env(state, backend):
    env = object.__new__(TopologySwitchingEnv)
    env.current_state = state
    env.backend = backend
    env.allow_handoff_with_hard_overloads = True
    env.done = False
    env.solved = False
    return env


def test_redispatch_selection_prefers_fewer_switches_within_teacher_epsilon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object()
    root_state = _state(loadings=(142.0,))
    child_state = _state(loadings=(140.0,))

    root = MCTSNode(env=_terminal_env(root_state, backend), depth=0)
    child = MCTSNode(
        env=_terminal_env(child_state, backend),
        depth=1,
        action_from_parent=GridFMAction(
            action_id=1,
            action_type="switch_off_branch",
            branch_id=0,
            branch_pos=0,
        ),
        visit_count=100,
    )
    root.children[1] = child

    redispatch_by_state = {
        id(root_state): 277.40,
        id(child_state): 276.73,
    }
    monkeypatch.setattr(
        mcts,
        "run_minimal_ac_redispatch",
        lambda backend, state: SimpleNamespace(
            validated=True,
            redispatch_l1_mw=redispatch_by_state[id(state)],
        ),
    )

    planner = MCTSPlanner(MCTSConfig())

    assert planner._redispatch_aware_root_action(root) == (0, None)


def test_redispatch_selection_keeps_material_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object()
    root_state = _state(loadings=(142.0,))
    child_state = _state(loadings=(130.0,))

    root = MCTSNode(env=_terminal_env(root_state, backend), depth=0)
    child = MCTSNode(
        env=_terminal_env(child_state, backend),
        depth=1,
        action_from_parent=GridFMAction(
            action_id=1,
            action_type="switch_off_branch",
            branch_id=0,
            branch_pos=0,
        ),
        visit_count=100,
    )
    root.children[1] = child

    redispatch_by_state = {
        id(root_state): 300.0,
        id(child_state): 200.0,
    }
    monkeypatch.setattr(
        mcts,
        "run_minimal_ac_redispatch",
        lambda backend, state: SimpleNamespace(
            validated=True,
            redispatch_l1_mw=redispatch_by_state[id(state)],
        ),
    )

    planner = MCTSPlanner(MCTSConfig())

    assert planner._redispatch_aware_root_action(root) == (1, 0)


class _StepEnv:
    def __init__(self, current_state, next_state):
        self.current_state = current_state
        self.next_state = next_state

    def clone(self):
        return _StepEnv(self.current_state, self.next_state)

    def step(self, action):
        self.current_state = self.next_state
        return SimpleNamespace(power_flow_success=True, reward=0.0)


def test_mcts_rejects_child_that_increases_hard_overload_count() -> None:
    action = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=0,
        branch_pos=0,
    )
    parent = MCTSNode(
        env=_StepEnv(
            _state(loadings=(130.0, 110.0)),
            _state(loadings=(130.0, 130.0)),
        ),
        depth=0,
        actions_by_id={1: action},
        action_priors={1: 1.0},
    )

    planner = MCTSPlanner(MCTSConfig())

    assert planner._create_child(parent, 1) is None


def test_mcts_allows_child_with_same_hard_overload_count() -> None:
    action = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=0,
        branch_pos=0,
    )
    parent = MCTSNode(
        env=_StepEnv(
            _state(loadings=(130.0, 110.0)),
            _state(loadings=(125.0, 110.0)),
        ),
        depth=0,
        actions_by_id={1: action},
        action_priors={1: 1.0},
    )

    planner = MCTSPlanner(MCTSConfig())

    assert planner._create_child(parent, 1) is not None

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner
from grid_topology_ai.termination import TerminationReason
from tests.outcome_evidence_helpers import terminal_evidence


_LOADING_INDEX = BRANCH_FEATURE_COLUMNS.index("loading_percent")


class _Evaluator:
    def evaluate(
        self,
        *,
        state: object,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        del state, action_mask
        return np.asarray([0.0, 0.9, 0.1]), 0.0


class _Env:
    def __init__(
        self,
        actions: list[GridFMAction],
        calls: dict[int, int],
    ) -> None:
        branch_features = np.zeros(
            (2, len(BRANCH_FEATURE_COLUMNS)),
            dtype=float,
        )
        branch_features[:, _LOADING_INDEX] = [140.0, 130.0]

        self.current_state = SimpleNamespace(
            scenario_id=7,
            branch_features=branch_features,
            metrics={},
        )
        self.initial_scenario_id = 7
        self.backend = object()
        self.done = False
        self.solved = False
        self.termination_reason = None
        self.terminal_outcome_evidence = None
        self._actions = list(actions)
        self._calls = calls

    def clone(self) -> "_Env":
        clone = _Env(self._actions, self._calls)
        clone.done = self.done
        clone.solved = self.solved
        clone.termination_reason = self.termination_reason
        clone.terminal_outcome_evidence = self.terminal_outcome_evidence
        return clone

    def valid_actions(self) -> list[GridFMAction]:
        return list(self._actions)

    def operational_action_mask(self) -> np.ndarray:
        return np.asarray([False, True, True], dtype=bool)

    def step(self, action: GridFMAction) -> SimpleNamespace:
        action_id = int(action.action_id)
        self._calls[action_id] = self._calls.get(action_id, 0) + 1

        if action_id == 1:
            self.done = True
            self.solved = False
            self.termination_reason = TerminationReason.POWER_FLOW_FAILED
            self.terminal_outcome_evidence = terminal_evidence(
                TerminationReason.POWER_FLOW_FAILED
            )
            return SimpleNamespace(
                reward=0.0,
                power_flow_success=False,
            )

        self.done = True
        self.solved = True
        self.termination_reason = TerminationReason.SOLVED
        self.terminal_outcome_evidence = terminal_evidence(
            TerminationReason.SOLVED
        )
        return SimpleNamespace(
            reward=0.0,
            power_flow_success=True,
        )


def test_mcts_does_not_select_action_with_known_power_flow_failure() -> None:
    actions = [
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
    calls: dict[int, int] = {}
    env = _Env(actions, calls)
    planner = MCTSPlanner(
        MCTSConfig(
            num_simulations=2,
            max_depth=1,
            top_k_actions=2,
            exploration_quota=0,
            include_stop_action=False,
            random_seed=1,
        ),
        evaluator=_Evaluator(),  # type: ignore[arg-type]
    )
    planner._should_include_stop_action = (  # type: ignore[method-assign]
        lambda state: False
    )

    result = planner.search_from_env(env)  # type: ignore[arg-type]

    assert calls == {1: 1, 2: 1}
    assert result.best_action_id == 2
    assert result.policy == {2: 1.0}
    assert result.visit_counts == {2: 1}
    assert 1 not in result.root.actions_by_id
    assert 1 not in result.root.children
    assert 2 in result.root.children

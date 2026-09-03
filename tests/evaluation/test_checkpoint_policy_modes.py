from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS
import grid_topology_ai.evaluation as evaluation
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner
from grid_topology_ai.termination import TerminationReason


def _metrics(*, secure: bool) -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": 95.0 if secure else 140.0,
        "num_overloaded_branches": 0 if secure else 1,
        "num_hard_overloaded_branches": 0 if secure else 1,
        "total_thermal_overload_mva": 0.0 if secure else 20.0,
        "num_outaged_branches": 1 if secure else 0,
        "num_low_voltage_buses": 0,
        "num_high_voltage_buses": 0,
        "total_voltage_violation": 0.0,
        "num_generator_p_violations": 0,
        "total_generator_p_violation_mw": 0.0,
        "num_generator_q_violations": 0,
        "total_generator_q_violation_mvar": 0.0,
        "num_angle_difference_violations": 0,
        "total_angle_difference_violation_degrees": 0.0,
    }


class _State:
    def __init__(self, *, secure: bool) -> None:
        self.metrics = _metrics(secure=secure)
        self.branch_features = np.zeros(
            (1, len(BRANCH_FEATURE_COLUMNS)),
            dtype=np.float32,
        )
        self.branch_features[
            0,
            BRANCH_FEATURE_COLUMNS.index("br_status"),
        ] = 1.0
        self.branch_features[
            0,
            BRANCH_FEATURE_COLUMNS.index("loading_percent"),
        ] = float(self.metrics["max_loading_percent"])


class _Action:
    def __init__(self, action_id: int, branch_id: int | None) -> None:
        self.action_id = action_id
        self.branch_id = branch_id


class _Env:
    executed_action_ids: list[int] = []

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.current_state = _State(secure=False)
        self.initial_state = self.current_state
        self.done = False
        self.solved = False
        self.termination_reason = None
        self.terminal_outcome_evidence = None

    def reset(self, scenario_id: int):
        del scenario_id
        return self.current_state

    def step(self, action: _Action):
        self.executed_action_ids.append(action.action_id)
        self.current_state = _State(secure=True)
        self.done = True
        self.solved = True
        self.termination_reason = TerminationReason.SOLVED
        return SimpleNamespace(reward=5.0, done=True, solved=True)


class _Planner:
    def __init__(self) -> None:
        self.random_seeds: list[int | None] = []

    def reset_rng(self, random_seed: int | None) -> None:
        self.random_seeds.append(random_seed)

    def search_from_env(self, env: _Env):
        del env
        actions = {
            1: _Action(1, 11),
            2: _Action(2, 22),
        }
        return SimpleNamespace(
            best_action_id=1,
            policy={1: 0.7, 2: 0.3},
            root=SimpleNamespace(actions_by_id=actions),
        )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allowed_action_ids: tuple[int, ...],
) -> None:
    _Env.executed_action_ids = []
    monkeypatch.setattr(evaluation, "_ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(evaluation, "TopologySwitchingEnv", _Env)
    monkeypatch.setattr(
        evaluation,
        "analyze_root_branches",
        lambda **kwargs: SimpleNamespace(
            allowed_action_ids=allowed_action_ids,
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "make_do_nothing_action",
        lambda: _Action(0, None),
    )


def test_constrained_episode_executes_action_from_constrained_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, allowed_action_ids=(2,))
    planner = _Planner()

    row = evaluation.run_episode(
        scenario_id=1,
        adapter=object(),
        backend=object(),
        action_space=object(),
        reward_fn=object(),
        planner=planner,
        max_steps=2,
        gamma=1.0,
        random_seed=73,
        min_hard_improvement=0.0,
        min_soft_improvement=0.0,
        min_constraint_visits=0,
        min_constraint_visit_fraction=0.0,
        policy_mode="constrained",
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )

    assert planner.random_seeds == [
        evaluation._evaluation_search_seed(
            base_seed=73,
            scenario_id=1,
            policy_mode="constrained",
            step=0,
        )
    ]
    assert _Env.executed_action_ids == [2]
    assert row["policy_mode"] == "constrained"
    assert row["actions"] == "[2]"
    assert row["constraint_changed_policy"] is True
    assert row["constraint_exhausted"] is False
    assert row["solved"] is True
    assert row["physically_secure"] is True


def test_empty_constrained_support_terminates_without_action_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, allowed_action_ids=())
    planner = _Planner()

    row = evaluation.run_episode(
        scenario_id=2,
        adapter=object(),
        backend=object(),
        action_space=object(),
        reward_fn=object(),
        planner=planner,
        max_steps=2,
        gamma=1.0,
        random_seed=91,
        min_hard_improvement=0.0,
        min_soft_improvement=0.0,
        min_constraint_visits=0,
        min_constraint_visit_fraction=0.0,
        policy_mode="constrained",
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )

    assert planner.random_seeds == [
        evaluation._evaluation_search_seed(
            base_seed=91,
            scenario_id=2,
            policy_mode="constrained",
            step=0,
        )
    ]
    assert _Env.executed_action_ids == []
    assert row["actions"] == "[]"
    assert row["constraint_exhausted"] is True
    assert row["empty_constrained_support_count"] == 1
    assert row["done"] is True
    assert row["solved"] is False
    assert row["termination_reason"] == "constraint_exhausted"


def test_evaluation_search_seed_is_stable_and_context_specific() -> None:
    baseline = evaluation._evaluation_search_seed(
        base_seed=42,
        scenario_id=7,
        policy_mode="ungated",
        step=0,
    )

    assert baseline == evaluation._evaluation_search_seed(
        base_seed=42,
        scenario_id=7,
        policy_mode="ungated",
        step=0,
    )
    assert len(
        {
            baseline,
            evaluation._evaluation_search_seed(
                base_seed=43,
                scenario_id=7,
                policy_mode="ungated",
                step=0,
            ),
            evaluation._evaluation_search_seed(
                base_seed=42,
                scenario_id=8,
                policy_mode="ungated",
                step=0,
            ),
            evaluation._evaluation_search_seed(
                base_seed=42,
                scenario_id=7,
                policy_mode="constrained",
                step=0,
            ),
            evaluation._evaluation_search_seed(
                base_seed=42,
                scenario_id=7,
                policy_mode="ungated",
                step=1,
            ),
        }
    ) == 5


def test_root_noise_survives_progressive_widening() -> None:
    actions = [
        GridFMAction(
            action_id=index,
            action_type="switch_off_branch",
            branch_id=100 + index,
            branch_pos=index - 1,
        )
        for index in range(1, 4)
    ]
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=2,
            widening_coefficient=1.0,
            widening_exponent=1.0,
            exploration_quota=0,
            use_root_dirichlet_noise=True,
            root_exploration_fraction=1.0,
        )
    )
    node = MCTSNode(
        env=SimpleNamespace(done=False, solved=False),  # type: ignore[arg-type]
        depth=0,
        visit_count=1,
        is_expanded=True,
        ranked_actions=actions,
        action_scores={1: 4.0, 2: 2.0, 3: 1.0},
        selection_scores={1: 4.0, 2: 2.0, 3: 1.0},
    )
    planner._set_active_actions(node, actions[:2])
    planner.rng = SimpleNamespace(  # type: ignore[assignment]
        dirichlet=lambda alpha: np.asarray([0.25, 0.75], dtype=float)
    )

    planner._add_root_dirichlet_noise(node)

    assert node.selection_scores[1] == pytest.approx(1.5)
    assert node.selection_scores[2] == pytest.approx(4.5)
    assert planner._widen_node(node) is True
    assert node.selection_scores == pytest.approx(
        {1: 1.5, 2: 4.5, 3: 1.0}
    )
    assert node.action_priors == pytest.approx(
        {
            1: 1.5 / 7.0,
            2: 4.5 / 7.0,
            3: 1.0 / 7.0,
        }
    )

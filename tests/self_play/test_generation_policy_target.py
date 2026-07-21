from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.self_play import generation


class _Action:
    def __init__(self, branch_id: int | None) -> None:
        self.branch_id = branch_id


def _search_result(policy: dict[int, float]) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        best_action_id=1,
        best_branch_id=11,
        visit_counts={action_id: int(prob * 10) for action_id, prob in policy.items()},
        root=SimpleNamespace(
            actions_by_id={
                1: _Action(11),
                2: _Action(22),
            }
        ),
    )


def _gate(action_id: int = 0, branch_id: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        selected_action_id=action_id,
        selected_branch_id=branch_id,
        selected_reason="fake_gate",
    )


def test_gate_preserves_mcts_visit_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    search_result = _search_result({1: 0.7, 2: 0.3})
    monkeypatch.setattr(generation, "analyze_root_branches", lambda **kwargs: _gate())

    decision = generation._select_generation_action(
        search_result=search_result,
        temperature=0.0,
        rng=np.random.default_rng(1),
        use_continuation_gate=True,
        min_hard_improvement=50.0,
        min_soft_improvement=15.0,
        min_gate_visits=5,
        min_gate_visit_fraction=0.01,
    )

    assert decision.selected_action_id == 0
    assert decision.raw_selected_action_id == 1
    assert decision.policy_target == {1: 0.7, 2: 0.3}
    assert decision.policy_target != {0: 1.0}
    assert search_result.policy == {1: 0.7, 2: 0.3}


def test_gate_can_select_action_outside_policy_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation, "analyze_root_branches", lambda **kwargs: _gate(0, None))

    decision = generation._select_generation_action(
        search_result=_search_result({1: 0.7, 2: 0.3}),
        temperature=0.0,
        rng=np.random.default_rng(2),
        use_continuation_gate=True,
        min_hard_improvement=50.0,
        min_soft_improvement=15.0,
        min_gate_visits=5,
        min_gate_visit_fraction=0.01,
    )

    assert decision.selected_action_id == 0
    assert 0 not in decision.policy_target


def test_no_gate_preserves_mcts_visit_policy() -> None:
    decision = generation._select_generation_action(
        search_result=_search_result({1: 0.7, 2: 0.3}),
        temperature=0.0,
        rng=np.random.default_rng(3),
        use_continuation_gate=False,
        min_hard_improvement=50.0,
        min_soft_improvement=15.0,
        min_gate_visits=5,
        min_gate_visit_fraction=0.01,
    )

    assert decision.selected_action_id == decision.raw_selected_action_id == 1
    assert decision.selected_branch_id == decision.raw_selected_branch_id == 11
    assert decision.policy_target == {1: 0.7, 2: 0.3}


def test_policy_target_is_an_independent_copy() -> None:
    search_result = _search_result({1: 0.7, 2: 0.3})
    decision = generation._select_generation_action(
        search_result=search_result,
        temperature=0.0,
        rng=np.random.default_rng(4),
        use_continuation_gate=False,
        min_hard_improvement=50.0,
        min_soft_improvement=15.0,
        min_gate_visits=5,
        min_gate_visit_fraction=0.01,
    )

    decision.policy_target[1] = 0.0

    assert search_result.policy == {1: 0.7, 2: 0.3}


def test_invalid_non_normalized_mcts_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        generation._select_generation_action(
            search_result=_search_result({1: 0.7, 2: 0.2}),
            temperature=0.0,
            rng=np.random.default_rng(5),
            use_continuation_gate=False,
            min_hard_improvement=50.0,
            min_soft_improvement=15.0,
            min_gate_visits=5,
            min_gate_visit_fraction=0.01,
            scenario_id=123,
            step=4,
        )


def test_empty_mcts_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty policy"):
        generation._select_generation_action(
            search_result=_search_result({}),
            temperature=0.0,
            rng=np.random.default_rng(6),
            use_continuation_gate=False,
            min_hard_improvement=50.0,
            min_soft_improvement=15.0,
            min_gate_visits=5,
            min_gate_visit_fraction=0.01,
        )


def test_gate_override_metadata_values(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: list[dict[str, object]] = []

    class _Writer:
        states_dir = tmp_path / "states"

        def __init__(self, output_dir, *, physics_config):
            self.output_dir = output_dir
            self.physics_config = physics_config

        def add_example(self, **kwargs):
            captured.append(kwargs)

        def save(self):
            return tmp_path / "examples.csv"

    from grid_topology_ai.config import GenerationConfig
    from grid_topology_ai.self_play.generation import (
        GenerationRequest,
        generate_self_play_examples,
    )

    class _Env:
        done = False
        solved = False
        termination_reason = "solved"
        current_state = object()
        def __init__(self, **kwargs): pass
        def reset(self, scenario_id): pass
        def valid_action_mask(self): return [True, True, True]
        def step(self, action):
            self.done = True; self.solved = True
            return SimpleNamespace(reward=0.0, done=True, solved=True, info={"termination_reason": "solved"})

    class _Planner:
        def __init__(self, **kwargs): pass
        def search_from_env(self, env): return _search_result({1: 0.7, 2: 0.3})

    class _Noop:
        def __init__(self, *args, **kwargs): pass
        def cache_info(self): return {}
        def clear_cache(self): pass

    transitions = tmp_path / "transitions.csv"
    transitions.write_text("scenario_id\n1\n", encoding="utf-8")
    monkeypatch.setattr(generation, "_ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(generation, "GridFMAdapter", _Noop)
    monkeypatch.setattr(generation, "GridFMPowerFlowBackend", _Noop)
    monkeypatch.setattr(generation, "GridFMActionSpace", _Noop)
    monkeypatch.setattr(generation, "GridFMReward", _Noop)
    monkeypatch.setattr(generation, "MCTSConfig", _Noop)
    monkeypatch.setattr(generation, "MCTSPlanner", _Planner)
    monkeypatch.setattr(generation, "TopologySwitchingEnv", _Env)
    monkeypatch.setattr(generation, "ExampleWriter", _Writer)
    monkeypatch.setattr(generation, "make_do_nothing_action", lambda: object())
    monkeypatch.setattr(generation, "analyze_root_branches", lambda **kwargs: _gate(0, None))

    generate_self_play_examples(GenerationRequest(
        raw_dir=tmp_path / "raw",
        transitions_csv=transitions,
        output_dir=tmp_path / "out",
        checkpoint=None,
        config=GenerationConfig(max_steps=1, use_continuation_gate=True),
        seed=7,
        clear_cache_between_scenarios=False,
    ))

    assert captured[0]["mcts_policy"] == {1: 0.7, 2: 0.3}
    assert captured[0]["selected_action_id"] == 0
    metadata = captured[0]["extra_metadata"]
    assert metadata["policy_target_source"] == "mcts_visit_distribution"
    assert metadata["execution_action_source"] == "continuation_gate"
    assert metadata["gate_overrode_mcts_selection"] is True

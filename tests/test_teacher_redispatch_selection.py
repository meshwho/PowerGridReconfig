from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from grid_topology_ai.termination import TerminationReason
from scripts.self_play import generate_impact_teacher_redispatch_runtime as teacher


@dataclass
class FakeNode:
    safety_score: float
    branch_ids: list[int | None]
    action_ids: list[int]
    env: Any
    solved: bool = False
    done: bool = False
    termination_reason: TerminationReason | None = None
    discounted_score: float = 0.0
    num_hard_overloaded: int = 0


@dataclass(frozen=True)
class FakeConfig:
    beam_width: int = 10
    relative_physical_epsilon: float = 0.01


@dataclass(frozen=True)
class FakeResult:
    best_node: FakeNode
    final_beam: list[FakeNode]
    pareto_front: list[FakeNode]
    config: FakeConfig
    evaluated_actions: int
    best_physical_safety: float
    selected_safety: float
    selected_switch_count: int
    retained_improvement_fraction: float


def node(
    name: str,
    safety: float,
    switches: int,
    *,
    hard: int = 0,
) -> FakeNode:
    state = SimpleNamespace(name=name, metrics={"name": name})
    env = SimpleNamespace(current_state=state, backend=object())
    return FakeNode(
        safety_score=float(safety),
        branch_ids=list(range(1, switches + 1)),
        action_ids=list(range(1, switches + 1)),
        env=env,
        num_hard_overloaded=int(hard),
    )


def result(root: FakeNode, selected: FakeNode) -> FakeResult:
    return FakeResult(
        best_node=selected,
        final_beam=[selected],
        pareto_front=[root, selected],
        config=FakeConfig(),
        evaluated_actions=100,
        best_physical_safety=float(selected.safety_score),
        selected_safety=float(selected.safety_score),
        selected_switch_count=len(selected.branch_ids),
        retained_improvement_fraction=1.0,
    )


def task_config() -> dict[str, float]:
    return {
        "terminal_redispatch_relative_epsilon": 0.01,
        "terminal_redispatch_absolute_epsilon_mw": 1.0,
        "min_meaningful_safety_improvement": 1.0,
    }


def install_assessment(monkeypatch, *, hard_free: bool) -> None:
    monkeypatch.setattr(
        teacher,
        "assess_physical_state",
        lambda metrics: SimpleNamespace(
            physically_secure=False,
            hard_overload_free=bool(hard_free),
        ),
    )


def install_redispatch(monkeypatch, values: dict[str, float]) -> None:
    def fake_redispatch(backend, state):
        del backend
        value = float(values[state.name])
        return SimpleNamespace(
            validated=True,
            redispatch_l1_mw=value,
        )

    monkeypatch.setattr(
        teacher,
        "run_minimal_ac_redispatch",
        fake_redispatch,
    )


def test_zero_switch_handoff_wins_when_topology_does_not_reduce_redispatch(
    monkeypatch,
) -> None:
    root = node("root", 83.15878, 0)
    topology = node("topology", 81.86080, 4)

    install_assessment(monkeypatch, hard_free=True)
    install_redispatch(
        monkeypatch,
        {
            "root": 123.849,
            "topology": 124.363,
        },
    )

    selected, diagnostics = teacher._redispatch_aware_selection(
        result(root, topology),
        task_config=task_config(),
    )

    assert selected.best_node.action_ids == [0]
    assert selected.best_node.branch_ids == [None]
    assert selected.selected_switch_count == 0
    assert diagnostics["teacher_terminal_selection_applied"] is True
    assert diagnostics["teacher_terminal_pareto_front_size"] == 1


def test_topology_wins_when_one_switch_removes_material_redispatch(
    monkeypatch,
) -> None:
    root = node("root", 100.0, 0)
    topology = node("topology", 50.0, 1)

    install_assessment(monkeypatch, hard_free=True)
    install_redispatch(
        monkeypatch,
        {
            "root": 141.181,
            "topology": 0.069,
        },
    )

    selected, diagnostics = teacher._redispatch_aware_selection(
        result(root, topology),
        task_config=task_config(),
    )

    assert selected.best_node.action_ids == [1, 0]
    assert selected.best_node.branch_ids == [1, None]
    assert selected.selected_switch_count == 1
    assert diagnostics["teacher_terminal_candidate_count"] == 2
    assert diagnostics["teacher_terminal_pareto_front_size"] == 2


def test_tiny_nonterminal_improvement_returns_zero_action_root(
    monkeypatch,
) -> None:
    root = node("root", 719.25736, 0, hard=2)
    topology = node("topology", 719.20043, 6, hard=2)

    install_assessment(monkeypatch, hard_free=False)
    monkeypatch.setattr(
        teacher,
        "run_minimal_ac_redispatch",
        lambda backend, state: pytest.fail("hard-overloaded endpoints must not run OPF"),
    )

    selected, diagnostics = teacher._redispatch_aware_selection(
        result(root, topology),
        task_config=task_config(),
    )

    assert selected.best_node is root
    assert selected.best_node.action_ids == []
    assert selected.selected_switch_count == 0
    assert diagnostics["teacher_terminal_selection_applied"] is False
    assert diagnostics["teacher_terminal_candidate_count"] == 0


def test_terminal_epsilon_prefers_fewer_switches_within_redispatch_tolerance() -> None:
    root = node("root", 100.0, 0)
    one_switch = node("one", 40.0, 1)
    two_switches = node("two", 30.0, 2)

    candidates = [
        teacher._TerminalCandidate(root, 3.0),
        teacher._TerminalCandidate(one_switch, 1.8),
        teacher._TerminalCandidate(two_switches, 1.0),
    ]

    selected, front, pool = teacher._select_terminal_candidate(
        candidates,
        relative_epsilon=0.01,
        absolute_epsilon_mw=1.0,
    )

    assert len(front) == 3
    assert [item.redispatch_l1_mw for item in pool] == [1.8, 1.0]
    assert selected.node is one_switch


def test_replay_promotes_search_stop_to_teacher_redispatch_handoff(
    monkeypatch,
) -> None:
    captured = {}

    def fake_replay(scenario_id, rows):
        captured["scenario_id"] = scenario_id
        captured["reason"] = rows[0]["termination_reason"]
        return "evidence"

    monkeypatch.setattr(
        teacher,
        "_provenance_replay_terminal_evidence",
        fake_replay,
    )

    rows = [
        {
            "termination_reason": TerminationReason.HANDOFF_TO_REDISPATCH.value,
        }
    ]

    evidence = teacher._replay_terminal_evidence(17, rows)

    assert evidence == "evidence"
    assert captured == {
        "scenario_id": 17,
        "reason": TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value,
    }


def test_task_config_records_redispatch_selection_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        teacher,
        "_provenance_make_task_config",
        lambda args: {"min_safety_improvement": 123.0},
    )

    config = teacher.make_task_config(SimpleNamespace())

    assert config["min_safety_improvement"] == 0.0
    assert config["min_meaningful_safety_improvement"] == 1.0
    assert config["terminal_redispatch_relative_epsilon"] == 0.01
    assert config["terminal_redispatch_absolute_epsilon_mw"] == 1.0

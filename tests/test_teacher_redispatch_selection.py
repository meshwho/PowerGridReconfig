from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

import grid_topology_ai.search.teacher as teacher
from grid_topology_ai.physics.redispatch import MinimalRedispatchResult
from grid_topology_ai.termination import TerminationReason


@dataclass
class FakeNode:
    safety_score: float
    branch_ids: list[int | None]
    action_ids: list[int]
    env: Any
    solved: bool = False
    done: bool = False
    discounted_score: float = 0.0
    num_hard_overloaded: int = 0
    termination_reason: TerminationReason | None = None


@dataclass(frozen=True)
class FakeConfig:
    beam_width: int = 10


@dataclass(frozen=True)
class FakeResult:
    best_node: FakeNode
    final_beam: list[FakeNode]
    redispatch_candidates: list[FakeNode]
    config: FakeConfig
    evaluated_actions: int = 100


def node(
    name: str,
    safety: float,
    action_ids: list[int],
    *,
    hard: int = 0,
    secure: bool = False,
    done: bool = False,
    reason: TerminationReason | None = None,
) -> FakeNode:
    state = SimpleNamespace(
        name=name,
        metrics={
            "name": name,
            "physically_secure": bool(secure),
            "hard_overload_free": int(hard) == 0,
        },
    )
    env = SimpleNamespace(current_state=state, backend=object())
    return FakeNode(
        safety_score=float(safety),
        branch_ids=[int(action_id) for action_id in action_ids],
        action_ids=[int(action_id) for action_id in action_ids],
        env=env,
        solved=bool(secure),
        done=bool(done),
        num_hard_overloaded=int(hard),
        termination_reason=reason,
    )


def result(*candidates: FakeNode, selected: FakeNode | None = None) -> FakeResult:
    if not candidates:
        raise ValueError("At least one candidate is required")
    best = candidates[0] if selected is None else selected
    return FakeResult(
        best_node=best,
        final_beam=[best],
        redispatch_candidates=list(candidates),
        config=FakeConfig(),
    )


def task_config() -> dict[str, float]:
    return {
        "terminal_redispatch_relative_epsilon": 0.01,
        "terminal_redispatch_absolute_epsilon_mw": 1.0,
    }


def install_assessment(monkeypatch) -> None:
    monkeypatch.setattr(
        teacher,
        "assess_physical_state",
        lambda metrics: SimpleNamespace(
            physically_secure=bool(metrics["physically_secure"]),
            hard_overload_free=bool(metrics["hard_overload_free"]),
        ),
    )


def install_redispatch(
    monkeypatch,
    values: dict[str, tuple[bool, float | None]],
    calls: list[str] | None = None,
) -> None:
    def fake_redispatch(backend, state):
        del backend
        if calls is not None:
            calls.append(str(state.name))
        validated, redispatch_l1_mw = values[str(state.name)]
        return SimpleNamespace(
            validated=bool(validated),
            redispatch_l1_mw=redispatch_l1_mw,
        )

    monkeypatch.setattr(teacher, "run_minimal_ac_redispatch", fake_redispatch)


def test_hard_overloaded_root_runs_redispatch_and_can_win(monkeypatch) -> None:
    root = node("root", 100.0, [], hard=2)
    calls: list[str] = []
    install_assessment(monkeypatch)
    install_redispatch(monkeypatch, {"root": (True, 12.0)}, calls)

    selected, diagnostics = teacher._redispatch_aware_selection(
        result(root),
        task_config=task_config(),
    )

    assert calls == ["root"]
    assert selected.best_node.action_ids == [0]
    assert selected.best_node.branch_ids == [None]
    assert teacher.switch_count(selected.best_node) == 0
    assert diagnostics["teacher_terminal_selection_applied"] is True
    assert diagnostics["teacher_selected_redispatch_l1_mw"] == pytest.approx(12.0)


def test_precomputed_initial_redispatch_is_reused_for_root(monkeypatch) -> None:
    root = node("root", 100.0, [], hard=2)
    topology = node("topology", 50.0, [11], hard=1)
    calls: list[str] = []
    install_assessment(monkeypatch)
    install_redispatch(monkeypatch, {"topology": (True, 1.0)}, calls)
    initial_redispatch = MinimalRedispatchResult(
        opf_success=True,
        assessment=SimpleNamespace(physically_secure=True),
        message="validated",
        redispatch_l1_mw=12.0,
    )

    selected, diagnostics = teacher._redispatch_aware_selection(
        result(root, topology),
        task_config=task_config(),
        initial_redispatch_result=initial_redispatch,
    )

    assert calls == ["topology"]
    assert selected.best_node.action_ids == [11, 0]
    assert diagnostics["teacher_terminal_candidate_count"] == 2


def test_hard_overloaded_candidate_is_rejected_when_redispatch_is_not_validated(
    monkeypatch,
) -> None:
    root = node("root", 100.0, [], hard=3)
    calls: list[str] = []
    install_assessment(monkeypatch)
    install_redispatch(monkeypatch, {"root": (False, None)}, calls)

    selected, diagnostics = teacher._redispatch_aware_selection(
        result(root),
        task_config=task_config(),
    )

    assert calls == ["root"]
    assert selected.best_node is root
    assert selected.best_node.action_ids == []
    assert diagnostics["teacher_terminal_selection_applied"] is False
    assert diagnostics["teacher_terminal_candidate_count"] == 0


def test_max_depth_candidate_still_receives_terminal_redispatch(monkeypatch) -> None:
    terminal = node(
        "depth_limit",
        70.0,
        [11, 12, 13],
        hard=1,
        done=True,
        reason=TerminationReason.TEACHER_DEPTH_LIMIT,
    )
    calls: list[str] = []
    install_assessment(monkeypatch)
    install_redispatch(monkeypatch, {"depth_limit": (True, 4.5)}, calls)

    selected, diagnostics = teacher._redispatch_aware_selection(
        result(terminal),
        task_config=task_config(),
    )

    assert calls == ["depth_limit"]
    assert selected.best_node.action_ids == [11, 12, 13, 0]
    assert selected.best_node.branch_ids == [11, 12, 13, None]
    assert diagnostics["teacher_terminal_selection_applied"] is True


def test_zero_switch_handoff_wins_within_terminal_redispatch_epsilon(
    monkeypatch,
) -> None:
    root = node("root", 83.0, [])
    topology = node("topology", 40.0, [11, 12])
    install_assessment(monkeypatch)
    install_redispatch(
        monkeypatch,
        {
            "root": (True, 1.8),
            "topology": (True, 1.0),
        },
    )

    selected, _ = teacher._redispatch_aware_selection(
        result(root, topology),
        task_config=task_config(),
    )

    assert selected.best_node.action_ids == [0]
    assert teacher.switch_count(selected.best_node) == 0


def test_topology_wins_when_it_materially_reduces_redispatch(monkeypatch) -> None:
    root = node("root", 100.0, [])
    topology = node("topology", 50.0, [11])
    install_assessment(monkeypatch)
    install_redispatch(
        monkeypatch,
        {
            "root": (True, 141.181),
            "topology": (True, 0.069),
        },
    )

    selected, diagnostics = teacher._redispatch_aware_selection(
        result(root, topology),
        task_config=task_config(),
    )

    assert selected.best_node.action_ids == [11, 0]
    assert teacher.switch_count(selected.best_node) == 1
    assert diagnostics["teacher_terminal_candidate_count"] == 2


def test_worse_j_candidate_survives_shortlist_and_can_win_on_redispatch(
    monkeypatch,
) -> None:
    better_j = node("better_j", 10.0, [11])
    worse_j = node("worse_j", 20.0, [12])
    archive = teacher.update_top_j_candidate_archive(
        {},
        [better_j, worse_j],
        per_switch_count=2,
    )
    shortlisted = archive[1]
    assert shortlisted == [better_j, worse_j]

    install_assessment(monkeypatch)
    install_redispatch(
        monkeypatch,
        {
            "better_j": (True, 50.0),
            "worse_j": (True, 5.0),
        },
    )

    selected, _ = teacher._redispatch_aware_selection(
        result(*shortlisted),
        task_config={
            "terminal_redispatch_relative_epsilon": 0.0,
            "terminal_redispatch_absolute_epsilon_mw": 0.0,
        },
    )

    assert selected.best_node.action_ids == [12, 0]
    assert selected.best_node.safety_score == pytest.approx(20.0)


def test_terminal_pareto_uses_switch_count_and_redispatch_only() -> None:
    root = node("root", 1000.0, [])
    one_switch = node("one", 1.0, [11])
    two_switches = node("two", 0.1, [21, 22])
    dominated = node("dominated", 0.0, [31, 32, 33])
    candidates = [
        teacher._TerminalCandidate(root, 10.0),
        teacher._TerminalCandidate(one_switch, 5.0),
        teacher._TerminalCandidate(two_switches, 1.0),
        teacher._TerminalCandidate(dominated, 6.0),
    ]

    front = teacher._terminal_pareto_front(candidates)

    pairs = [
        (teacher.switch_count(item.node), item.redispatch_l1_mw)
        for item in front
    ]
    assert pairs == [(0, 10.0), (1, 5.0), (2, 1.0)]


def test_terminal_epsilon_prefers_fewer_switches_within_redispatch_tolerance() -> None:
    root = node("root", 100.0, [])
    one_switch = node("one", 40.0, [11])
    two_switches = node("two", 30.0, [21, 22])
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

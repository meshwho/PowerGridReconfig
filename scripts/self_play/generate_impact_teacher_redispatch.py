from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

from grid_topology_ai.physical_objective import assess_physical_state
from grid_topology_ai.redispatch import run_minimal_ac_redispatch
from grid_topology_ai.search.trajectory_selection import switch_count
from grid_topology_ai.termination import TerminationReason, parse_termination_reason
from scripts.self_play import generate_impact_teacher_provenance as base


_TEACHER_SELECTION_MODE = "redispatch_aware_epsilon_minimum_switch"
_TERMINAL_REDISPATCH_RELATIVE_EPSILON = 0.01
_TERMINAL_REDISPATCH_ABSOLUTE_EPSILON_MW = 1.0
_MIN_MEANINGFUL_SAFETY_IMPROVEMENT = 1.0
_TOLERANCE = 1e-9

_EXTRA_SELECTION_ROW_FIELDS = (
    "terminal_redispatch_relative_epsilon",
    "terminal_redispatch_absolute_epsilon_mw",
    "min_meaningful_safety_improvement",
    "teacher_terminal_selection_applied",
    "teacher_terminal_candidate_count",
    "teacher_terminal_pareto_front_size",
)

_ORIGINAL_MAKE_TASK_CONFIG = base.make_task_config
_ORIGINAL_PROCESS_SCENARIO_BATCH = base.process_scenario_batch
_ORIGINAL_REPLAY_TERMINAL_EVIDENCE = base._replay_terminal_evidence
_ORIGINAL_SELECTION_PROVENANCE_IS_VALID = base._selection_provenance_is_valid
_ORIGINAL_REQUIRED_CHECKPOINT_ROW_FIELDS = base._REQUIRED_CHECKPOINT_ROW_FIELDS
_ORIGINAL_SELECTION_ROW_FIELDS = base._SELECTION_ROW_FIELDS


@dataclass(frozen=True)
class _TerminalCandidate:
    node: Any
    redispatch_l1_mw: float


def make_task_config(args) -> dict[str, Any]:
    task_config = _ORIGINAL_MAKE_TASK_CONFIG(args)

    # The old gate cannot distinguish a useful terminal handoff from a weak
    # partial topology trajectory. The redispatch-aware selector handles those
    # cases explicitly, so keep the legacy gate disabled here.
    task_config["min_safety_improvement"] = 0.0
    task_config["min_meaningful_safety_improvement"] = (
        _MIN_MEANINGFUL_SAFETY_IMPROVEMENT
    )
    task_config["terminal_redispatch_relative_epsilon"] = (
        _TERMINAL_REDISPATCH_RELATIVE_EPSILON
    )
    task_config["terminal_redispatch_absolute_epsilon_mw"] = (
        _TERMINAL_REDISPATCH_ABSOLUTE_EPSILON_MW
    )
    return task_config


def _action_key(node: Any) -> tuple[int, ...]:
    return tuple(int(action_id) for action_id in node.action_ids)


def _terminal_candidate_key(candidate: _TerminalCandidate) -> tuple[object, ...]:
    return (
        switch_count(candidate.node),
        float(candidate.redispatch_l1_mw),
        float(candidate.node.safety_score),
        _action_key(candidate.node),
    )


def _same_terminal_objectives(
    left: _TerminalCandidate,
    right: _TerminalCandidate,
) -> bool:
    return (
        switch_count(left.node) == switch_count(right.node)
        and abs(left.redispatch_l1_mw - right.redispatch_l1_mw) <= _TOLERANCE
    )


def _terminal_dominates(
    left: _TerminalCandidate,
    right: _TerminalCandidate,
) -> bool:
    left_switches = switch_count(left.node)
    right_switches = switch_count(right.node)
    left_redispatch = float(left.redispatch_l1_mw)
    right_redispatch = float(right.redispatch_l1_mw)

    no_worse = (
        left_switches <= right_switches
        and left_redispatch <= right_redispatch + _TOLERANCE
    )
    strictly_better = (
        left_switches < right_switches
        or left_redispatch < right_redispatch - _TOLERANCE
    )
    return no_worse and strictly_better


def _terminal_pareto_front(
    candidates: Sequence[_TerminalCandidate],
) -> list[_TerminalCandidate]:
    unique: list[_TerminalCandidate] = []

    for candidate in candidates:
        duplicate_index = next(
            (
                index
                for index, other in enumerate(unique)
                if _same_terminal_objectives(candidate, other)
            ),
            None,
        )
        if duplicate_index is None:
            unique.append(candidate)
            continue

        if _terminal_candidate_key(candidate) < _terminal_candidate_key(
            unique[duplicate_index]
        ):
            unique[duplicate_index] = candidate

    front = [
        candidate
        for candidate in unique
        if not any(
            other is not candidate and _terminal_dominates(other, candidate)
            for other in unique
        )
    ]
    return sorted(front, key=_terminal_candidate_key)


def _with_handoff(node: Any) -> Any:
    if node.action_ids and int(node.action_ids[-1]) == 0:
        return node

    return replace(
        node,
        action_ids=[*node.action_ids, 0],
        branch_ids=[*node.branch_ids, None],
        done=True,
        solved=False,
        termination_reason=TerminationReason.HANDOFF_TO_REDISPATCH,
    )


def _terminal_candidate(node: Any) -> _TerminalCandidate | None:
    state = node.env.current_state
    if state is None:
        return None

    assessment = assess_physical_state(state.metrics)
    if assessment.physically_secure:
        return _TerminalCandidate(
            node=node,
            redispatch_l1_mw=0.0,
        )

    if not assessment.hard_overload_free:
        return None

    # A MAX_STEPS_REACHED node cannot execute the handoff action during replay.
    # Keep it in the physical fallback instead of creating an unreachable label.
    if node.done:
        reason = parse_termination_reason(node.termination_reason)
        if reason not in {
            TerminationReason.HANDOFF_TO_REDISPATCH,
            TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        }:
            return None

    redispatch = run_minimal_ac_redispatch(
        node.env.backend,
        state,
    )
    if not redispatch.validated or redispatch.redispatch_l1_mw is None:
        return None

    redispatch_l1_mw = float(redispatch.redispatch_l1_mw)
    if not math.isfinite(redispatch_l1_mw) or redispatch_l1_mw < 0.0:
        return None

    return _TerminalCandidate(
        node=_with_handoff(node),
        redispatch_l1_mw=redispatch_l1_mw,
    )


def _root_node(result) -> Any | None:
    roots = [
        node
        for node in result.pareto_front
        if switch_count(node) == 0 and not node.action_ids
    ]
    if not roots:
        return None
    return min(roots, key=lambda node: float(node.safety_score))


def _retained_physical_improvement(
    *,
    root_safety: float,
    best_physical_safety: float,
    selected_safety: float,
) -> float:
    available = max(float(root_safety) - float(best_physical_safety), 0.0)
    if available <= _TOLERANCE:
        return 1.0

    retained = (
        float(root_safety) - float(selected_safety)
    ) / available
    return float(min(max(retained, 0.0), 1.0))


def _select_terminal_candidate(
    candidates: Sequence[_TerminalCandidate],
    *,
    relative_epsilon: float,
    absolute_epsilon_mw: float,
) -> tuple[_TerminalCandidate, list[_TerminalCandidate], list[_TerminalCandidate]]:
    front = _terminal_pareto_front(candidates)
    if not front:
        raise ValueError("Terminal redispatch selection requires at least one candidate.")

    best_redispatch = min(candidate.redispatch_l1_mw for candidate in front)
    threshold = (
        best_redispatch * (1.0 + float(relative_epsilon))
        + float(absolute_epsilon_mw)
    )
    pool = [
        candidate
        for candidate in front
        if candidate.redispatch_l1_mw <= threshold + _TOLERANCE
    ]
    selected = min(pool, key=_terminal_candidate_key)
    return selected, front, sorted(pool, key=_terminal_candidate_key)


def _redispatch_aware_selection(
    result,
    *,
    task_config: dict[str, Any],
) -> tuple[Any, dict[str, object]]:
    terminal_candidates = [
        candidate
        for node in result.pareto_front
        if (candidate := _terminal_candidate(node)) is not None
    ]

    diagnostics: dict[str, object] = {
        "terminal_redispatch_relative_epsilon": float(
            task_config["terminal_redispatch_relative_epsilon"]
        ),
        "terminal_redispatch_absolute_epsilon_mw": float(
            task_config["terminal_redispatch_absolute_epsilon_mw"]
        ),
        "min_meaningful_safety_improvement": float(
            task_config["min_meaningful_safety_improvement"]
        ),
        "teacher_terminal_selection_applied": False,
        "teacher_terminal_candidate_count": int(len(terminal_candidates)),
        "teacher_terminal_pareto_front_size": 0,
    }

    root = _root_node(result)
    root_safety = (
        float(root.safety_score)
        if root is not None
        else float(result.selected_safety)
    )

    if terminal_candidates:
        selected, terminal_front, terminal_pool = _select_terminal_candidate(
            terminal_candidates,
            relative_epsilon=float(
                task_config["terminal_redispatch_relative_epsilon"]
            ),
            absolute_epsilon_mw=float(
                task_config["terminal_redispatch_absolute_epsilon_mw"]
            ),
        )
        diagnostics["teacher_terminal_selection_applied"] = True
        diagnostics["teacher_terminal_pareto_front_size"] = int(
            len(terminal_front)
        )

        retained = _retained_physical_improvement(
            root_safety=root_safety,
            best_physical_safety=float(result.best_physical_safety),
            selected_safety=float(selected.node.safety_score),
        )
        updated = replace(
            result,
            best_node=selected.node,
            final_beam=[
                candidate.node
                for candidate in terminal_pool[: result.config.beam_width]
            ],
            selected_safety=float(selected.node.safety_score),
            selected_switch_count=int(switch_count(selected.node)),
            retained_improvement_fraction=retained,
        )
        return updated, diagnostics

    meaningful_improvement = float(root_safety) - float(result.selected_safety)
    minimum = float(task_config["min_meaningful_safety_improvement"])

    if (
        root is not None
        and not bool(result.best_node.solved)
        and meaningful_improvement < minimum
    ):
        # Return the zero-action root. The existing generator will classify this
        # as no_teacher_action_found and will not write a misleading policy label.
        result = replace(
            result,
            best_node=root,
            final_beam=[root],
            selected_safety=float(root.safety_score),
            selected_switch_count=0,
            retained_improvement_fraction=0.0,
        )

    return result, diagnostics


def _selection_provenance(
    result,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    return {
        "teacher_selection_mode": _TEACHER_SELECTION_MODE,
        "relative_physical_epsilon": float(
            result.config.relative_physical_epsilon
        ),
        "teacher_best_physical_safety": float(result.best_physical_safety),
        "teacher_selected_safety": float(result.selected_safety),
        "teacher_selected_switch_count": int(result.selected_switch_count),
        "teacher_retained_improvement_fraction": float(
            result.retained_improvement_fraction
        ),
        "teacher_pareto_front_size": int(len(result.pareto_front)),
        **diagnostics,
    }


def _instrumented_planner_search(self, env, scenario_id: int):
    scenario_id = int(scenario_id)
    backend = env.backend
    before = backend.performance_info()
    result = None
    base._SELECTION_PROVENANCE_BY_SCENARIO.pop(scenario_id, None)

    try:
        result = base._original_planner_search(
            self,
            env=env,
            scenario_id=scenario_id,
        )
        task_config = base.teacher._require_worker_context()["task_config"]
        result, diagnostics = _redispatch_aware_selection(
            result,
            task_config=task_config,
        )
        base._SELECTION_PROVENANCE_BY_SCENARIO[scenario_id] = (
            _selection_provenance(result, diagnostics)
        )
        return result
    finally:
        after = backend.performance_info()
        logical_evaluations = (
            int(result.evaluated_actions)
            if result is not None
            else int(getattr(self, "evaluated_actions", 0))
        )
        base._SEARCH_WORKLOAD_BY_SCENARIO[scenario_id] = base._search_workload(
            before=before,
            after=after,
            logical_evaluations=logical_evaluations,
        )


def _selection_provenance_is_valid(row: dict[str, Any]) -> bool:
    if not _ORIGINAL_SELECTION_PROVENANCE_IS_VALID(row):
        return False

    try:
        relative_epsilon = float(row["terminal_redispatch_relative_epsilon"])
        absolute_epsilon = float(row["terminal_redispatch_absolute_epsilon_mw"])
        minimum_improvement = float(row["min_meaningful_safety_improvement"])
        terminal_count = int(row["teacher_terminal_candidate_count"])
        terminal_front_size = int(row["teacher_terminal_pareto_front_size"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False

    applied = row.get("teacher_terminal_selection_applied")
    if not isinstance(applied, bool):
        return False
    if not 0.0 <= relative_epsilon < 1.0:
        return False
    if not math.isfinite(absolute_epsilon) or absolute_epsilon < 0.0:
        return False
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        return False
    if terminal_count < 0 or terminal_front_size < 0:
        return False
    if applied and (terminal_count == 0 or terminal_front_size == 0):
        return False
    if not applied and terminal_front_size != 0:
        return False
    return True


def _replay_terminal_evidence(
    scenario_id: int,
    rows: list[dict[str, Any]],
):
    reasons = {
        parse_termination_reason(
            row.get("termination_reason"),
            allow_none=False,
        )
        for row in rows
    }
    if reasons == {TerminationReason.HANDOFF_TO_REDISPATCH}:
        for row in rows:
            row["termination_reason"] = (
                TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value
            )

    return _ORIGINAL_REPLAY_TERMINAL_EVIDENCE(scenario_id, rows)


def process_scenario_batch(
    scenario_ids: list[int],
) -> list[dict[str, Any]]:
    _install_overrides()
    return _ORIGINAL_PROCESS_SCENARIO_BATCH(scenario_ids)


def _install_overrides() -> None:
    base._TEACHER_SELECTION_MODE = _TEACHER_SELECTION_MODE
    base._SELECTION_ROW_FIELDS = (
        *_ORIGINAL_SELECTION_ROW_FIELDS,
        *_EXTRA_SELECTION_ROW_FIELDS,
    )
    base._REQUIRED_CHECKPOINT_ROW_FIELDS = (
        *_ORIGINAL_REQUIRED_CHECKPOINT_ROW_FIELDS,
        *_EXTRA_SELECTION_ROW_FIELDS,
    )
    base.make_task_config = make_task_config
    base._instrumented_planner_search = _instrumented_planner_search
    base._selection_provenance_is_valid = _selection_provenance_is_valid
    base._replay_terminal_evidence = _replay_terminal_evidence
    base.process_scenario_batch = process_scenario_batch


def main() -> None:
    _install_overrides()
    base.main()


if __name__ == "__main__":
    main()

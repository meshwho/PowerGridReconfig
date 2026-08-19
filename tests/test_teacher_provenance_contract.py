from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
    topology_action_provenance,
)
from grid_topology_ai.outcome_contract import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.redispatch import MinimalRedispatchResult
from grid_topology_ai.return_contract import terminal_utility_from_outcome
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.topology_actions import (
    ActionSpaceConfig,
    build_branch_action_slots,
)
from scripts.self_play import generate_impact_teacher_redispatch_runtime as provenance
from tests.outcome_evidence_helpers import terminal_evidence


def _selection_fields() -> dict[str, object]:
    return {
        "teacher_selection_mode": "redispatch_aware_epsilon_minimum_switch",
        "relative_physical_epsilon": 0.01,
        "teacher_best_physical_safety": 35.563075,
        "teacher_selected_safety": 36.510767,
        "teacher_selected_switch_count": 3,
        "teacher_retained_improvement_fraction": 0.993496,
        "teacher_pareto_front_size": 6,
        "terminal_redispatch_relative_epsilon": 0.01,
        "terminal_redispatch_absolute_epsilon_mw": 1.0,
        "min_meaningful_safety_improvement": 1.0,
        "teacher_terminal_selection_applied": False,
        "teacher_terminal_candidate_count": 0,
        "teacher_terminal_pareto_front_size": 0,
    }


def _terminal_selection_diagnostics() -> dict[str, object]:
    fields = _selection_fields()
    return {
        "terminal_redispatch_relative_epsilon": fields[
            "terminal_redispatch_relative_epsilon"
        ],
        "terminal_redispatch_absolute_epsilon_mw": fields[
            "terminal_redispatch_absolute_epsilon_mw"
        ],
        "min_meaningful_safety_improvement": fields[
            "min_meaningful_safety_improvement"
        ],
        "teacher_terminal_selection_applied": fields[
            "teacher_terminal_selection_applied"
        ],
        "teacher_terminal_candidate_count": fields[
            "teacher_terminal_candidate_count"
        ],
        "teacher_terminal_pareto_front_size": fields[
            "teacher_terminal_pareto_front_size"
        ],
    }


def _current_checkpoint_row() -> dict[str, object]:
    evidence = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    )
    action_provenance = topology_action_provenance(
        ActionSpaceConfig(),
        build_branch_action_slots(
            np.asarray([10, 20], dtype=np.int64)
        ),
    )
    return {
        **physics_provenance(DEFAULT_PHYSICS_CONFIG),
        **action_provenance,
        "run_id": "teacher-run",
        "iteration": 1,
        "episode_id": "teacher-episode",
        "scenario_id": 5,
        "solved": False,
        "termination_reason": evidence.termination_reason.value,
        "terminal_outcome_evidence_schema_version": (
            TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
        ),
        "terminal_outcome_evidence_json": evidence.to_json(),
        "outcome_objective_version": OUTCOME_OBJECTIVE_VERSION,
        "outcome_value_target_contract_version": (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        ),
        "redispatch_attempted": False,
        "redispatch_opf_success": False,
        "redispatch_validated": False,
        **_selection_fields(),
    }


def test_stale_successful_checkpoint_is_retried() -> None:
    stale = {
        "ok": True,
        "reason": None,
        "rows": [{"scenario_id": 5, "step": 0}],
    }
    current = {
        "ok": True,
        "reason": None,
        "rows": [_current_checkpoint_row()],
    }
    legacy_row = _current_checkpoint_row()
    legacy_row.pop("teacher_selection_mode")
    legacy = {
        "ok": True,
        "reason": None,
        "rows": [legacy_row],
    }

    assert provenance._checkpoint_result_is_current(stale) is False
    assert provenance._checkpoint_result_is_current(legacy) is False
    assert provenance._checkpoint_result_is_current(current) is True
    assert provenance._checkpoint_result_is_current(
        {"ok": False, "reason": "exception", "rows": []}
    ) is False
    assert provenance._checkpoint_result_is_current(
        {"ok": False, "reason": "no_teacher_action_found", "rows": []}
    ) is True


def test_invalid_selection_provenance_is_not_current() -> None:
    row = _current_checkpoint_row()
    row["relative_physical_epsilon"] = 1.0

    assert provenance._checkpoint_result_is_current(
        {"ok": True, "reason": None, "rows": [row]}
    ) is False


def test_instrumented_search_captures_selection_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def performance_info(self) -> dict[str, object]:
            return {}

    search_result = SimpleNamespace(
        evaluated_actions=17,
        config=SimpleNamespace(relative_physical_epsilon=0.01),
        best_physical_safety=35.563075,
        selected_safety=36.510767,
        selected_switch_count=3,
        retained_improvement_fraction=0.993496,
        pareto_front=[object()] * 6,
    )
    monkeypatch.setattr(
        provenance,
        "_original_planner_search",
        lambda self, env, scenario_id: search_result,
    )
    monkeypatch.setattr(
        provenance,
        "_require_worker_context",
        lambda: {"task_config": {}},
    )
    monkeypatch.setattr(
        provenance,
        "_redispatch_aware_selection",
        lambda result, *, task_config: (
            result,
            _terminal_selection_diagnostics(),
        ),
    )
    provenance._SELECTION_PROVENANCE_BY_SCENARIO.clear()

    returned = provenance._instrumented_planner_search(
        object(),
        SimpleNamespace(backend=FakeBackend()),
        scenario_id=61,
    )

    assert returned is search_result
    assert provenance._SELECTION_PROVENANCE_BY_SCENARIO[61] == _selection_fields()
    provenance._SELECTION_PROVENANCE_BY_SCENARIO.clear()


def test_success_result_gets_identity_evidence_and_state_provenance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.npz"
    np.savez_compressed(
        state_path,
        branch_ids=np.asarray([10, 20], dtype=np.int64),
        metadata_json=np.array(json.dumps({"source": "teacher"})),
    )

    evidence = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    )
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=True,
    )
    monkeypatch.setattr(
        provenance,
        "_worker_run_id",
        lambda: "impact_teacher_test",
    )

    def replay_with_defaults(scenario_id, rows):
        del scenario_id
        provenance._set_redispatch_diagnostics(rows, None)
        return evidence

    monkeypatch.setattr(
        provenance,
        "_replay_terminal_evidence",
        replay_with_defaults,
    )
    monkeypatch.setattr(
        provenance.teacher,
        "_require_worker_context",
        lambda: {"action_space": action_space},
    )
    provenance._SELECTION_PROVENANCE_BY_SCENARIO[5] = _selection_fields()

    result = {
        "ok": True,
        "scenario_id": 5,
        "rows": [
            {
                "state_path": str(state_path),
                "scenario_id": 5,
                "step": 0,
                "selected_action_id": 0,
                "solved": False,
                "done": True,
                "termination_reason": (
                    TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value
                ),
                "bus_feature_columns": ["Pd", "Qd"],
                "branch_feature_columns": ["pf", "qf"],
            }
        ],
    }

    finalized = provenance._finalize_success_result(result)
    row = finalized["rows"][0]

    assert row["run_id"] == "impact_teacher_test"
    assert row["iteration"] == 1
    assert row["episode_id"] == "impact_teacher_test_scenario_000005"
    assert row["terminal_outcome_evidence_json"] == evidence.to_json()
    assert row["redispatch_attempted"] is False
    assert json.loads(row["bus_feature_columns"]) == ["Pd", "Qd"]
    assert json.loads(row["branch_feature_columns"]) == ["pf", "qf"]
    assert json.loads(row["topology_action_config"])[
        "require_connected_after_switch"
    ] is True
    assert len(json.loads(row["action_layout"])) == 3
    for field, value in _selection_fields().items():
        assert row[field] == value
    assert 5 not in provenance._SELECTION_PROVENANCE_BY_SCENARIO

    with np.load(state_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))

    assert metadata["run_id"] == row["run_id"]
    assert metadata["iteration"] == row["iteration"]
    assert metadata["episode_id"] == row["episode_id"]
    assert metadata["terminal_outcome_evidence"] == evidence.to_dict()
    assert metadata["topology_action_contract_version"] == 1
    assert metadata["redispatch_attempted"] is False
    assert metadata["episode_termination_reason"] == (
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value
    )
    for field, value in _selection_fields().items():
        assert metadata[field] == value


def _patch_replay_context(
    monkeypatch: pytest.MonkeyPatch,
    assessment,
    *,
    done: bool = False,
    terminal_evidence_value: TerminalOutcomeEvidence | None = None,
    topology_utility: float = -0.4,
) -> None:
    class FakeEnv:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.done = done
            self.terminal_outcome_evidence = terminal_evidence_value
            self.current_state = SimpleNamespace(metrics={})

        def reset(self, scenario_id: int) -> object:
            del scenario_id
            return self.current_state

        def step(self, action_id: int) -> object:
            raise AssertionError(f"unexpected action {action_id}")

    monkeypatch.setattr(provenance, "TopologySwitchingEnv", FakeEnv)
    monkeypatch.setattr(
        provenance,
        "state_utility",
        lambda state, physics_config=None: float(topology_utility),
    )
    if assessment is not None:
        monkeypatch.setattr(
            provenance,
            "assess_physical_state",
            lambda metrics: assessment,
        )
    monkeypatch.setattr(
        provenance.teacher,
        "_require_worker_context",
        lambda: {
            "adapter": object(),
            "backend": object(),
            "action_space": object(),
            "reward_fn": object(),
            "physics_config": DEFAULT_PHYSICS_CONFIG,
            "task_config": {"max_steps": 5},
        },
    )


def _handoff_row() -> dict[str, object]:
    return {
        "scenario_id": 5,
        "step": 0,
        "selected_action_id": 0,
        "solved": False,
        "termination_reason": (
            TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value
        ),
    }


def test_teacher_handoff_with_hard_overload_uses_explicit_hard_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hard_evidence = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD
    )
    _patch_replay_context(
        monkeypatch,
        hard_evidence.assessment,
    )
    monkeypatch.setattr(
        provenance,
        "run_minimal_ac_redispatch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("redispatch must not run for a hard-overloaded handoff")
        ),
    )

    rows = [_handoff_row()]
    evidence = provenance._replay_terminal_evidence(5, rows)

    assert isinstance(evidence, TerminalOutcomeEvidence)
    assert evidence.termination_reason is (
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD
    )
    assert evidence.assessment is not None
    assert evidence.assessment.hard_overload_free is False
    assert rows[0]["redispatch_attempted"] is False


def test_teacher_handoff_becomes_validated_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    )
    secure = terminal_evidence(TerminationReason.SOLVED)
    assert handoff.assessment is not None
    assert secure.assessment is not None

    topology_utility = 0.35
    _patch_replay_context(
        monkeypatch,
        handoff.assessment,
        topology_utility=topology_utility,
    )
    monkeypatch.setattr(
        provenance,
        "run_minimal_ac_redispatch",
        lambda backend, state: MinimalRedispatchResult(
            opf_success=True,
            assessment=secure.assessment,
            message="validated",
            redispatch_l1_mw=20.0,
            redispatch_up_mw=10.0,
            redispatch_down_mw=10.0,
            redispatch_max_generator_delta_mw=6.0,
        ),
    )

    rows = [_handoff_row()]
    evidence = provenance._replay_terminal_evidence(5, rows)

    assert evidence.termination_reason is TerminationReason.REDISPATCH_VALIDATED
    assert evidence.redispatch_assessment is secure.assessment
    assert evidence.topology_utility == pytest.approx(topology_utility)
    assert rows[0]["redispatch_attempted"] is True
    assert rows[0]["redispatch_validated"] is True
    assert rows[0]["redispatch_l1_mw"] == pytest.approx(20.0)
    assert terminal_utility_from_outcome(
        False,
        evidence.termination_reason,
        evidence=evidence,
    )[0] == pytest.approx(topology_utility)


@pytest.mark.parametrize(
    "redispatch_result",
    [
        MinimalRedispatchResult(
            opf_success=False,
            assessment=None,
            message="infeasible",
        ),
        MinimalRedispatchResult(
            opf_success=True,
            assessment=terminal_evidence(
                TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
            ).assessment,
            message="unsafe",
            redispatch_l1_mw=5.0,
            redispatch_up_mw=2.5,
            redispatch_down_mw=2.5,
            redispatch_max_generator_delta_mw=2.5,
        ),
    ],
)
def test_failed_or_unsafe_redispatch_preserves_topology_utility(
    monkeypatch: pytest.MonkeyPatch,
    redispatch_result: MinimalRedispatchResult,
) -> None:
    handoff = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    )
    assert handoff.assessment is not None

    topology_utility = -0.35
    _patch_replay_context(
        monkeypatch,
        handoff.assessment,
        topology_utility=topology_utility,
    )
    monkeypatch.setattr(
        provenance,
        "run_minimal_ac_redispatch",
        lambda backend, state: redispatch_result,
    )

    rows = [_handoff_row()]
    evidence = provenance._replay_terminal_evidence(5, rows)

    assert evidence.termination_reason is (
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    )
    assert evidence.topology_utility == pytest.approx(topology_utility)
    assert rows[0]["redispatch_attempted"] is True
    assert rows[0]["redispatch_validated"] is False
    assert terminal_utility_from_outcome(
        False,
        evidence.termination_reason,
        evidence=evidence,
    )[0] == pytest.approx(topology_utility)


def test_topology_only_solved_episode_bypasses_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solved = terminal_evidence(TerminationReason.SOLVED)
    _patch_replay_context(
        monkeypatch,
        assessment=None,
        done=True,
        terminal_evidence_value=solved,
    )
    monkeypatch.setattr(
        provenance,
        "run_minimal_ac_redispatch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("redispatch must not run after topology-only success")
        ),
    )

    rows = [
        {
            "scenario_id": 5,
            "step": 0,
            "selected_action_id": 0,
            "solved": True,
            "termination_reason": TerminationReason.SOLVED.value,
        }
    ]
    evidence = provenance._replay_terminal_evidence(5, rows)

    assert evidence is solved
    assert rows[0]["redispatch_attempted"] is False

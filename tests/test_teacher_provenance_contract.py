from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.outcome_contract import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.termination import TerminationReason
from scripts.self_play import generate_impact_teacher_provenance as provenance
from tests.outcome_evidence_helpers import terminal_evidence


def _current_checkpoint_row() -> dict[str, object]:
    evidence = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    )
    return {
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
        "outcome_objective_version": 1,
        "outcome_value_target_contract_version": 5,
        "topology_action_contract_version": 1,
        "topology_action_config": "{}",
        "topology_action_config_fingerprint": "config-fingerprint",
        "action_layout": "[]",
        "action_layout_fingerprint": "layout-fingerprint",
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

    assert provenance._checkpoint_result_is_current(stale) is False
    assert provenance._checkpoint_result_is_current(current) is True
    assert provenance._checkpoint_result_is_current(
        {"ok": False, "reason": "exception", "rows": []}
    ) is False
    assert provenance._checkpoint_result_is_current(
        {"ok": False, "reason": "no_teacher_action_found", "rows": []}
    ) is True


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
    monkeypatch.setattr(
        provenance,
        "_replay_terminal_evidence",
        lambda scenario_id, rows: evidence,
    )
    monkeypatch.setattr(
        provenance.teacher,
        "_require_worker_context",
        lambda: {"action_space": action_space},
    )

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
    assert json.loads(row["bus_feature_columns"]) == ["Pd", "Qd"]
    assert json.loads(row["branch_feature_columns"]) == ["pf", "qf"]
    assert json.loads(row["topology_action_config"])[
        "require_connected_after_switch"
    ] is True
    assert len(json.loads(row["action_layout"])) == 3

    with np.load(state_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))

    assert metadata["run_id"] == row["run_id"]
    assert metadata["iteration"] == row["iteration"]
    assert metadata["episode_id"] == row["episode_id"]
    assert metadata["terminal_outcome_evidence"] == evidence.to_dict()
    assert metadata["topology_action_contract_version"] == 1
    assert metadata["episode_termination_reason"] == (
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value
    )


def test_teacher_handoff_with_hard_overload_uses_explicit_hard_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hard_evidence = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD
    )

    class FakeEnv:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.done = False
            self.terminal_outcome_evidence = None
            self.current_state = SimpleNamespace(metrics={})

        def reset(self, scenario_id: int) -> object:
            del scenario_id
            return self.current_state

        def step(self, action_id: int) -> object:
            raise AssertionError(f"unexpected action {action_id}")

    monkeypatch.setattr(provenance, "TopologySwitchingEnv", FakeEnv)
    monkeypatch.setattr(
        provenance,
        "assess_physical_state",
        lambda metrics: hard_evidence.assessment,
    )
    monkeypatch.setattr(
        provenance.teacher,
        "_require_worker_context",
        lambda: {
            "adapter": object(),
            "backend": object(),
            "action_space": object(),
            "reward_fn": object(),
            "task_config": {"max_steps": 5},
        },
    )

    evidence = provenance._replay_terminal_evidence(
        5,
        [
            {
                "scenario_id": 5,
                "step": 0,
                "selected_action_id": 0,
                "solved": False,
                "termination_reason": (
                    TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value
                ),
            }
        ],
    )

    assert isinstance(evidence, TerminalOutcomeEvidence)
    assert evidence.termination_reason is (
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD
    )
    assert evidence.assessment is not None
    assert evidence.assessment.hard_overload_free is False

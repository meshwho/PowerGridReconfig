from __future__ import annotations

import json

from grid_topology_ai.termination import TerminationReason
from scripts.self_play import generate_impact_teacher_parallel_fast as teacher
from scripts.self_play.generate_impact_teacher_provenance import (
    load_scenario_checkpoints,
)
from tests.outcome_evidence_helpers import terminal_evidence


def _current_row() -> dict[str, object]:
    evidence = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    )
    return {
        "state_id": "state-4",
        "run_id": "teacher-run",
        "iteration": 1,
        "episode_id": "teacher-episode",
        "scenario_id": 4,
        "solved": False,
        "termination_reason": evidence.termination_reason.value,
        "terminal_outcome_evidence_schema_version": 2,
        "terminal_outcome_evidence_json": evidence.to_json(),
        "outcome_objective_version": 1,
        "outcome_value_target_contract_version": 5,
        "topology_action_contract_version": 1,
        "topology_action_config": "{}",
        "topology_action_config_fingerprint": "config-fingerprint",
        "action_layout": "[]",
        "action_layout_fingerprint": "layout-fingerprint",
        "redispatch_attempted": False,
        "redispatch_opf_success": False,
        "redispatch_validated": False,
        "teacher_selection_mode": "epsilon_optimal_minimum_switch",
        "relative_physical_epsilon": 0.01,
        "teacher_best_physical_safety": 35.5,
        "teacher_selected_safety": 36.0,
        "teacher_selected_switch_count": 3,
        "teacher_retained_improvement_fraction": 0.995,
        "teacher_pareto_front_size": 5,
    }


def test_retryable_and_stale_checkpoints_are_not_restored(tmp_path) -> None:
    checkpoint_path = tmp_path / "teacher_checkpoint.jsonl"
    records = [
        {
            "version": teacher.CHECKPOINT_VERSION,
            "scenario_id": 1,
            "ok": False,
            "reason": "exception",
            "rows": [],
        },
        {
            "version": teacher.CHECKPOINT_VERSION,
            "scenario_id": 2,
            "ok": False,
            "reason": "no_teacher_action_found",
            "rows": [],
        },
        {
            "version": teacher.CHECKPOINT_VERSION,
            "scenario_id": 3,
            "ok": True,
            "reason": None,
            "rows": [{"state_id": "stale-state-3"}],
        },
        {
            "version": teacher.CHECKPOINT_VERSION,
            "scenario_id": 4,
            "ok": True,
            "reason": None,
            "rows": [_current_row()],
        },
    ]
    checkpoint_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    restored = load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=[1, 2, 3, 4],
    )

    assert 1 not in restored
    assert restored[2]["reason"] == "no_teacher_action_found"
    assert 3 not in restored
    assert restored[4]["ok"] is True

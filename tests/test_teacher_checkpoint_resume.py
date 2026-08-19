from __future__ import annotations

import json

import numpy as np

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
    topology_action_provenance,
)
from grid_topology_ai.outcome_contract import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
)
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.topology_actions import (
    ActionSpaceConfig,
    build_branch_action_slots,
)
from scripts.self_play import generate_impact_teacher_redispatch_runtime as teacher
from tests.outcome_evidence_helpers import terminal_evidence


def _selection_fields() -> dict[str, object]:
    return {
        "teacher_selection_mode": "redispatch_aware_epsilon_minimum_switch",
        "relative_physical_epsilon": 0.01,
        "teacher_best_physical_safety": 35.5,
        "teacher_selected_safety": 36.0,
        "teacher_selected_switch_count": 3,
        "teacher_retained_improvement_fraction": 0.995,
        "teacher_pareto_front_size": 5,
        "terminal_redispatch_relative_epsilon": 0.01,
        "terminal_redispatch_absolute_epsilon_mw": 1.0,
        "min_meaningful_safety_improvement": 1.0,
        "teacher_terminal_selection_applied": False,
        "teacher_terminal_candidate_count": 0,
        "teacher_terminal_pareto_front_size": 0,
    }


def _current_row(scenario_id: int = 4) -> dict[str, object]:
    evidence = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    )
    action_provenance = topology_action_provenance(
        ActionSpaceConfig(),
        build_branch_action_slots(
            np.array([10, 20], dtype=np.int64)
        ),
    )

    return {
        **physics_provenance(DEFAULT_PHYSICS_CONFIG),
        **action_provenance,
        "state_id": f"state-{scenario_id}",
        "run_id": "teacher-run",
        "iteration": 1,
        "episode_id": f"teacher-episode-{scenario_id}",
        "scenario_id": int(scenario_id),
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


def test_retryable_and_stale_checkpoints_are_not_restored(tmp_path) -> None:
    checkpoint_path = tmp_path / "teacher_checkpoint.jsonl"
    stale_contract_row = _current_row(5)
    stale_contract_row["outcome_value_target_contract_version"] = (
        OUTCOME_VALUE_TARGET_CONTRACT_VERSION - 1
    )

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
            "rows": [_current_row(4)],
        },
        {
            "version": teacher.CHECKPOINT_VERSION,
            "scenario_id": 5,
            "ok": True,
            "reason": None,
            "rows": [stale_contract_row],
        },
    ]
    checkpoint_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    restored = teacher.load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=[1, 2, 3, 4, 5],
    )

    assert 1 not in restored
    assert restored[2]["reason"] == "no_teacher_action_found"
    assert 3 not in restored
    assert restored[4]["ok"] is True
    assert 5 not in restored


def test_teacher_workload_uses_current_exact_cache_counters() -> None:
    workload = teacher._search_workload(
        before={
            "hits": 10,
            "misses": 20,
            "l1_hits": 6,
            "l1_misses": 24,
            "l2_hits": 4,
            "l2_misses": 20,
            "negative_hits": 1,
            "evictions": 2,
            "l2_enabled": True,
            "stock_runpf_calls": 30,
            "q_limit_resolves": 4,
        },
        after={
            "hits": 18,
            "misses": 25,
            "l1_hits": 12,
            "l1_misses": 31,
            "l2_hits": 6,
            "l2_misses": 25,
            "negative_hits": 2,
            "evictions": 5,
            "l2_enabled": True,
            "stock_runpf_calls": 37,
            "q_limit_resolves": 6,
        },
        logical_evaluations=15,
    )

    assert workload["cache_hits"] == 8
    assert workload["cache_misses"] == 5
    assert workload["l1_hits"] == 6
    assert workload["l2_hits"] == 2
    assert workload["negative_hits"] == 1
    assert workload["stock_runpf_calls"] == 7
    assert workload["q_limit_resolves"] == 2
    assert workload["solves_per_cache_miss"] == 7 / 5
    assert "tolerant_cache_hits" not in workload
    assert "cold_start_misses" not in workload

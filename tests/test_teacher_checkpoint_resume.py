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
from scripts.self_play import generate_impact_teacher_parallel_fast as teacher
from scripts.self_play.generate_impact_teacher_provenance import (
    _search_workload,
    load_scenario_checkpoints,
)
from tests.outcome_evidence_helpers import terminal_evidence


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

    restored = load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=[1, 2, 3, 4, 5],
    )

    assert 1 not in restored
    assert restored[2]["reason"] == "no_teacher_action_found"
    assert 3 not in restored
    assert restored[4]["ok"] is True
    assert 5 not in restored


def test_teacher_workload_reports_cache_reuse_modes() -> None:
    workload = _search_workload(
        before={
            "hits": 10,
            "misses": 20,
            "exact_cache_hits": 7,
            "tolerant_cache_hits": 3,
            "warm_start_hits": 8,
            "cold_start_misses": 12,
            "stock_runpf_calls": 60,
            "q_limit_resolves": 40,
        },
        after={
            "hits": 19,
            "misses": 31,
            "exact_cache_hits": 13,
            "tolerant_cache_hits": 6,
            "warm_start_hits": 15,
            "cold_start_misses": 16,
            "stock_runpf_calls": 95,
            "q_limit_resolves": 64,
        },
        logical_evaluations=20,
    )

    assert workload["cache_hits"] == 9
    assert workload["cache_misses"] == 11
    assert workload["exact_cache_hits"] == 6
    assert workload["tolerant_cache_hits"] == 3
    assert workload["warm_start_hits"] == 7
    assert workload["cold_start_misses"] == 4
    assert workload["cache_hit_rate"] == 9 / 20
    assert workload["warm_start_rate"] == 7 / 11
    assert workload["stock_runpf_calls"] == 35
    assert workload["q_limit_resolves"] == 24
    assert workload["solves_per_cache_miss"] == 35 / 11

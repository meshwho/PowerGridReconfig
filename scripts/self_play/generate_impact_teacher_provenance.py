from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from grid_topology_ai.contracts import (
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    topology_action_provenance,
)
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.outcome_contract import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
    TerminalOutcomeEvidence,
    redispatch_status_for_reason,
)
from grid_topology_ai.physical_objective import assess_physical_state
from grid_topology_ai.redispatch import (
    MinimalRedispatchResult,
    empty_redispatch_diagnostics,
    run_minimal_ac_redispatch,
)
from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
)
from grid_topology_ai.topology_actions import build_branch_action_slots
from scripts.self_play import generate_impact_teacher_parallel_fast as teacher


_original_make_task_config = teacher.make_task_config
_original_load_scenario_checkpoints = teacher.load_scenario_checkpoints
_original_process_scenario_batch = teacher.process_scenario_batch
_original_append_scenario_checkpoint = teacher.append_scenario_checkpoint
_original_planner_search = teacher.ImpactBeamSearchPlanner.search

_REDISPATCH_ROW_FIELDS = (
    "redispatch_attempted",
    "redispatch_opf_success",
    "redispatch_validated",
    "redispatch_l1_mw",
    "redispatch_up_mw",
    "redispatch_down_mw",
    "redispatch_max_generator_delta_mw",
    "redispatch_message",
)
_REQUIRED_CHECKPOINT_ROW_FIELDS = (
    "run_id",
    "iteration",
    "episode_id",
    "terminal_outcome_evidence_schema_version",
    "terminal_outcome_evidence_json",
    "outcome_objective_version",
    "outcome_value_target_contract_version",
    "topology_action_contract_version",
    "topology_action_config",
    "topology_action_config_fingerprint",
    "action_layout",
    "action_layout_fingerprint",
    "redispatch_attempted",
    "redispatch_opf_success",
    "redispatch_validated",
)

_SEARCH_WORKLOAD_BY_SCENARIO: dict[int, dict[str, object]] = {}
_PARENT_WORKLOAD_BY_SCENARIO: dict[int, dict[str, object]] = {}


def make_task_config(args: argparse.Namespace) -> dict[str, Any]:
    task_config = _original_make_task_config(args)
    physics_config = teacher.PhysicsConfig.from_mapping(
        task_config["physics_config"]
    )
    task_config.update(teacher.physics_provenance(physics_config))
    return task_config


def _checkpoint_result_is_current(result: dict[str, Any]) -> bool:
    if not bool(result.get("ok", False)):
        return result.get("reason") != "exception"

    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        return False

    for row in rows:
        if not isinstance(row, dict):
            return False
        if any(row.get(field) is None for field in _REQUIRED_CHECKPOINT_ROW_FIELDS):
            return False

        run_id = row.get("run_id")
        episode_id = row.get("episode_id")
        iteration = row.get("iteration")
        if not isinstance(run_id, str) or not run_id.strip():
            return False
        if not isinstance(episode_id, str) or not episode_id.strip():
            return False
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            return False
        if iteration <= 0:
            return False

        try:
            evidence = TerminalOutcomeEvidence.from_json(
                row["terminal_outcome_evidence_json"]
            )
            reason = parse_termination_reason(
                row.get("termination_reason"),
                allow_none=False,
            )
        except (TypeError, ValueError):
            return False

        if (
            evidence.solved is not bool(row.get("solved"))
            or evidence.termination_reason is not reason
        ):
            return False

    return True


def load_scenario_checkpoints(
    checkpoint_path: Path,
    allowed_scenario_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    results = _original_load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=allowed_scenario_ids,
    )
    return {
        scenario_id: result
        for scenario_id, result in results.items()
        if _checkpoint_result_is_current(result)
    }


def _counter_delta(
    before: dict[str, object],
    after: dict[str, object],
    key: str,
) -> int:
    return max(int(after.get(key, 0)) - int(before.get(key, 0)), 0)


def _search_workload(
    before: dict[str, object],
    after: dict[str, object],
    logical_evaluations: int,
) -> dict[str, object]:
    cache_hits = _counter_delta(before, after, "hits")
    cache_misses = _counter_delta(before, after, "misses")
    stock_runpf_calls = _counter_delta(before, after, "stock_runpf_calls")
    q_limit_resolves = _counter_delta(before, after, "q_limit_resolves")
    cache_lookups = cache_hits + cache_misses

    return {
        "logical_evaluations": int(logical_evaluations),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "cache_hit_rate": (
            float(cache_hits) / float(cache_lookups)
            if cache_lookups > 0
            else 0.0
        ),
        "stock_runpf_calls": int(stock_runpf_calls),
        "q_limit_resolves": int(q_limit_resolves),
        "solves_per_cache_miss": (
            float(stock_runpf_calls) / float(cache_misses)
            if cache_misses > 0
            else 0.0
        ),
    }


def _instrumented_planner_search(self, env, scenario_id: int):
    backend = env.backend
    before = backend.performance_info()
    result = None

    try:
        result = _original_planner_search(
            self,
            env=env,
            scenario_id=int(scenario_id),
        )
        return result
    finally:
        after = backend.performance_info()
        logical_evaluations = (
            int(result.evaluated_actions)
            if result is not None
            else int(getattr(self, "evaluated_actions", 0))
        )
        _SEARCH_WORKLOAD_BY_SCENARIO[int(scenario_id)] = _search_workload(
            before=before,
            after=after,
            logical_evaluations=logical_evaluations,
        )


def _install_worker_instrumentation() -> None:
    if teacher.ImpactBeamSearchPlanner.search is not _instrumented_planner_search:
        teacher.ImpactBeamSearchPlanner.search = _instrumented_planner_search


def append_scenario_checkpoint(
    checkpoint_path: Path,
    result: dict[str, Any],
) -> None:
    _original_append_scenario_checkpoint(
        checkpoint_path=checkpoint_path,
        result=result,
    )

    performance = result.get("performance")
    if isinstance(performance, dict):
        _PARENT_WORKLOAD_BY_SCENARIO[int(result["scenario_id"])] = dict(
            performance
        )


def _print_power_flow_workload_summary() -> None:
    print("\n" + "=" * 100)
    print("Power-flow workload for scenarios processed in this run")
    print("=" * 100)

    if not _PARENT_WORKLOAD_BY_SCENARIO:
        print("Instrumented scenarios: 0")
        print("No new scenarios were processed in this run.")
        return

    items = list(_PARENT_WORKLOAD_BY_SCENARIO.values())
    logical_evaluations = sum(
        int(item.get("logical_evaluations", 0))
        for item in items
    )
    cache_hits = sum(int(item.get("cache_hits", 0)) for item in items)
    cache_misses = sum(int(item.get("cache_misses", 0)) for item in items)
    stock_runpf_calls = sum(
        int(item.get("stock_runpf_calls", 0))
        for item in items
    )
    q_limit_resolves = sum(
        int(item.get("q_limit_resolves", 0))
        for item in items
    )
    cache_lookups = cache_hits + cache_misses
    cache_hit_rate = (
        float(cache_hits) / float(cache_lookups)
        if cache_lookups > 0
        else 0.0
    )
    solves_per_cache_miss = (
        float(stock_runpf_calls) / float(cache_misses)
        if cache_misses > 0
        else 0.0
    )

    print(f"Instrumented scenarios:    {len(items)}")
    print(f"Logical evaluations:       {logical_evaluations}")
    print(f"PF cache hits:             {cache_hits}")
    print(f"PF cache misses:           {cache_misses}")
    print(f"Cache hit rate:            {cache_hit_rate:.1%}")
    print(f"Stock PYPOWER solves:      {stock_runpf_calls}")
    print(f"Q-limit re-solves:         {q_limit_resolves}")
    print(f"Solves / cache miss:       {solves_per_cache_miss:.3f}")


def _worker_run_id() -> str:
    ctx = teacher._require_worker_context()
    states_dir = Path(ctx["state_store"].output_dir).resolve()
    payload = {
        "states_dir": str(states_dir),
        "task_config": ctx["task_config"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"impact_teacher_{digest[:24]}"


def _set_redispatch_diagnostics(
    rows: list[dict[str, Any]],
    result: MinimalRedispatchResult | None,
) -> None:
    diagnostics = (
        empty_redispatch_diagnostics()
        if result is None
        else result.diagnostics()
    )
    for row in rows:
        row.update(diagnostics)


def _replay_terminal_evidence(
    scenario_id: int,
    rows: list[dict[str, Any]],
) -> TerminalOutcomeEvidence:
    ctx = teacher._require_worker_context()
    _set_redispatch_diagnostics(rows, None)

    ordered_rows = sorted(rows, key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered_rows]
    if steps != list(range(len(steps))):
        raise ValueError(
            f"Teacher scenario {scenario_id} has non-contiguous steps: {steps}."
        )

    solved_values = {bool(row["solved"]) for row in ordered_rows}
    reason_values = {
        parse_termination_reason(
            row.get("termination_reason"),
            allow_none=False,
        )
        for row in ordered_rows
    }
    if len(solved_values) != 1 or len(reason_values) != 1:
        raise ValueError(
            f"Teacher scenario {scenario_id} contains mixed terminal outcomes."
        )

    solved = solved_values.pop()
    recorded_reason = reason_values.pop()
    assert recorded_reason is not None

    env = TopologySwitchingEnv(
        adapter=ctx["adapter"],
        backend=ctx["backend"],
        action_space=ctx["action_space"],
        reward_fn=ctx["reward_fn"],
        max_steps=int(ctx["task_config"]["max_steps"]),
    )
    env.reset(int(scenario_id))

    for row in ordered_rows:
        action_id = int(row["selected_action_id"])
        if action_id == 0:
            break
        if env.done:
            raise ValueError(
                f"Teacher scenario {scenario_id} contains an action after termination."
            )
        step_result = env.step(action_id)
        if not step_result.power_flow_success:
            raise ValueError(
                f"Teacher scenario {scenario_id} replay hit a power-flow failure."
            )

    if (
        env.done
        and env.terminal_outcome_evidence is not None
        and env.terminal_outcome_evidence.solved is solved
        and env.terminal_outcome_evidence.termination_reason is recorded_reason
    ):
        return env.terminal_outcome_evidence

    final_state = env.current_state
    if final_state is None:
        raise ValueError(
            f"Teacher scenario {scenario_id} has no terminal state for provenance."
        )

    assessment = assess_physical_state(final_state.metrics)
    reason = recorded_reason
    if (
        reason is TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
        and not assessment.hard_overload_free
    ):
        reason = TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD

    if (
        reason is TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
        and assessment.hard_overload_free
    ):
        redispatch = run_minimal_ac_redispatch(
            ctx["backend"],
            final_state,
        )
        _set_redispatch_diagnostics(rows, redispatch)

        if redispatch.validated:
            assert redispatch.assessment is not None
            validated_reason = TerminationReason.REDISPATCH_VALIDATED
            return TerminalOutcomeEvidence(
                solved=False,
                termination_reason=validated_reason,
                assessment=assessment,
                redispatch_status=redispatch_status_for_reason(validated_reason),
                redispatch_assessment=redispatch.assessment,
            )

    return TerminalOutcomeEvidence(
        solved=solved,
        termination_reason=reason,
        assessment=assessment,
        redispatch_status=redispatch_status_for_reason(reason),
    )


def _load_state_file(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.array(data[name], copy=True)
            for name in data.files
        }

    if "metadata_json" not in arrays:
        raise ValueError(f"State file is missing metadata_json: {path}")

    metadata = json.loads(str(np.asarray(arrays["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"State metadata_json must contain an object: {path}")
    return arrays, metadata


def _write_state_metadata(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    arrays["metadata_json"] = np.array(
        json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    temp_path = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(temp_path, **arrays)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _csv_topology_provenance(provenance: dict[str, object]) -> dict[str, object]:
    return {
        "topology_action_contract_version": provenance[
            "topology_action_contract_version"
        ],
        "topology_action_config": json.dumps(
            provenance["topology_action_config"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "topology_action_config_fingerprint": provenance[
            "topology_action_config_fingerprint"
        ],
        "action_layout": json.dumps(
            provenance["action_layout"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "action_layout_fingerprint": provenance[
            "action_layout_fingerprint"
        ],
    }


def _json_feature_columns(row: dict[str, Any]) -> None:
    for field in ("bus_feature_columns", "branch_feature_columns"):
        value = row.get(field)
        if isinstance(value, str):
            continue
        row[field] = json.dumps(
            value,
            separators=(",", ":"),
            allow_nan=False,
        )


def _finalize_success_result(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Successful teacher result must contain rows.")

    scenario_id = int(result["scenario_id"])
    if any(int(row["scenario_id"]) != scenario_id for row in rows):
        raise ValueError(
            f"Teacher result for scenario {scenario_id} contains mixed scenario IDs."
        )

    evidence = _replay_terminal_evidence(scenario_id, rows)
    run_id = _worker_run_id()
    iteration = 1
    episode_id = f"{run_id}_scenario_{scenario_id:06d}"
    reason_value = evidence.termination_reason.value
    evidence_json = evidence.to_json()
    evidence_mapping = evidence.to_dict()
    ctx = teacher._require_worker_context()

    for row in rows:
        state_path = Path(str(row["state_path"]))
        arrays, metadata = _load_state_file(state_path)
        branch_ids = np.asarray(arrays["branch_ids"], dtype=np.int64)
        layout = build_branch_action_slots(branch_ids)
        action_provenance = topology_action_provenance(
            ctx["action_space"].config,
            layout,
        )

        row.update(
            {
                "run_id": run_id,
                "iteration": iteration,
                "episode_id": episode_id,
                "solved": evidence.solved,
                "done": True,
                "termination_reason": reason_value,
                "terminal_outcome_evidence_schema_version": (
                    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
                ),
                "terminal_outcome_evidence_json": evidence_json,
                "outcome_objective_version": OUTCOME_OBJECTIVE_VERSION,
                "outcome_value_target_contract_version": (
                    OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                ),
                **_csv_topology_provenance(action_provenance),
            }
        )
        _json_feature_columns(row)

        if int(row["selected_action_id"]) == 0:
            row["step_termination_reason"] = reason_value

        metadata.update(
            {
                "run_id": run_id,
                "iteration": iteration,
                "episode_id": episode_id,
                "episode_done": True,
                "episode_solved": evidence.solved,
                "episode_termination_reason": reason_value,
                "terminal_outcome_evidence_schema_version": (
                    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
                ),
                "terminal_outcome_evidence": evidence_mapping,
                "outcome_objective_version": OUTCOME_OBJECTIVE_VERSION,
                "outcome_value_target_contract_version": (
                    OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                ),
                **{
                    field: row.get(field)
                    for field in _REDISPATCH_ROW_FIELDS
                },
                **action_provenance,
            }
        )
        if int(row["selected_action_id"]) == 0:
            metadata["step_termination_reason"] = reason_value

        _write_state_metadata(
            state_path,
            arrays,
            metadata,
        )

    return result


def process_scenario_batch(
    scenario_ids: list[int],
) -> list[dict[str, Any]]:
    _install_worker_instrumentation()
    results = _original_process_scenario_batch(scenario_ids)

    finalized: list[dict[str, Any]] = []
    for result in results:
        scenario_id = int(result["scenario_id"])
        performance = _SEARCH_WORKLOAD_BY_SCENARIO.pop(scenario_id, None)
        if performance is not None:
            result["performance"] = performance

        if bool(result.get("ok", False)):
            result = _finalize_success_result(result)
        finalized.append(result)

    return finalized


def main() -> None:
    _PARENT_WORKLOAD_BY_SCENARIO.clear()
    _install_worker_instrumentation()
    teacher.make_task_config = make_task_config
    teacher.load_scenario_checkpoints = load_scenario_checkpoints
    teacher.process_scenario_batch = process_scenario_batch
    teacher.append_scenario_checkpoint = append_scenario_checkpoint
    teacher.main()
    _print_power_flow_workload_summary()


if __name__ == "__main__":
    main()

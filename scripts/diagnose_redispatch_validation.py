from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.physics.redispatch import run_minimal_ac_redispatch
from grid_topology_ai.runtime import (
    build_memory_mapped_teacher_context,
    ensure_runtime_scenario_store,
)


FEASIBILITY_FIELDS = (
    "thermal_feasible",
    "hard_overload_free",
    "voltage_feasible",
    "generator_p_feasible",
    "generator_q_feasible",
    "angle_difference_feasible",
    "topology_connected",
    "power_flow_converged",
    "all_values_finite",
)

ASSESSMENT_FIELDS = (
    "max_loading_percent",
    "num_overloaded_branches",
    "num_hard_overloaded_branches",
    "total_voltage_violation",
    "num_generator_p_violations",
    "total_generator_p_violation_mw",
    "num_generator_q_violations",
    "total_generator_q_violation_mvar",
    "num_angle_difference_violations",
    "total_angle_difference_violation_degrees",
)


def _load_scenario_ids(
    transitions_path: Path,
    *,
    difficulty_class: str | None,
    limit: int | None,
) -> list[int]:
    transitions = pd.read_csv(transitions_path)
    if "scenario_id" not in transitions.columns:
        raise ValueError("Transitions CSV must contain scenario_id.")

    if difficulty_class is not None:
        if "difficulty_class" not in transitions.columns:
            raise ValueError(
                "Transitions CSV must contain difficulty_class when a class is requested."
            )
        normalized = transitions["difficulty_class"].astype(str).str.strip().str.lower()
        transitions = transitions.loc[normalized == difficulty_class]

    scenario_ids = [
        int(value)
        for value in transitions["scenario_id"].drop_duplicates().tolist()
    ]
    if limit is not None:
        scenario_ids = scenario_ids[: int(limit)]
    return scenario_ids


def _assessment_failure_key(assessment) -> str:
    if assessment is None:
        return "assessment_missing"
    failed = [
        field.removesuffix("_feasible")
        for field in FEASIBILITY_FIELDS
        if not bool(getattr(assessment, field))
    ]
    return "+".join(failed) if failed else "none"


def _row_from_result(scenario_id: int, result) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario_id": int(scenario_id),
        "opf_success": bool(result.opf_success),
        "validated": bool(result.validated),
        "message": str(result.message),
        "redispatch_l1_mw": result.redispatch_l1_mw,
        "redispatch_up_mw": result.redispatch_up_mw,
        "redispatch_down_mw": result.redispatch_down_mw,
        "redispatch_max_generator_delta_mw": result.redispatch_max_generator_delta_mw,
        "failure_key": _assessment_failure_key(result.assessment),
    }
    if result.assessment is None:
        for field in (*FEASIBILITY_FIELDS, *ASSESSMENT_FIELDS):
            row[field] = None
        return row

    for field in (*FEASIBILITY_FIELDS, *ASSESSMENT_FIELDS):
        row[field] = getattr(result.assessment, field)
    return row


def _print_summary(rows: pd.DataFrame) -> None:
    total = len(rows)
    opf_success = int(rows["opf_success"].sum())
    validated = int(rows["validated"].sum())
    rejected_after_success = rows[rows["opf_success"] & ~rows["validated"]]

    print("\nRedispatch validation probe")
    print("=" * 80)
    print(f"Scenarios:                 {total}")
    print(f"OPF success:               {opf_success} ({100.0 * opf_success / total:.2f}%)")
    print(f"Validated:                 {validated} ({100.0 * validated / total:.2f}%)")
    if opf_success:
        print(
            "Validated / OPF success: "
            f"{validated}/{opf_success} ({100.0 * validated / opf_success:.2f}%)"
        )
    print(f"Rejected after OPF success:{len(rejected_after_success):9d}")

    if rejected_after_success.empty:
        return

    print("\nFailure combinations after OPF success:")
    counts = Counter(rejected_after_success["failure_key"].astype(str))
    for key, count in counts.most_common():
        print(
            f"  {key:45s} {count:6d}  "
            f"{100.0 * count / len(rejected_after_success):7.2f}%"
        )

    print("\nConstraint failure rates after OPF success (%):")
    for field in FEASIBILITY_FIELDS:
        available = rejected_after_success[field].dropna()
        if available.empty:
            print(f"  {field:35s} missing")
            continue
        failed = int((~available.astype(bool)).sum())
        print(
            f"  {field:35s} {failed:6d}/{len(available):6d} "
            f"{100.0 * failed / len(available):7.2f}%"
        )

    p_excess = pd.to_numeric(
        rejected_after_success["total_generator_p_violation_mw"],
        errors="coerce",
    ).dropna()
    if not p_excess.empty:
        print("\nGenerator P excess after OPF success (MW):")
        print(
            p_excess.quantile([0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0])
            .to_string()
        )

    missing = rejected_after_success[
        rejected_after_success["failure_key"] == "assessment_missing"
    ]
    if not missing.empty:
        print("\nOPF-success results rejected before assessment:")
        print(missing["message"].value_counts(dropna=False).head(20).to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe why PYPOWER AC OPF success results fail strict redispatch validation."
        )
    )
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument(
        "--difficulty-class",
        choices=("simple", "medium", "hard"),
        default=None,
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--physics-config-json",
        type=Path,
        default=None,
        help="Optional JSON object overriding the default PhysicsConfig fields.",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and int(args.limit) <= 0:
        raise ValueError("--limit must be positive.")

    physics_payload = DEFAULT_PHYSICS_CONFIG.to_dict()
    if args.physics_config_json is not None:
        overrides = json.loads(args.physics_config_json.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("--physics-config-json must contain a JSON object.")
        physics_payload.update(overrides)

    scenario_ids = _load_scenario_ids(
        args.transitions,
        difficulty_class=args.difficulty_class,
        limit=args.limit,
    )
    if not scenario_ids:
        raise ValueError("No scenarios selected.")

    runtime_store_dir = ensure_runtime_scenario_store(args.raw_dir)
    task_config = {
        "physics_config": physics_payload,
        "disable_cache": bool(args.disable_cache),
    }

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="redispatch-validation-probe-"
    ) as states_dir:
        context = build_memory_mapped_teacher_context(
            runtime_store_dir=runtime_store_dir,
            states_dir=states_dir,
            task_config=task_config,
            scenario_ids=scenario_ids,
            memory_registry=None,
        )
        env = TopologySwitchingEnv(
            adapter=context["adapter"],
            backend=context["backend"],
            action_space=context["action_space"],
            reward_fn=context["reward_fn"],
            max_steps=1,
        )

        for index, scenario_id in enumerate(scenario_ids, start=1):
            state = env.reset(int(scenario_id))
            result = run_minimal_ac_redispatch(context["backend"], state)
            row = _row_from_result(scenario_id, result)
            rows.append(row)
            print(
                f"[{index:4d}/{len(scenario_ids):4d}] scenario {scenario_id}: "
                f"opf_success={result.opf_success} validated={result.validated} "
                f"failure={row['failure_key']}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    _print_summary(frame)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"\nSaved diagnostics: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

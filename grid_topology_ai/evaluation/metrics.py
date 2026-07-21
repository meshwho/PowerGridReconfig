from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.contracts import EVALUATION_METRICS_CONTRACT_VERSION
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.physical_objective import physical_objective_contract
from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
    validate_outcome_invariants,
)


def compute_safety_score(
    row: dict[str, Any],
    physics_config: PhysicsConfig | None = None,
) -> float:
    config = physics_config or DEFAULT_PHYSICS_CONFIG
    score = 0.0
    reason = parse_termination_reason(row.get("termination_reason"))
    solved = bool(row.get("solved", False))
    physically_secure = bool(row.get("physically_secure", False))
    validate_outcome_invariants(
        solved=solved,
        termination_reason=reason,
        physically_secure=physically_secure,
    )
    final_loading = float(row.get("final_max_loading_percent", 999.0))
    overloaded = int(row.get("final_num_overloaded_branches", 99))
    hard = int(row.get("final_num_hard_overloaded_branches", 99))
    discounted_return = float(row.get("discounted_return", 0.0))

    if solved:
        score += 1000.0
    elif reason is TerminationReason.HANDOFF_TO_REDISPATCH and hard == 0:
        score += 500.0
    elif reason is TerminationReason.MAX_STEPS_REACHED:
        score -= 300.0
    elif reason is TerminationReason.POWER_FLOW_FAILED:
        score -= 1000.0
    else:
        score -= 100.0

    score -= 300.0 * hard
    score -= 50.0 * overloaded

    overload_threshold = (
        config.overload_limit_percent
        + config.thermal_tolerance_percent
    )
    if final_loading > overload_threshold:
        score -= 5.0 * (
            final_loading - config.overload_limit_percent
        )

    score += 0.05 * discounted_return
    return float(score)


def attach_difficulty_metadata(
    df: pd.DataFrame,
    transitions_path: Path,
) -> pd.DataFrame:
    transitions = pd.read_csv(transitions_path)

    if "difficulty_class" not in transitions.columns:
        return df

    if "scenario_id" not in transitions.columns:
        return df

    difficulty = (
        transitions[["scenario_id", "difficulty_class"]]
        .drop_duplicates(subset=["scenario_id"])
        .copy()
    )
    difficulty["scenario_id"] = difficulty["scenario_id"].astype(int)
    return df.merge(difficulty, on="scenario_id", how="left")


def print_row(row: dict[str, Any]) -> None:
    print(
        f"Scenario {int(row['scenario_id']):>5} | "
        f"reason={row['termination_reason']} | "
        f"solved={row['solved']} | "
        f"steps={row['steps']} | "
        f"branches={row['branches']} | "
        f"final_loading={float(row['final_max_loading_percent']):.2f}% | "
        f"overloaded={row['final_num_overloaded_branches']} | "
        f"hard={row['final_num_hard_overloaded_branches']} | "
        f"R={float(row['discounted_return']):.2f} | "
        f"score={float(row['safety_score']):.2f}"
    )


def _safe_mean(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None

    value = series.mean()

    if pd.isna(value):
        return None

    return float(value)


def build_evaluation_metrics(
    df: pd.DataFrame,
    failed_results: list[dict[str, Any]],
    requested_scenarios: int,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    # Empty task configs are retained only for the metrics-only public helper.
    # Evaluation workers always provide the complete provenance payload.
    physics_config = (
        PhysicsConfig.from_mapping(task_config["physics_config"])
        if "physics_config" in task_config
        else DEFAULT_PHYSICS_CONFIG
    )
    if (
        "physics_config_fingerprint" in task_config
        and physics_config.fingerprint() != task_config["physics_config_fingerprint"]
    ):
        raise ValueError("Evaluation metrics received mismatched PhysicsConfig.")
    solved = df["solved"].astype(bool)
    physically_secure = df["physically_secure"].astype(bool)
    if not solved.equals(physically_secure):
        raise ValueError(
            "Evaluation rows violate the outcome contract: solved must equal "
            "physically_secure. Regenerate evaluation metrics."
        )
    for index, row in df.iterrows():
        validate_outcome_invariants(
            solved=bool(row["solved"]),
            termination_reason=row["termination_reason"],
            physically_secure=bool(row["physically_secure"]),
        )
    termination_counts = {
        str(key): int(value)
        for key, value in df["termination_reason"]
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    evaluated_scenarios = int(len(df))
    requested_count = int(requested_scenarios)
    failed_scenarios = int(len(failed_results))
    solve_count = int(physically_secure.sum())

    def rate(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    component_fields = (
        "power_flow_converged",
        "all_values_finite",
        "topology_connected",
        "thermal_solved",
        "thermal_feasible",
        "hard_overload_free",
        "voltage_feasible",
        "generator_p_feasible",
        "generator_q_feasible",
        "angle_difference_feasible",
        "physically_secure",
    )
    component_counts = {
        field: int(df[field].astype(bool).sum()) for field in component_fields
    }
    hard_overload_free_count = component_counts["hard_overload_free"]
    voltage_feasible_count = component_counts["voltage_feasible"]
    physically_secure_count = component_counts["physically_secure"]
    safe_handoff_count = int(df["safe_handoff"].astype(bool).sum())
    unsafe_terminal_state_count = int(
        df["unsafe_terminal_state"].astype(bool).sum()
    )
    power_flow_failure_count = int(
        (df["termination_reason"] == TerminationReason.POWER_FLOW_FAILED.value).sum()
    )

    metrics: dict[str, Any] = {
        "evaluation_metrics_contract_version": (
            EVALUATION_METRICS_CONTRACT_VERSION
        ),
        "requested_scenarios": requested_count,
        "evaluated_scenarios": evaluated_scenarios,
        "failed_scenarios": failed_scenarios,
        "solve_count": solve_count,
        "solve_rate": rate(solve_count, evaluated_scenarios),
        "pf_alg": physics_config.pf_alg,
        "physics_config_contract_version": task_config.get(
            "physics_config_contract_version", 1
        ),
        "physics_config": physics_config.to_dict(),
        "physics_config_fingerprint": physics_config.fingerprint(),
        "physical_objective_contract": physical_objective_contract(physics_config),
        "evaluation_coverage_rate": rate(evaluated_scenarios, requested_count),
        "solve_rate_requested": rate(solve_count, requested_count),
        "failed_scenario_rate_requested": rate(
            failed_scenarios, requested_count
        ),
        "hard_overload_free_count": hard_overload_free_count,
        "hard_overload_free_rate": rate(
            hard_overload_free_count, evaluated_scenarios
        ),
        "voltage_feasible_count": voltage_feasible_count,
        "voltage_feasible_rate": rate(voltage_feasible_count, evaluated_scenarios),
        "physically_secure_count": physically_secure_count,
        "physically_secure_rate": rate(
            physically_secure_count, evaluated_scenarios
        ),
        "safe_handoff_count": safe_handoff_count,
        "safe_handoff_rate": rate(safe_handoff_count, evaluated_scenarios),
        "unsafe_terminal_state_count": unsafe_terminal_state_count,
        "unsafe_terminal_state_rate": rate(
            unsafe_terminal_state_count, evaluated_scenarios
        ),
        "power_flow_failure_count": power_flow_failure_count,
        "power_flow_failure_rate": rate(
            power_flow_failure_count, evaluated_scenarios
        ),
        "avg_steps": _safe_mean(df["steps"]),
        "avg_steps_to_solve": _safe_mean(df.loc[solved, "steps"]),
        "avg_discounted_return": _safe_mean(df["discounted_return"]),
        "avg_final_loading_percent": _safe_mean(df["final_max_loading_percent"]),
        "avg_final_num_overloaded_branches": _safe_mean(
            df["final_num_overloaded_branches"]
        ),
        "avg_final_num_hard_overloaded_branches": _safe_mean(
            df["final_num_hard_overloaded_branches"]
        ),
        "avg_safety_score": _safe_mean(df["safety_score"]),
        "total_safety_score": float(df["safety_score"].sum()),
        "termination_reason_counts": termination_counts,
        "task_config": dict(task_config),
    }

    for field, count in component_counts.items():
        metrics[f"{field}_count"] = count
        metrics[f"{field}_rate"] = rate(count, evaluated_scenarios)

    for field in (
        "num_low_voltage_buses",
        "num_high_voltage_buses",
        "num_generator_p_violations",
        "num_generator_q_violations",
        "num_angle_difference_violations",
        "total_thermal_overload_mva",
        "total_generator_p_violation_mw",
        "total_generator_q_violation_mvar",
        "total_angle_difference_violation_degrees",
        "total_voltage_violation",
    ):
        metrics[f"avg_{field}"] = _safe_mean(df[field])

    if "pf_alg" in task_config:
        metrics["pf_alg"] = int(task_config["pf_alg"])

    if "difficulty_class" in df.columns:
        difficulty_metrics: dict[str, Any] = {}

        for difficulty in ["simple", "medium", "hard"]:
            subset = df[df["difficulty_class"] == difficulty]
            subset_solved = subset["solved"].astype(bool)

            if len(subset) == 0:
                solve_rate = None
                avg_steps_to_solve = None
            else:
                solve_rate = float(subset_solved.mean())
                avg_steps_to_solve = _safe_mean(
                    subset.loc[subset_solved, "steps"]
                )

            metrics[f"count_{difficulty}"] = int(len(subset))
            metrics[f"solve_rate_{difficulty}"] = solve_rate
            metrics[f"avg_steps_to_solve_{difficulty}"] = avg_steps_to_solve
            difficulty_metrics[difficulty] = {
                "count": int(len(subset)),
                "solve_count": int(subset_solved.sum()) if len(subset) else 0,
                "solve_rate": solve_rate,
                "avg_steps": _safe_mean(subset["steps"]) if len(subset) else None,
                "avg_steps_to_solve": avg_steps_to_solve,
                "avg_safety_score": (
                    _safe_mean(subset["safety_score"]) if len(subset) else None
                ),
            }

        metrics["difficulty_metrics"] = difficulty_metrics

    return metrics


def print_summary(
    df: pd.DataFrame,
    failed_results: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"\nEvaluated scenarios: {len(df)}")
    print(f"Failed scenarios:    {len(failed_results)}")

    if failed_results:
        print("\nFailures:")
        for item in failed_results[:20]:
            print(f"  Scenario {item['scenario_id']}: failed")
        if len(failed_results) > 20:
            print(f"  ... {len(failed_results) - 20} more failures")

    print("\nTermination reasons:")
    print(df["termination_reason"].value_counts(dropna=False).to_string())
    print("\nSolved:")
    print(df["solved"].value_counts(dropna=False).to_string())
    print("\nAverage metrics:")
    print(f"  Avg discounted return: {df['discounted_return'].mean():.4f}")
    print(f"  Avg final loading:     {df['final_max_loading_percent'].mean():.4f}%")
    print(f"  Avg overloaded:        {df['final_num_overloaded_branches'].mean():.4f}")
    print(f"  Avg hard overloaded:   {df['final_num_hard_overloaded_branches'].mean():.4f}")
    print(f"  Avg safety score:     {df['safety_score'].mean():.4f}")
    print(f"  Total safety score:   {df['safety_score'].sum():.4f}")

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def compute_safety_score(row: dict[str, Any]) -> float:
    score = 0.0
    reason = row.get("termination_reason")
    solved = bool(row.get("solved", False))
    final_loading = float(row.get("final_max_loading_percent", 999.0))
    overloaded = int(row.get("final_num_overloaded_branches", 99))
    hard = int(row.get("final_num_hard_overloaded_branches", 99))
    discounted_return = float(row.get("discounted_return", 0.0))

    if solved:
        score += 1000.0
    elif reason in {
        "handoff_to_redispatch",
        "handoff_to_redispatch_with_hard_overload",
    }:
        score += 500.0
    elif reason == "max_steps_reached":
        score -= 300.0
    elif reason == "power_flow_failed":
        score -= 1000.0
    else:
        score -= 100.0

    score -= 300.0 * hard
    score -= 50.0 * overloaded

    if final_loading > 100.0:
        score -= 5.0 * (final_loading - 100.0)

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
    solved = df["solved"].astype(bool)
    termination_counts = {
        str(key): int(value)
        for key, value in df["termination_reason"]
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    metrics: dict[str, Any] = {
        "requested_scenarios": int(requested_scenarios),
        "evaluated_scenarios": int(len(df)),
        "failed_scenarios": int(len(failed_results)),
        "solve_count": int(solved.sum()),
        "solve_rate": float(solved.mean()) if len(df) > 0 else 0.0,
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

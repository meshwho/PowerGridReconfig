from __future__ import annotations

import hashlib
import json
import math
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.config import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    physics_config_payload,
    require_physics_config_payload,
    resolve_physics_config,
)
from grid_topology_ai.physics.objective import (
    STOP_POLICIES,
    assess_physical_state,
)
from grid_topology_ai.physics.utility import state_security_penalty, state_utility
from grid_topology_ai.topology_actions import (
    require_topology_action_payload,
    topology_action_payload,
)
from grid_topology_ai.search.root_policy import (
    constrain_policy,
    normalize_policy,
    require_action_in_policy_support,
    select_action_from_policy,
)
from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
    termination_reason_value,
    validate_outcome_invariants,
)


@dataclass(slots=True)
class EvaluationEpisodeTrace:
    actions: list[int] = field(default_factory=list)
    branches: list[int | None] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    total_reward: float = 0.0
    discounted_return: float = 0.0
    constraint_changed_policy_steps: int = 0
    empty_constrained_support_count: int = 0

    @property
    def constraint_exhausted(self) -> bool:
        return self.empty_constrained_support_count > 0


def _canonical_topology_quality_fields(
    *,
    env: Any,
    effective_reason: TerminationReason | None,
    physics_config: PhysicsConfig | None,
) -> dict[str, float]:
    initial_state = getattr(env, "initial_state", None)
    final_state = getattr(env, "current_state", None)

    J0 = (
        float("nan")
        if initial_state is None
        else float(
            state_security_penalty(
                initial_state,
                physics_config=physics_config,
            )
        )
    )

    if effective_reason is TerminationReason.POWER_FLOW_FAILED:
        evidence = getattr(env, "terminal_outcome_evidence", None)
        final_utility = -1.0 if evidence is None else float(evidence.topology_utility)
        return {
            "J0": J0,
            "Jfinal": float("nan"),
            "delta_J": float("nan"),
            "relative_J_improvement": float("nan"),
            "final_topology_utility": final_utility,
        }

    if final_state is None:
        return {
            "J0": J0,
            "Jfinal": float("nan"),
            "delta_J": float("nan"),
            "relative_J_improvement": float("nan"),
            "final_topology_utility": float("nan"),
        }

    Jfinal = float(
        state_security_penalty(
            final_state,
            physics_config=physics_config,
        )
    )
    delta_J = J0 - Jfinal if math.isfinite(J0) else float("nan")
    if math.isfinite(J0) and J0 > 0.0:
        relative_improvement = delta_J / J0
    elif J0 == 0.0 and Jfinal == 0.0:
        relative_improvement = 0.0
    else:
        relative_improvement = float("nan")

    final_utility = float(
        state_utility(
            final_state,
            physics_config=physics_config,
        )
    )
    evidence = getattr(env, "terminal_outcome_evidence", None)
    if evidence is not None:
        evidence_utility = float(evidence.topology_utility)
        if not math.isclose(
            final_utility,
            evidence_utility,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Evaluation final topology utility does not match terminal "
                "outcome evidence."
            )

    return {
        "J0": J0,
        "Jfinal": Jfinal,
        "delta_J": delta_J,
        "relative_J_improvement": float(relative_improvement),
        "final_topology_utility": final_utility,
    }


def build_evaluation_episode_row(
    *,
    scenario_id: int,
    policy_mode: str,
    env: Any,
    trace: EvaluationEpisodeTrace,
    physics_config: PhysicsConfig | None,
) -> dict[str, Any]:
    effective_reason = (
        TerminationReason.CONSTRAINT_EXHAUSTED
        if trace.constraint_exhausted
        else env.termination_reason
    )
    effective_done = bool(env.done or trace.constraint_exhausted)
    effective_solved = bool(env.solved)
    physical = _physical_result_fields(
        env=env,
        effective_done=effective_done,
        effective_solved=effective_solved,
        effective_reason=effective_reason,
    )
    topology_quality = _canonical_topology_quality_fields(
        env=env,
        effective_reason=effective_reason,
        physics_config=physics_config,
    )

    row = {
        "scenario_id": int(scenario_id),
        "policy_mode": str(policy_mode),
        "steps": len(trace.actions),
        "use_continuation_gate": policy_mode == "constrained",
        "actions": str(trace.actions),
        "branches": str(trace.branches),
        "rewards": str([round(value, 4) for value in trace.rewards]),
        "constraint_changed_policy": bool(trace.constraint_changed_policy_steps),
        "constraint_changed_policy_steps": int(trace.constraint_changed_policy_steps),
        "constraint_exhausted": trace.constraint_exhausted,
        "empty_constrained_support_count": int(trace.empty_constrained_support_count),
        "total_reward": float(trace.total_reward),
        "discounted_return": float(trace.discounted_return),
        "done": effective_done,
        "solved": effective_solved,
        "termination_reason": termination_reason_value(effective_reason),
        **physical,
        **topology_quality,
    }
    row["safety_score"] = compute_safety_score(
        row,
        physics_config=physics_config,
    )
    return row


def _physical_result_fields(
    *,
    env: Any,
    effective_done: bool,
    effective_solved: bool,
    effective_reason: TerminationReason | None,
) -> dict[str, Any]:
    final_state = env.current_state
    if final_state is None:
        return {
            "final_max_loading_percent": float("nan"),
            "final_num_overloaded_branches": -1,
            "final_num_hard_overloaded_branches": -1,
            "final_num_outaged_branches": -1,
            "thermal_solved": False,
            "thermal_feasible": False,
            "power_flow_converged": False,
            "all_values_finite": False,
            "topology_connected": False,
            "hard_overload_free": False,
            "voltage_feasible": False,
            "generator_p_feasible": False,
            "generator_q_feasible": False,
            "angle_difference_feasible": False,
            "physically_secure": False,
            "num_generator_p_violations": -1,
            "num_generator_q_violations": -1,
            "num_angle_difference_violations": -1,
            "total_generator_p_violation_mw": float("nan"),
            "total_generator_q_violation_mvar": float("nan"),
            "total_angle_difference_violation_degrees": float("nan"),
            "total_voltage_violation": float("nan"),
            "num_low_voltage_buses": -1,
            "num_high_voltage_buses": -1,
            "total_thermal_overload_mva": float("nan"),
            "safe_handoff": False,
            "unsafe_terminal_state": effective_done,
        }

    assessment = assess_physical_state(final_state.metrics)
    validate_outcome_invariants(
        solved=effective_solved,
        termination_reason=effective_reason,
        physically_secure=assessment.physically_secure,
    )
    safe_handoff = (
        effective_reason is TerminationReason.HANDOFF_TO_REDISPATCH
        and assessment.hard_overload_free
        and not assessment.physically_secure
    )
    return {
        "final_max_loading_percent": float(final_state.metrics["max_loading_percent"]),
        "final_num_overloaded_branches": int(
            final_state.metrics["num_overloaded_branches"]
        ),
        "final_num_hard_overloaded_branches": int(
            final_state.metrics["num_hard_overloaded_branches"]
        ),
        "final_num_outaged_branches": int(final_state.metrics["num_outaged_branches"]),
        "thermal_solved": assessment.thermal_solved,
        "thermal_feasible": assessment.thermal_feasible,
        "power_flow_converged": assessment.power_flow_converged,
        "all_values_finite": assessment.all_values_finite,
        "topology_connected": assessment.topology_connected,
        "hard_overload_free": assessment.hard_overload_free,
        "voltage_feasible": assessment.voltage_feasible,
        "generator_p_feasible": assessment.generator_p_feasible,
        "generator_q_feasible": assessment.generator_q_feasible,
        "angle_difference_feasible": assessment.angle_difference_feasible,
        "physically_secure": assessment.physically_secure,
        "num_generator_p_violations": assessment.num_generator_p_violations,
        "num_generator_q_violations": assessment.num_generator_q_violations,
        "num_angle_difference_violations": (assessment.num_angle_difference_violations),
        "total_generator_p_violation_mw": (assessment.total_generator_p_violation_mw),
        "total_generator_q_violation_mvar": (
            assessment.total_generator_q_violation_mvar
        ),
        "total_angle_difference_violation_degrees": (
            assessment.total_angle_difference_violation_degrees
        ),
        "total_voltage_violation": assessment.total_voltage_violation,
        "num_low_voltage_buses": assessment.num_low_voltage_buses,
        "num_high_voltage_buses": assessment.num_high_voltage_buses,
        "total_thermal_overload_mva": assessment.total_thermal_overload_mva,
        "safe_handoff": safe_handoff,
        "unsafe_terminal_state": bool(
            effective_done and not assessment.physically_secure and not safe_handoff
        ),
    }


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
        config.overload_limit_percent + config.thermal_tolerance_percent
    )
    if final_loading > overload_threshold:
        score -= 5.0 * (final_loading - config.overload_limit_percent)

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
        f"J0={float(row.get('J0', float('nan'))):.2f} | "
        f"Jfinal={float(row.get('Jfinal', float('nan'))):.2f} | "
        f"dJ={float(row.get('delta_J', float('nan'))):+.2f} | "
        f"U={float(row.get('final_topology_utility', float('nan'))):+.3f} | "
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


def _safe_min(
    series: pd.Series,
) -> float | None:
    if len(series) == 0:
        return None

    value = series.min()

    if pd.isna(value):
        return None

    return float(value)


def _numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def build_evaluation_metrics(
    df: pd.DataFrame,
    failed_results: list[dict[str, Any]],
    requested_scenarios: int,
    physics_config: PhysicsConfig | None = None,
) -> dict[str, Any]:
    physics_config = physics_config or DEFAULT_PHYSICS_CONFIG
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

    J0 = _numeric_column(df, "J0")
    Jfinal = _numeric_column(df, "Jfinal")
    delta_J = _numeric_column(df, "delta_J")
    relative_J_improvement = _numeric_column(
        df,
        "relative_J_improvement",
    )
    final_topology_utility = _numeric_column(
        df,
        "final_topology_utility",
    )
    topology_quality_count = int(delta_J.notna().sum())
    topology_improved_count = int((delta_J > 0.0).sum())

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
    unsafe_terminal_state_count = int(df["unsafe_terminal_state"].astype(bool).sum())
    power_flow_failure_count = int(
        (df["termination_reason"] == TerminationReason.POWER_FLOW_FAILED.value).sum()
    )

    metrics: dict[str, Any] = {
        "requested_scenarios": requested_count,
        "evaluated_scenarios": evaluated_scenarios,
        "failed_scenarios": failed_scenarios,
        "solve_count": solve_count,
        "solve_rate": rate(solve_count, evaluated_scenarios),
        "pf_alg": physics_config.pf_alg,
        "evaluation_coverage_rate": rate(
            evaluated_scenarios,
            requested_count,
        ),
        "solve_rate_requested": rate(
            solve_count,
            requested_count,
        ),
        "failed_scenario_rate_requested": rate(
            failed_scenarios,
            requested_count,
        ),
        "hard_overload_free_count": hard_overload_free_count,
        "hard_overload_free_rate": rate(
            hard_overload_free_count,
            evaluated_scenarios,
        ),
        "voltage_feasible_count": voltage_feasible_count,
        "voltage_feasible_rate": rate(
            voltage_feasible_count,
            evaluated_scenarios,
        ),
        "physically_secure_count": physically_secure_count,
        "physically_secure_rate": rate(
            physically_secure_count,
            evaluated_scenarios,
        ),
        "physically_secure_rate_requested": rate(
            physically_secure_count,
            requested_count,
        ),
        "safe_handoff_count": safe_handoff_count,
        "safe_handoff_rate": rate(safe_handoff_count, evaluated_scenarios),
        "unsafe_terminal_state_count": unsafe_terminal_state_count,
        "unsafe_terminal_state_rate": rate(
            unsafe_terminal_state_count, evaluated_scenarios
        ),
        "power_flow_failure_count": power_flow_failure_count,
        "power_flow_failure_rate": rate(
            power_flow_failure_count,
            evaluated_scenarios,
        ),
        "power_flow_failure_rate_requested": rate(
            power_flow_failure_count,
            requested_count,
        ),
        "topology_quality_count": topology_quality_count,
        "topology_improved_count": topology_improved_count,
        "topology_improved_rate": rate(
            topology_improved_count,
            topology_quality_count,
        ),
        "avg_J0": _safe_mean(J0),
        "avg_Jfinal": _safe_mean(Jfinal),
        "avg_delta_J": _safe_mean(delta_J),
        "avg_relative_J_improvement": _safe_mean(relative_J_improvement),
        "avg_final_topology_utility": _safe_mean(final_topology_utility),
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
    }

    for component, count in component_counts.items():
        metrics[f"{component}_count"] = count

        metrics[f"{component}_rate"] = rate(
            count,
            evaluated_scenarios,
        )

        requested_rate_key = f"{component}_rate_requested"

        if requested_rate_key not in metrics:
            metrics[requested_rate_key] = rate(
                count,
                requested_count,
            )

    for measurement in (
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
        metrics[f"avg_{measurement}"] = _safe_mean(df[measurement])

    if "difficulty_class" in df.columns:
        difficulty_metrics: dict[str, Any] = {}

        for difficulty in ["simple", "medium", "hard"]:
            subset = df[df["difficulty_class"] == difficulty]
            subset_solved = subset["solved"].astype(bool)
            subset_delta_J = _numeric_column(subset, "delta_J")

            if len(subset) == 0:
                solve_rate = None
                avg_steps_to_solve = None
            else:
                solve_rate = float(subset_solved.mean())
                avg_steps_to_solve = _safe_mean(subset.loc[subset_solved, "steps"])

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
                "avg_J0": _safe_mean(_numeric_column(subset, "J0")),
                "avg_Jfinal": _safe_mean(_numeric_column(subset, "Jfinal")),
                "avg_delta_J": _safe_mean(subset_delta_J),
                "avg_relative_J_improvement": _safe_mean(
                    _numeric_column(subset, "relative_J_improvement")
                ),
                "avg_final_topology_utility": _safe_mean(
                    _numeric_column(subset, "final_topology_utility")
                ),
                "topology_improved_rate": rate(
                    int((subset_delta_J > 0.0).sum()),
                    int(subset_delta_J.notna().sum()),
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
    print(
        f"  Avg hard overloaded:   {df['final_num_hard_overloaded_branches'].mean():.4f}"
    )
    if "J0" in df.columns:
        print(
            f"  Avg J0:                {pd.to_numeric(df['J0'], errors='coerce').mean():.4f}"
        )
        print(
            f"  Avg Jfinal:            {pd.to_numeric(df['Jfinal'], errors='coerce').mean():.4f}"
        )
        print(
            f"  Avg delta J:           {pd.to_numeric(df['delta_J'], errors='coerce').mean():+.4f}"
        )
        print(
            "  Avg relative J gain:   "
            f"{100.0 * pd.to_numeric(df['relative_J_improvement'], errors='coerce').mean():+.2f}%"
        )
        print(
            "  Avg topology utility:  "
            f"{pd.to_numeric(df['final_topology_utility'], errors='coerce').mean():+.4f}"
        )
    print(f"  Avg safety score:     {df['safety_score'].mean():.4f}")
    print(f"  Total safety score:   {df['safety_score'].sum():.4f}")


GridFMActionSpace = GridFMAdapter = GridFMPowerFlowBackend = None
GridFMReward = MCTSConfig = MCTSPlanner = None
NeuralPolicyValueEvaluator = TopologySwitchingEnv = None
analyze_root_branches = make_do_nothing_action = None
_RUNTIME_DEPENDENCIES_LOADED = False
_WORKER_CONTEXT: dict[str, Any] | None = None
_PHYSICAL_OUTCOME_FIELDS = (
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


def _policies_close(left: dict[int, float], right: dict[int, float]) -> bool:
    return left.keys() == right.keys() and all(
        math.isclose(left[action_id], right[action_id], rel_tol=1e-12, abs_tol=1e-12)
        for action_id in left
    )


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class _RootPolicyDecision:
    raw_policy: dict[int, float]
    policy: dict[int, float]
    action_id: int | None
    branch_id: int | None
    allowed_action_ids: tuple[int, ...]
    constraint_changed_policy: bool
    empty_constrained_support: bool


def _select_root_policy(
    search_result: Any,
    *,
    constrained: bool,
    continuation_analysis: Any | None,
) -> _RootPolicyDecision:
    mode = "constrained" if constrained else "ungated"
    raw_policy = normalize_policy(
        search_result.policy,
        context=f"{mode} MCTS root policy",
    )
    if not constrained:
        policy = raw_policy
        allowed_action_ids = tuple(raw_policy)
    else:
        if continuation_analysis is None:
            raise ValueError("constrained evaluation requires continuation analysis")
        allowed = {
            int(action_id) for action_id in continuation_analysis.allowed_action_ids
        }
        if 0 in raw_policy:
            allowed.add(0)
        allowed_action_ids = tuple(sorted(allowed))
        policy = constrain_policy(
            raw_policy,
            allowed_action_ids,
            context="constrained MCTS root policy",
        )
    changed = not _policies_close(raw_policy, policy)
    if not policy:
        return _RootPolicyDecision(
            raw_policy, {}, None, None, allowed_action_ids, changed, True
        )
    action_id = select_action_from_policy(
        policy,
        temperature=0.0,
        rng=np.random.default_rng(0),
        context=f"{mode} evaluation policy",
    )
    require_action_in_policy_support(
        action_id,
        policy,
        context=f"{mode} evaluation policy",
    )
    if action_id == 0:
        branch_id = None
    else:
        action = search_result.root.actions_by_id.get(action_id)
        if action is None:
            raise RuntimeError(
                f"Action {action_id} is present in the {mode} policy but "
                "missing from root.actions_by_id."
            )
        branch_id = action.branch_id
    return _RootPolicyDecision(
        raw_policy,
        policy,
        int(action_id),
        branch_id,
        allowed_action_ids,
        changed,
        False,
    )


def _evaluation_search_seed(
    *,
    base_seed: int,
    scenario_id: int,
    policy_mode: str,
    step: int,
) -> int:
    payload = (f"{int(base_seed)}:{int(scenario_id)}:{policy_mode}:{int(step)}").encode(
        "utf-8"
    )

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    raw_dir: Path
    transitions_csv: Path
    checkpoint: Path
    config: EvaluationConfig
    physics_config: PhysicsConfig | None = None
    output_csv: Path | None = None
    output_json: Path | None = None
    limit: int | None = None
    scenario_ids: tuple[int, ...] | None = None
    quiet: bool = False
    pf_alg: int | None = None
    disable_cache: bool = False
    stop_policy: str = "no_hard_overloads"
    min_hard_improvement: float = 50.0
    min_soft_improvement: float = 15.0
    min_gate_visits: int = 5
    min_gate_visit_fraction: float = 0.01
    clear_caches_every: int = 100
    use_dc_screening: bool = False
    dc_top_k: int = 30
    dc_candidate_pool: int = 120
    dc_keep_policy_actions: int = 5
    dc_keep_loading_actions: int = 5
    dc_policy_weight: float = 0.0
    dc_failure_penalty: float = 1_000_000_000.0
    dc_max_depth: int = 0

    @property
    def resolved_pf_alg(self) -> int:
        return self.resolved_physics_config.pf_alg

    @property
    def resolved_physics_config(self) -> PhysicsConfig:
        legacy = self.config.pf_alg if self.pf_alg is None else self.pf_alg
        return resolve_physics_config(self.physics_config, legacy)

    def __post_init__(self) -> None:
        if self.limit is not None and int(self.limit) <= 0:
            raise ValueError("limit must be None or > 0")
        if self.scenario_ids is not None:
            if self.limit is not None:
                raise ValueError("scenario_ids and limit cannot be used together.")

            normalized_ids = tuple(
                int(scenario_id) for scenario_id in self.scenario_ids
            )

            if not normalized_ids:
                raise ValueError("scenario_ids must not be empty.")

            if len(normalized_ids) != len(set(normalized_ids)):
                raise ValueError("scenario_ids must not contain duplicates.")

            object.__setattr__(
                self,
                "scenario_ids",
                normalized_ids,
            )
        if self.resolved_pf_alg not in {1, 2, 3, 4}:
            raise ValueError("resolved pf_alg must be one of 1, 2, 3, or 4")
        if self.stop_policy not in STOP_POLICIES:
            raise ValueError("Unsupported stop_policy")
        if (
            min(
                float(self.min_hard_improvement),
                float(self.min_soft_improvement),
            )
            < 0
        ):
            raise ValueError("continuation improvement thresholds must be >= 0")
        if int(self.min_gate_visits) < 0:
            raise ValueError("min_gate_visits must be >= 0")
        if not 0 <= float(self.min_gate_visit_fraction) <= 1:
            raise ValueError("min_gate_visit_fraction must be in [0, 1]")
        if int(self.clear_caches_every) < 0:
            raise ValueError("clear_caches_every must be >= 0")
        if int(self.dc_top_k) <= 0:
            raise ValueError("dc_top_k must be > 0")
        if (
            min(
                int(self.dc_keep_policy_actions),
                int(self.dc_keep_loading_actions),
            )
            < 0
        ):
            raise ValueError("DC keep counts must be >= 0")
        if (
            min(
                float(self.dc_policy_weight),
                float(self.dc_failure_penalty),
            )
            < 0
        ):
            raise ValueError("DC weights and penalties must be >= 0")
        if isinstance(self.dc_max_depth, bool) or not isinstance(
            self.dc_max_depth,
            (int, np.integer),
        ):
            raise ValueError("dc_max_depth must be -1 or a non-negative integer.")

        dc_max_depth = int(self.dc_max_depth)

        if dc_max_depth < -1:
            raise ValueError("dc_max_depth must be -1 or a non-negative integer.")

        object.__setattr__(
            self,
            "dc_max_depth",
            dc_max_depth,
        )


def _ensure_runtime_dependencies() -> None:
    global GridFMActionSpace, GridFMAdapter, GridFMPowerFlowBackend, GridFMReward
    global MCTSConfig, MCTSPlanner, NeuralPolicyValueEvaluator, TopologySwitchingEnv
    global analyze_root_branches, make_do_nothing_action
    global _RUNTIME_DEPENDENCIES_LOADED

    if _RUNTIME_DEPENDENCIES_LOADED:
        return

    from grid_topology_ai.actions import GridFMActionSpace as ActionSpace
    from grid_topology_ai.data import GridFMAdapter as Adapter
    from grid_topology_ai.environment import TopologySwitchingEnv as Env
    from grid_topology_ai.models.neural_evaluator import (
        NeuralPolicyValueEvaluator as Evaluator,
    )
    from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend as Backend
    from grid_topology_ai.physics.utility import GridFMReward as Reward
    from grid_topology_ai.search.continuation_gate import (
        analyze_root_branches as analyze,
        make_do_nothing_action as stop_action,
    )
    from grid_topology_ai.search.mcts import (
        MCTSConfig as SearchConfig,
        MCTSPlanner as Planner,
    )

    GridFMActionSpace, GridFMAdapter = ActionSpace, Adapter
    GridFMPowerFlowBackend, GridFMReward = Backend, Reward
    MCTSConfig, MCTSPlanner = SearchConfig, Planner
    NeuralPolicyValueEvaluator, TopologySwitchingEnv = Evaluator, Env
    analyze_root_branches, make_do_nothing_action = analyze, stop_action
    _RUNTIME_DEPENDENCIES_LOADED = True


def _clear_context_caches(context: dict[str, Any] | None) -> None:
    if context is None:
        return
    for name in ("backend", "action_space", "evaluator"):
        clear = getattr(context.get(name), "clear_cache", None)
        if clear is not None:
            clear()


def _release_worker_context() -> None:
    global _WORKER_CONTEXT
    context, _WORKER_CONTEXT = _WORKER_CONTEXT, None
    _clear_context_caches(context)


def _require_worker_context() -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context is not initialized.")
    return _WORKER_CONTEXT


def init_worker_context(
    raw_dir_str: str,
    checkpoint_path_str: str,
    task_config: dict[str, Any],
) -> None:
    global _WORKER_CONTEXT
    _ensure_runtime_dependencies()

    physics = require_physics_config_payload(
        task_config,
        source="evaluation task",
    )
    (
        topology_action_config,
        action_layout,
    ) = require_topology_action_payload(
        task_config,
        source="evaluation task",
    )

    checkpoint_path = Path(checkpoint_path_str)
    adapter = GridFMAdapter(
        Path(raw_dir_str),
        physics_config=physics,
    )
    cache = not bool(task_config["disable_cache"])
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=physics,
        enable_cache=cache,
    )

    evaluator = NeuralPolicyValueEvaluator(
        checkpoint_path=checkpoint_path,
        device=str(task_config["device"]),
        enable_cache=cache,
        physics_config=physics,
    )

    require_topology_action_payload(
        evaluator.checkpoint,
        source=str(checkpoint_path),
        expected_action_space_config=(topology_action_config),
        expected_action_layout=action_layout,
    )

    action_space = GridFMActionSpace(
        require_connected_after_switch=(
            topology_action_config.require_connected_after_switch
        ),
        min_loading_for_switch_percent=(
            topology_action_config.min_loading_for_switch_percent
        ),
        closeable_branch_ids=(topology_action_config.closeable_branch_ids),
        enable_cache=cache,
    )

    reward_fn = GridFMReward(
        physics_config=physics,
        discount_factor=float(task_config["gamma"]),
    )
    search_config = MCTSConfig(
        num_simulations=int(task_config["simulations"]),
        max_depth=int(task_config["depth"]),
        top_k_actions=int(task_config["top_k"]),
        widening_coefficient=float(task_config["widening_coefficient"]),
        widening_exponent=float(task_config["widening_exponent"]),
        exploration_quota=int(task_config["exploration_quota"]),
        random_seed=int(task_config["random_seed"]),
        gamma=float(task_config["gamma"]),
        c_puct=float(task_config["c_puct"]),
        include_stop_action=True,
        prior_exponent=float(task_config["prior_exponent"]),
        stop_policy=str(task_config["stop_policy"]),
        use_root_dirichlet_noise=False,
        use_dc_screening=bool(task_config["use_dc_screening"]),
        dc_top_k_actions=int(task_config["dc_top_k"]),
        dc_candidate_pool=int(task_config["dc_candidate_pool"]),
        dc_keep_policy_actions=int(task_config["dc_keep_policy_actions"]),
        dc_keep_loading_actions=int(task_config["dc_keep_loading_actions"]),
        dc_policy_weight=float(task_config["dc_policy_weight"]),
        dc_failure_penalty=float(task_config["dc_failure_penalty"]),
        dc_max_depth=int(task_config["dc_max_depth"]),
    )
    _WORKER_CONTEXT = {
        "adapter": adapter,
        "backend": backend,
        "action_space": action_space,
        "reward_fn": reward_fn,
        "evaluator": evaluator,
        "planner": MCTSPlanner(
            config=search_config,
            evaluator=evaluator,
            physics_config=physics,
        ),
        "physics_config": physics,
        "topology_action_config": (topology_action_config),
        "action_layout": action_layout,
        "task_config": task_config,
        "processed_in_worker": 0,
    }


def clear_worker_caches_if_needed() -> None:
    context = _require_worker_context()
    every = int(context["task_config"]["clear_caches_every"])
    if every <= 0:
        return
    context["processed_in_worker"] = int(context["processed_in_worker"]) + 1
    if context["processed_in_worker"] % every == 0:
        _clear_context_caches(context)


def run_episode(
    scenario_id: int,
    adapter: Any,
    backend: Any,
    action_space: Any,
    reward_fn: Any,
    planner: Any,
    max_steps: int,
    gamma: float,
    random_seed: int,
    use_continuation_gate: bool,
    min_hard_improvement: float,
    min_soft_improvement: float,
    min_gate_visits: int,
    min_gate_visit_fraction: float,
    allow_handoff_with_hard_overloads: bool = False,
    physics_config: PhysicsConfig | None = None,
    policy_mode: str | None = None,
) -> dict[str, Any]:
    _ensure_runtime_dependencies()
    mode = str(policy_mode or ("constrained" if use_continuation_gate else "ungated"))
    if mode not in {"ungated", "constrained"}:
        raise ValueError(f"Unsupported evaluation policy mode: {mode}")
    constrained = mode == "constrained"
    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=max_steps,
        allow_handoff_with_hard_overloads=(allow_handoff_with_hard_overloads),
    )
    env.reset(scenario_id)
    trace = EvaluationEpisodeTrace()
    discount = 1.0

    for step in range(max_steps):
        if env.done:
            break

        search_seed = _evaluation_search_seed(
            base_seed=random_seed,
            scenario_id=scenario_id,
            policy_mode=mode,
            step=step,
        )

        planner.reset_rng(search_seed)

        result = planner.search_from_env(env)

        if result.best_action_id is None:
            break

        analysis = None
        if constrained:
            analysis = analyze_root_branches(
                result=result,
                min_hard_improvement=min_hard_improvement,
                min_soft_improvement=min_soft_improvement,
                min_visits=min_gate_visits,
                min_visit_fraction=min_gate_visit_fraction,
                physics_config=physics_config,
            )
        decision = _select_root_policy(
            result,
            constrained=constrained,
            continuation_analysis=analysis,
        )
        trace.constraint_changed_policy_steps += int(decision.constraint_changed_policy)
        if decision.empty_constrained_support:
            trace.empty_constrained_support_count += 1
            break

        assert decision.action_id is not None
        action_id = int(decision.action_id)
        action = (
            make_do_nothing_action()
            if action_id == 0
            else result.root.actions_by_id[action_id]
        )
        step_result = env.step(action)
        reward = float(step_result.reward)
        trace.actions.append(action_id)
        trace.branches.append(decision.branch_id)
        trace.rewards.append(reward)
        trace.total_reward += reward
        trace.discounted_return += discount * reward
        discount *= gamma
        if step_result.done:
            break

    return build_evaluation_episode_row(
        scenario_id=scenario_id,
        policy_mode=mode,
        env=env,
        trace=trace,
        physics_config=physics_config,
    )


def run_episode_from_worker_context(
    scenario_id: int,
    policy_mode: str | None = None,
) -> dict[str, Any]:
    context = _require_worker_context()
    task = context["task_config"]
    mode = str(policy_mode or task["policy_mode"])
    constrained = mode == "constrained"
    try:
        row = run_episode(
            scenario_id=int(scenario_id),
            adapter=context["adapter"],
            backend=context["backend"],
            action_space=context["action_space"],
            reward_fn=context["reward_fn"],
            planner=context["planner"],
            max_steps=int(task["max_steps"]),
            gamma=float(task["gamma"]),
            random_seed=int(task["random_seed"]),
            use_continuation_gate=constrained,
            min_hard_improvement=float(task["min_hard_improvement"]),
            min_soft_improvement=float(task["min_soft_improvement"]),
            min_gate_visits=int(task["min_gate_visits"]),
            min_gate_visit_fraction=float(task["min_gate_visit_fraction"]),
            allow_handoff_with_hard_overloads=bool(
                task["allow_handoff_with_hard_overloads"]
            ),
            physics_config=context["physics_config"],
            policy_mode=mode,
        )
        return {
            "ok": True,
            "scenario_id": int(scenario_id),
            "policy_mode": mode,
            "row": row,
            "traceback": None,
        }
    # Intentional process-worker boundary: serialize one evaluation-mode
    # failure with its traceback without terminating the worker pool.
    except Exception:
        return {
            "ok": False,
            "scenario_id": int(scenario_id),
            "policy_mode": mode,
            "row": None,
            "traceback": traceback.format_exc(),
        }


def run_scenario_batch(scenario_ids: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        results.append(run_episode_from_worker_context(int(scenario_id)))
        clear_worker_caches_if_needed()
    return results


def load_scenario_ids(
    transitions_path: Path,
    limit: int | None,
) -> list[int]:
    transitions = pd.read_csv(transitions_path)
    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions CSV must contain scenario_id column: {transitions_path}"
        )
    scenario_ids = sorted(int(value) for value in transitions["scenario_id"].unique())
    return scenario_ids if limit is None else scenario_ids[: int(limit)]


def chunk_list(values: list[int], batch_size: int) -> list[list[int]]:
    size = max(int(batch_size), 1)
    return [values[index : index + size] for index in range(0, len(values), size)]


def _load_checkpoint_topology_action_payload(
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {checkpoint_path}")

    (
        action_space_config,
        action_layout,
    ) = require_topology_action_payload(
        checkpoint,
        source=str(checkpoint_path),
    )

    return topology_action_payload(
        action_space_config,
        action_layout,
    )


def _make_task_config(request: EvaluationRequest) -> dict[str, Any]:
    if GridFMReward is None:
        _ensure_runtime_dependencies()

    config = request.config
    topology_provenance = _load_checkpoint_topology_action_payload(
        request.checkpoint
    )

    return {
        "simulations": int(config.simulations),
        "depth": int(config.depth),
        "max_steps": int(config.max_steps),
        "top_k": int(config.top_k),
        "widening_coefficient": float(config.widening_coefficient),
        "widening_exponent": float(config.widening_exponent),
        "exploration_quota": int(config.exploration_quota),
        "random_seed": int(config.random_seed),
        "gamma": float(config.gamma),
        "c_puct": float(config.c_puct),
        "prior_exponent": float(config.prior_exponent),
        "stop_policy": str(request.stop_policy),
        "device": str(config.device),
        "pf_alg": request.resolved_physics_config.pf_alg,
        **physics_config_payload(request.resolved_physics_config),
        **topology_provenance,
        "disable_cache": bool(request.disable_cache),
        "use_continuation_gate": bool(config.use_continuation_gate),
        "policy_mode": config.primary_policy_mode,
        "min_hard_improvement": float(request.min_hard_improvement),
        "min_soft_improvement": float(request.min_soft_improvement),
        "min_gate_visits": int(request.min_gate_visits),
        "min_gate_visit_fraction": float(request.min_gate_visit_fraction),
        "allow_handoff_with_hard_overloads": bool(
            config.allow_handoff_with_hard_overloads
        ),
        "clear_caches_every": int(request.clear_caches_every),
        "use_dc_screening": bool(request.use_dc_screening),
        "dc_top_k": int(request.dc_top_k),
        "dc_candidate_pool": int(request.dc_candidate_pool),
        "dc_keep_policy_actions": int(request.dc_keep_policy_actions),
        "dc_keep_loading_actions": int(request.dc_keep_loading_actions),
        "dc_policy_weight": float(request.dc_policy_weight),
        "dc_failure_penalty": float(request.dc_failure_penalty),
        "dc_max_depth": int(request.dc_max_depth),
        "reward_config": GridFMReward(
            physics_config=request.resolved_physics_config,
            discount_factor=config.gamma,
        ).config_dict(),
    }


_RAW_DATA_FILES = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
)


def _record_batch_results(
    batch_results: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    quiet: bool,
) -> None:
    for result in batch_results:
        if result["ok"]:
            rows.append(result["row"])
            if not quiet:
                print_row(result["row"])
        else:
            failed.append(result)
            print(f"Scenario {result['scenario_id']} [{result['policy_mode']}]: failed")
            print(result["traceback"])


def run_sequential(
    scenario_batches: list[list[int]],
    raw_dir: Path,
    checkpoint_path: Path,
    task_config: dict[str, Any],
    quiet: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    init_worker_context(str(raw_dir), str(checkpoint_path), task_config)
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    iterator = (
        tqdm(
            scenario_batches,
            desc="Evaluating batches",
            unit="batch",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else scenario_batches
    )
    for batch in iterator:
        _record_batch_results(
            run_scenario_batch(batch),
            rows=rows,
            failed=failed,
            quiet=quiet,
        )
    return rows, failed


def run_parallel(
    scenario_batches: list[list[int]],
    raw_dir: Path,
    checkpoint_path: Path,
    task_config: dict[str, Any],
    num_workers: int,
    quiet: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=int(num_workers),
        initializer=init_worker_context,
        initargs=(str(raw_dir), str(checkpoint_path), task_config),
    ) as executor:
        futures = [
            executor.submit(run_scenario_batch, batch) for batch in scenario_batches
        ]
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="Evaluating batches",
                unit="batch",
                dynamic_ncols=True,
            )
        for future in iterator:
            _record_batch_results(
                future.result(),
                rows=rows,
                failed=failed,
                quiet=quiet,
            )
    return rows, failed


def _prepare_results_frame(
    rows: list[dict[str, Any]],
    transitions_path: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    defaults = {
        "policy_mode": "ungated",
        "constraint_changed_policy": False,
        "constraint_changed_policy_steps": 0,
        "constraint_exhausted": False,
        "empty_constrained_support_count": 0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    df = df.sort_values(["scenario_id"]).reset_index(drop=True)
    return attach_difficulty_metadata(
        df=df,
        transitions_path=transitions_path,
    )


def _prepare_output_frame(
    successful_rows: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    output = successful_rows.copy()
    output["evaluation_failed"] = False
    output["evaluation_error"] = ""

    if failures:
        failed_rows: list[dict[str, object]] = []

        for failure in failures:
            row: dict[str, object] = {
                "scenario_id": int(failure["scenario_id"]),
                "policy_mode": str(
                    failure.get(
                        "policy_mode",
                        "ungated",
                    )
                ),
                "evaluation_failed": True,
                "evaluation_error": str(failure.get("traceback") or ""),
                "solved": False,
            }

            for field in _PHYSICAL_OUTCOME_FIELDS:
                row[field] = False

            failed_rows.append(row)

        output = pd.concat(
            [
                output,
                pd.DataFrame(failed_rows),
            ],
            ignore_index=True,
            sort=False,
        )

    return output.sort_values(["scenario_id"]).reset_index(drop=True)


def evaluate_checkpoint(request: EvaluationRequest) -> dict[str, Any]:
    sequential = int(request.config.num_workers) <= 1
    try:
        for label, path in (
            ("Raw directory", request.raw_dir),
            ("Transitions CSV", request.transitions_csv),
            ("Checkpoint", request.checkpoint),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        available_scenario_ids = load_scenario_ids(
            request.transitions_csv,
            limit=None,
        )

        if request.scenario_ids is None:
            scenario_ids = (
                available_scenario_ids
                if request.limit is None
                else available_scenario_ids[: int(request.limit)]
            )
        else:
            scenario_ids = list(request.scenario_ids)

            missing_scenario_ids = sorted(
                set(scenario_ids) - set(available_scenario_ids)
            )

            if missing_scenario_ids:
                raise ValueError(
                    "Requested evaluation scenarios are missing from "
                    f"{request.transitions_csv}: "
                    f"{missing_scenario_ids[:20]}"
                )
        batches = chunk_list(
            scenario_ids,
            request.config.batch_size,
        )
        task = _make_task_config(request)

        print("=" * 100)
        print("Evaluating checkpoint")
        print("=" * 100)
        print(f"Policy mode: {task['policy_mode']}")
        print(f"Scenarios: {len(scenario_ids)} | workers: {request.config.num_workers}")

        runner = run_sequential if sequential else run_parallel
        kwargs: dict[str, Any] = {
            "scenario_batches": batches,
            "raw_dir": request.raw_dir,
            "checkpoint_path": request.checkpoint,
            "task_config": task,
            "quiet": bool(request.quiet),
        }
        if not sequential:
            kwargs["num_workers"] = int(request.config.num_workers)
        rows, failures = runner(**kwargs)
        if not rows:
            raise RuntimeError("No scenarios were successfully evaluated.")

        df = _prepare_results_frame(rows, request.transitions_csv)
        metrics = build_evaluation_metrics(
            df=df,
            failed_results=failures,
            requested_scenarios=len(scenario_ids),
            physics_config=request.resolved_physics_config,
        )

        if request.output_csv is not None:
            request.output_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_frame = _prepare_output_frame(
                df,
                failures,
            )
            temporary_csv = request.output_csv.with_name(
                f".{request.output_csv.name}.tmp"
            )
            output_frame.to_csv(
                temporary_csv,
                index=False,
            )
            temporary_csv.replace(request.output_csv)
            print(f"\nSaved evaluation CSV: {request.output_csv}")
        if request.output_json is not None:
            _save_json(metrics, request.output_json)
            print(f"\nSaved evaluation JSON: {request.output_json}")

        print_summary(df, failures)
        if sequential:
            context = _require_worker_context()
            for label, name in (
                ("Power flow", "backend"),
                ("Action space", "action_space"),
                ("Neural evaluator", "evaluator"),
            ):
                print(f"\n{label} cache:")
                print(context[name].cache_info())
        else:
            print("\nParallel mode uses separate per-process caches.")
        print("\nDone.")
        return metrics
    finally:
        if sequential:
            _release_worker_context()

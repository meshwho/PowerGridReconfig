from __future__ import annotations

from dataclasses import replace

import pytest

from grid_topology_ai.config import AcceptanceConfig
from grid_topology_ai.config.acceptance import PRIMARY_ACCEPTANCE_METRIC
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    EVALUATION_METRICS_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.evaluation.paired_results import (
    PAIRED_COMPARISON_VERSION,
    PAIRED_OUTCOME_FIELDS,
)
from grid_topology_ai.physics.objective import physical_objective_contract
from grid_topology_ai.self_play.acceptance import (
    accept_candidate,
    passes_confidence_gates,
    require_metrics_pf_alg,
)

_COMPONENT_FIELDS = (
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


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _metrics(
    *,
    utility: float | None = None,
    requested_scenarios: int = 100,
    failed_scenarios: int = 0,
    physically_secure_count: int = 50,
    power_flow_failure_count: int = 0,
    component_counts: dict[str, int] | None = None,
    pf_alg: object = 3,
    task_config: object | None = None,
    **overrides: object,
) -> dict[str, object]:
    evaluated_scenarios = requested_scenarios - failed_scenarios
    if utility is None:
        utility = _rate(physically_secure_count, requested_scenarios)

    physics_config = DEFAULT_PHYSICS_CONFIG
    if (
        isinstance(pf_alg, int)
        and not isinstance(pf_alg, bool)
        and pf_alg in {1, 2, 3, 4}
    ):
        physics_config = replace(physics_config, pf_alg=pf_alg)

    counts = {
        field: evaluated_scenarios
        for field in _COMPONENT_FIELDS
    }
    counts["power_flow_converged"] = (
        evaluated_scenarios - power_flow_failure_count
    )
    counts["physically_secure"] = physically_secure_count
    if component_counts is not None:
        counts.update(component_counts)

    metrics: dict[str, object] = {
        "evaluation_metrics_contract_version": (
            EVALUATION_METRICS_CONTRACT_VERSION
        ),
        "requested_scenarios": requested_scenarios,
        "evaluated_scenarios": evaluated_scenarios,
        "failed_scenarios": failed_scenarios,
        "solve_count": physically_secure_count,
        "solve_rate": _rate(
            physically_secure_count,
            evaluated_scenarios,
        ),
        "solve_rate_requested": _rate(
            physically_secure_count,
            requested_scenarios,
        ),
        "evaluation_coverage_rate": _rate(
            evaluated_scenarios,
            requested_scenarios,
        ),
        "failed_scenario_rate_requested": _rate(
            failed_scenarios,
            requested_scenarios,
        ),
        "power_flow_failure_count": power_flow_failure_count,
        "power_flow_failure_rate": _rate(
            power_flow_failure_count,
            evaluated_scenarios,
        ),
        "power_flow_failure_rate_requested": _rate(
            power_flow_failure_count,
            requested_scenarios,
        ),
        PRIMARY_ACCEPTANCE_METRIC: float(utility),
        "pf_alg": pf_alg,
        "task_config": (
            {"pf_alg": pf_alg}
            if task_config is None
            else task_config
        ),
        **physics_provenance(physics_config),
        "physical_objective_contract": physical_objective_contract(
            physics_config
        ),
    }

    for field, count in counts.items():
        metrics[f"{field}_count"] = count
        metrics[f"{field}_rate"] = _rate(
            count,
            evaluated_scenarios,
        )
        metrics[f"{field}_rate_requested"] = _rate(
            count,
            requested_scenarios,
        )

    metrics.update(overrides)
    return metrics


def _config(
    *,
    min_improvement: float = 0.0,
    reject_if_failed_scenarios_above: int = 0,
) -> AcceptanceConfig:
    return AcceptanceConfig(
        min_improvement=min_improvement,
        reject_if_failed_scenarios_above=(
            reject_if_failed_scenarios_above
        ),
    )


def _paired_metric(
    *,
    scenario_count: int = 100,
    parent_count: int = 100,
    candidate_count: int = 100,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> dict[str, object]:
    difference = (candidate_count - parent_count) / scenario_count
    improved = max(candidate_count - parent_count, 0)
    regressed = max(parent_count - candidate_count, 0)
    return {
        "parent_count": parent_count,
        "candidate_count": candidate_count,
        "parent_rate": parent_count / scenario_count,
        "candidate_rate": candidate_count / scenario_count,
        "rate_difference": difference,
        "ci_lower": difference if ci_lower is None else ci_lower,
        "ci_upper": difference if ci_upper is None else ci_upper,
        "improved_scenarios": improved,
        "regressed_scenarios": regressed,
        "unchanged_scenarios": scenario_count - improved - regressed,
    }


def _continuous_metric(
    *,
    mean_improvement: float = 0.10,
    ci_lower: float = 0.10,
    ci_upper: float = 0.10,
    scenario_count: int = 100,
) -> dict[str, object]:
    return {
        "valid_pairs": scenario_count,
        "parent_mean": 0.20,
        "candidate_mean": 0.20 + mean_improvement,
        "mean_improvement": mean_improvement,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "improved_scenarios": scenario_count,
        "regressed_scenarios": 0,
        "unchanged_scenarios": 0,
    }


def _comparison(
    *,
    utility_improvement: float = 0.10,
    utility_ci_lower: float = 0.10,
    utility_ci_upper: float = 0.10,
    metric_overrides: dict[str, dict[str, object]] | None = None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 5000,
) -> dict[str, object]:
    metrics = {
        field: _paired_metric()
        for field in PAIRED_OUTCOME_FIELDS
    }
    metrics["physically_secure"] = _paired_metric(
        parent_count=50,
        candidate_count=50,
    )
    if metric_overrides is not None:
        metrics.update(metric_overrides)

    return {
        "paired_comparison_version": PAIRED_COMPARISON_VERSION,
        "policy_mode": "ungated",
        "scenario_count": 100,
        "confidence_level": confidence_level,
        "bootstrap_samples": bootstrap_samples,
        "seed": 7,
        "metrics": metrics,
        "continuous_metrics": {
            "final_topology_utility": _continuous_metric(
                mean_improvement=utility_improvement,
                ci_lower=utility_ci_lower,
                ci_upper=utility_ci_upper,
            ),
        },
    }


def test_primary_acceptance_metric_is_topology_utility() -> None:
    assert PRIMARY_ACCEPTANCE_METRIC == "avg_final_topology_utility"
    assert AcceptanceConfig().metric == PRIMARY_ACCEPTANCE_METRIC


def test_accepts_utility_improvement_without_new_solved_scenarios() -> None:
    assert accept_candidate(
        new_metrics=_metrics(utility=0.30, physically_secure_count=0),
        best_metrics=_metrics(utility=0.10, physically_secure_count=0),
        config=_config(),
    )


def test_rejects_utility_tie() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(utility=0.10),
        best_metrics=_metrics(utility=0.10),
        config=_config(),
    )


def test_rejects_utility_improvement_below_threshold() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(utility=0.14),
        best_metrics=_metrics(utility=0.10),
        config=_config(min_improvement=0.05),
    )


def test_accepts_utility_improvement_at_threshold() -> None:
    assert accept_candidate(
        new_metrics=_metrics(utility=0.15),
        best_metrics=_metrics(utility=0.10),
        config=_config(min_improvement=0.05),
    )


def test_physically_secure_regression_remains_a_safety_gate() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(
            utility=0.60,
            physically_secure_count=49,
        ),
        best_metrics=_metrics(
            utility=0.20,
            physically_secure_count=50,
        ),
        config=_config(),
    )


@pytest.mark.parametrize(
    "component",
    [
        "topology_connected",
        "hard_overload_free",
        "voltage_feasible",
        "generator_p_feasible",
        "generator_q_feasible",
    ],
)
def test_physical_non_inferiority_still_blocks_promotion(
    component: str,
) -> None:
    assert not accept_candidate(
        new_metrics=_metrics(
            utility=0.60,
            component_counts={component: 99},
        ),
        best_metrics=_metrics(
            utility=0.20,
            component_counts={component: 100},
        ),
        config=_config(),
    )


def test_power_flow_failure_regression_blocks_promotion() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(
            utility=0.60,
            power_flow_failure_count=1,
        ),
        best_metrics=_metrics(
            utility=0.20,
            power_flow_failure_count=0,
        ),
        config=_config(),
    )


def test_failed_scenario_limit_still_blocks_promotion() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(
            utility=0.60,
            failed_scenarios=1,
            physically_secure_count=49,
        ),
        best_metrics=_metrics(
            utility=0.20,
            failed_scenarios=1,
            physically_secure_count=49,
        ),
        config=_config(reject_if_failed_scenarios_above=0),
    )


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), float("-inf"), -1.01, 1.01, True, "0.2"],
)
def test_invalid_topology_utility_fails_closed(invalid: object) -> None:
    candidate = _metrics(utility=0.30)
    candidate[PRIMARY_ACCEPTANCE_METRIC] = invalid
    with pytest.raises(ValueError):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(utility=0.10),
            config=_config(),
        )


def test_negative_topology_utility_is_valid() -> None:
    assert accept_candidate(
        new_metrics=_metrics(utility=-0.20),
        best_metrics=_metrics(utility=-0.40),
        config=_config(),
    )


def test_missing_topology_utility_fails_closed() -> None:
    candidate = _metrics(utility=0.30)
    candidate.pop(PRIMARY_ACCEPTANCE_METRIC)
    with pytest.raises(ValueError, match=PRIMARY_ACCEPTANCE_METRIC):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(utility=0.10),
            config=_config(),
        )


def test_different_fixed_evaluation_sizes_fail_closed() -> None:
    with pytest.raises(ValueError, match="same fixed evaluation set size"):
        accept_candidate(
            new_metrics=_metrics(
                utility=0.30,
                requested_scenarios=200,
                physically_secure_count=100,
            ),
            best_metrics=_metrics(utility=0.10),
            config=_config(),
        )


def test_confidence_gate_accepts_confirmed_utility_improvement() -> None:
    assert passes_confidence_gates(
        comparison=_comparison(
            utility_improvement=0.10,
            utility_ci_lower=0.06,
            utility_ci_upper=0.14,
        ),
        config=_config(min_improvement=0.05),
    )


def test_positive_utility_point_estimate_without_positive_ci_is_rejected() -> None:
    assert not passes_confidence_gates(
        comparison=_comparison(
            utility_improvement=0.10,
            utility_ci_lower=-0.01,
            utility_ci_upper=0.20,
        ),
        config=_config(),
    )


def test_confidence_lower_bound_must_strictly_exceed_threshold() -> None:
    assert not passes_confidence_gates(
        comparison=_comparison(
            utility_improvement=0.10,
            utility_ci_lower=0.05,
            utility_ci_upper=0.15,
        ),
        config=_config(min_improvement=0.05),
    )


def test_paired_physically_secure_regression_blocks_promotion() -> None:
    regression = _paired_metric(
        parent_count=50,
        candidate_count=49,
        ci_lower=-0.03,
        ci_upper=0.0,
    )
    assert not passes_confidence_gates(
        comparison=_comparison(
            metric_overrides={"physically_secure": regression},
        ),
        config=_config(),
    )


def test_paired_voltage_regression_blocks_promotion() -> None:
    regression = _paired_metric(
        parent_count=100,
        candidate_count=99,
        ci_lower=-0.02,
        ci_upper=0.0,
    )
    assert not passes_confidence_gates(
        comparison=_comparison(
            metric_overrides={"voltage_feasible": regression},
        ),
        config=_config(),
    )


def test_missing_continuous_primary_metric_fails_closed() -> None:
    comparison = _comparison()
    comparison["continuous_metrics"] = {}
    with pytest.raises(ValueError, match="final_topology_utility"):
        passes_confidence_gates(
            comparison=comparison,
            config=_config(),
        )


def test_comparison_settings_must_match_acceptance_config() -> None:
    with pytest.raises(ValueError, match="bootstrap_samples"):
        passes_confidence_gates(
            comparison=_comparison(bootstrap_samples=1000),
            config=_config(),
        )


def test_acceptance_config_rejects_old_primary_metric() -> None:
    with pytest.raises(ValueError, match="avg_final_topology_utility"):
        AcceptanceConfig(metric="physically_secure_rate_requested")


def test_acceptance_min_improvement_matches_utility_difference_range() -> None:
    assert AcceptanceConfig(min_improvement=1.5).min_improvement == 1.5
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        AcceptanceConfig(min_improvement=2.01)


def test_require_metrics_pf_alg_still_checks_runtime_physics() -> None:
    require_metrics_pf_alg(
        _metrics(utility=0.1, pf_alg=3),
        expected_pf_alg=3,
        source="test",
    )
    with pytest.raises(ValueError, match="expected PF_ALG=3"):
        require_metrics_pf_alg(
            _metrics(utility=0.1, pf_alg=1),
            expected_pf_alg=3,
            source="test",
        )

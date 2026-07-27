from __future__ import annotations

from dataclasses import replace

import pytest
from grid_topology_ai.evaluation.paired_results import (
    PAIRED_COMPARISON_VERSION,
    PAIRED_OUTCOME_FIELDS,
)
from grid_topology_ai.config import AcceptanceConfig
from grid_topology_ai.self_play.acceptance import (
    accept_candidate,
    passes_confidence_gates,
    require_metrics_pf_alg,
)
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    EVALUATION_METRICS_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.physical_objective import physical_objective_contract
from grid_topology_ai.config.acceptance import (
    PRIMARY_ACCEPTANCE_METRIC,
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

    physics_config = DEFAULT_PHYSICS_CONFIG
    if (
        isinstance(pf_alg, int)
        and not isinstance(pf_alg, bool)
        and pf_alg in {1, 2, 3, 4}
    ):
        physics_config = replace(
            physics_config,
            pf_alg=pf_alg,
        )

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

def _paired_metric(
    *,
    scenario_count: int = 100,
    parent_count: int = 100,
    candidate_count: int = 100,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> dict[str, object]:
    difference = (
        candidate_count - parent_count
    ) / scenario_count

    improved = max(
        candidate_count - parent_count,
        0,
    )
    regressed = max(
        parent_count - candidate_count,
        0,
    )

    return {
        "parent_count": parent_count,
        "candidate_count": candidate_count,
        "parent_rate": (
            parent_count / scenario_count
        ),
        "candidate_rate": (
            candidate_count / scenario_count
        ),
        "rate_difference": difference,
        "ci_lower": (
            difference
            if ci_lower is None
            else ci_lower
        ),
        "ci_upper": (
            difference
            if ci_upper is None
            else ci_upper
        ),
        "improved_scenarios": improved,
        "regressed_scenarios": regressed,
        "unchanged_scenarios": (
            scenario_count
            - improved
            - regressed
        ),
    }


def _comparison(
    *,
    primary_ci_lower: float = 0.10,
    primary_ci_upper: float = 0.10,
    metric_overrides: dict[
        str,
        dict[str, object],
    ] | None = None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 5000,
) -> dict[str, object]:
    metrics = {
        field: _paired_metric()
        for field in PAIRED_OUTCOME_FIELDS
    }

    metrics["physically_secure"] = (
        _paired_metric(
            parent_count=50,
            candidate_count=60,
            ci_lower=primary_ci_lower,
            ci_upper=primary_ci_upper,
        )
    )

    if metric_overrides is not None:
        metrics.update(metric_overrides)

    return {
        "paired_comparison_version": (
            PAIRED_COMPARISON_VERSION
        ),
        "policy_mode": "ungated",
        "scenario_count": 100,
        "confidence_level": confidence_level,
        "bootstrap_samples": bootstrap_samples,
        "seed": 7,
        "metrics": metrics,
    }

def test_accepts_confirmed_paired_improvement() -> None:
    assert passes_confidence_gates(
        comparison=_comparison(
            primary_ci_lower=0.06,
            primary_ci_upper=0.14,
        ),
        config=_config(
            min_improvement=0.05
        ),
    )


def test_positive_point_estimate_is_not_enough() -> None:
    assert not passes_confidence_gates(
        comparison=_comparison(
            primary_ci_lower=-0.01,
            primary_ci_upper=0.20,
        ),
        config=_config(),
    )


def test_lower_bound_must_exceed_threshold() -> None:
    assert not passes_confidence_gates(
        comparison=_comparison(
            primary_ci_lower=0.05,
            primary_ci_upper=0.15,
        ),
        config=_config(
            min_improvement=0.05
        ),
    )


def test_voltage_regression_blocks_promotion() -> None:
    voltage = _paired_metric(
        parent_count=100,
        candidate_count=99,
        ci_lower=-0.03,
        ci_upper=0.0,
    )

    assert not passes_confidence_gates(
        comparison=_comparison(
            metric_overrides={
                "voltage_feasible": voltage,
            }
        ),
        config=_config(),
    )


def test_power_flow_regression_blocks_promotion() -> None:
    power_flow = _paired_metric(
        parent_count=100,
        candidate_count=99,
        ci_lower=-0.02,
        ci_upper=0.0,
    )

    assert not passes_confidence_gates(
        comparison=_comparison(
            metric_overrides={
                "power_flow_converged": power_flow,
            }
        ),
        config=_config(),
    )


def test_comparison_settings_must_match_config() -> None:
    with pytest.raises(
        ValueError,
        match="bootstrap_samples",
    ):
        passes_confidence_gates(
            comparison=_comparison(
                bootstrap_samples=1000,
            ),
            config=_config(),
        )


def test_missing_paired_metric_fails_closed() -> None:
    comparison = _comparison()
    metrics = dict(comparison["metrics"])
    metrics.pop("angle_difference_feasible")
    comparison["metrics"] = metrics

    with pytest.raises(
        ValueError,
        match="angle_difference_feasible",
    ):
        passes_confidence_gates(
            comparison=comparison,
            config=_config(),
        )

def test_accepts_confirmed_paired_improvement() -> None:
    assert passes_confidence_gates(
        comparison=_comparison(
            primary_ci_lower=0.06,
            primary_ci_upper=0.14,
        ),
        config=_config(
            min_improvement=0.05
        ),
    )


def test_positive_point_estimate_is_not_enough() -> None:
    assert not passes_confidence_gates(
        comparison=_comparison(
            primary_ci_lower=-0.01,
            primary_ci_upper=0.20,
        ),
        config=_config(),
    )


def test_lower_bound_must_exceed_threshold() -> None:
    assert not passes_confidence_gates(
        comparison=_comparison(
            primary_ci_lower=0.05,
            primary_ci_upper=0.15,
        ),
        config=_config(
            min_improvement=0.05
        ),
    )


def test_voltage_regression_blocks_promotion() -> None:
    voltage = _paired_metric(
        parent_count=100,
        candidate_count=99,
        ci_lower=-0.03,
        ci_upper=0.0,
    )

    assert not passes_confidence_gates(
        comparison=_comparison(
            metric_overrides={
                "voltage_feasible": voltage,
            }
        ),
        config=_config(),
    )


def test_power_flow_regression_blocks_promotion() -> None:
    power_flow = _paired_metric(
        parent_count=100,
        candidate_count=99,
        ci_lower=-0.02,
        ci_upper=0.0,
    )

    assert not passes_confidence_gates(
        comparison=_comparison(
            metric_overrides={
                "power_flow_converged": power_flow,
            }
        ),
        config=_config(),
    )


def test_comparison_settings_must_match_config() -> None:
    with pytest.raises(
        ValueError,
        match="bootstrap_samples",
    ):
        passes_confidence_gates(
            comparison=_comparison(
                bootstrap_samples=1000,
            ),
            config=_config(),
        )


def test_missing_paired_metric_fails_closed() -> None:
    comparison = _comparison()
    metrics = dict(comparison["metrics"])
    metrics.pop("angle_difference_feasible")
    comparison["metrics"] = metrics

    with pytest.raises(
        ValueError,
        match="angle_difference_feasible",
    ):
        passes_confidence_gates(
            comparison=comparison,
            config=_config(),
        )

def _config(
    *,
    min_improvement: float = 0.0,
    reject_if_failed_scenarios_above: int = 0,
) -> AcceptanceConfig:
    return AcceptanceConfig(
        metric=PRIMARY_ACCEPTANCE_METRIC,
        min_improvement=min_improvement,
        reject_if_failed_scenarios_above=(
            reject_if_failed_scenarios_above
        ),
    )


def test_accepts_strict_requested_physical_improvement() -> None:
    assert accept_candidate(
        new_metrics=_metrics(physically_secure_count=60),
        best_metrics=_metrics(physically_secure_count=50),
        config=_config(),
    )


def test_rejects_exact_tie() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(physically_secure_count=50),
        best_metrics=_metrics(physically_secure_count=50),
        config=_config(),
    )


def test_rejects_numerical_noise() -> None:
    requested = 2_000_000_000_000

    assert not accept_candidate(
        new_metrics=_metrics(
            requested_scenarios=requested,
            physically_secure_count=1_000_000_000_001,
        ),
        best_metrics=_metrics(
            requested_scenarios=requested,
            physically_secure_count=1_000_000_000_000,
        ),
        config=_config(),
    )


def test_rejects_improvement_below_required_minimum() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(physically_secure_count=54),
        best_metrics=_metrics(physically_secure_count=50),
        config=_config(min_improvement=0.05),
    )


def test_accepts_improvement_at_required_minimum() -> None:
    assert accept_candidate(
        new_metrics=_metrics(physically_secure_count=55),
        best_metrics=_metrics(physically_secure_count=50),
        config=_config(min_improvement=0.05),
    )


def test_rejects_too_many_failed_scenarios() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(
            failed_scenarios=1,
            physically_secure_count=60,
        ),
        best_metrics=_metrics(
            failed_scenarios=1,
            physically_secure_count=50,
        ),
        config=_config(reject_if_failed_scenarios_above=0),
    )


def test_accepts_failed_scenarios_at_threshold() -> None:
    assert accept_candidate(
        new_metrics=_metrics(
            failed_scenarios=1,
            physically_secure_count=60,
        ),
        best_metrics=_metrics(
            failed_scenarios=1,
            physically_secure_count=50,
        ),
        config=_config(reject_if_failed_scenarios_above=1),
    )


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_rejects_non_finite_primary_rate(
    invalid_value: float,
) -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate[PRIMARY_ACCEPTANCE_METRIC] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01, True, "0.6"])
def test_rejects_invalid_rate_type_or_range(
    invalid_value: object,
) -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate[PRIMARY_ACCEPTANCE_METRIC] = invalid_value

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


def test_missing_primary_metric_fails_closed() -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate.pop(PRIMARY_ACCEPTANCE_METRIC)

    with pytest.raises(ValueError, match=PRIMARY_ACCEPTANCE_METRIC):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


def test_missing_failed_scenarios_fails_closed() -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate.pop("failed_scenarios")

    with pytest.raises(ValueError, match="failed_scenarios"):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "evaluation_coverage_rate",
        "failed_scenario_rate_requested",
        "power_flow_failure_rate_requested",
        "topology_connected_rate_requested",
        "hard_overload_free_rate_requested",
        "voltage_feasible_rate_requested",
        "generator_p_feasible_rate_requested",
        "generator_q_feasible_rate_requested",
    ],
)
def test_missing_mandatory_gate_fails_closed(
    missing_field: str,
) -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


@pytest.mark.parametrize(
    ("field", "component"),
    [
        ("topology_connected_rate_requested", "topology_connected"),
        ("hard_overload_free_rate_requested", "hard_overload_free"),
        ("voltage_feasible_rate_requested", "voltage_feasible"),
        (
            "generator_p_feasible_rate_requested",
            "generator_p_feasible",
        ),
        (
            "generator_q_feasible_rate_requested",
            "generator_q_feasible",
        ),
    ],
)
def test_rejects_physical_non_inferiority_regression(
    field: str,
    component: str,
) -> None:
    candidate = _metrics(
        physically_secure_count=60,
        component_counts={component: 99},
    )
    best = _metrics(
        physically_secure_count=50,
        component_counts={component: 100},
    )

    assert candidate[field] < best[field]
    assert not accept_candidate(
        new_metrics=candidate,
        best_metrics=best,
        config=_config(),
    )


def test_rejects_power_flow_failure_regression() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(
            physically_secure_count=60,
            power_flow_failure_count=1,
        ),
        best_metrics=_metrics(
            physically_secure_count=50,
            power_flow_failure_count=0,
        ),
        config=_config(),
    )


def test_rejects_failed_scenario_and_coverage_regression() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(
            failed_scenarios=2,
            physically_secure_count=60,
        ),
        best_metrics=_metrics(
            failed_scenarios=1,
            physically_secure_count=50,
        ),
        config=_config(reject_if_failed_scenarios_above=2),
    )


def test_rejects_inconsistent_requested_rate() -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate["voltage_feasible_rate_requested"] = 0.25

    with pytest.raises(ValueError, match="inconsistent"):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


def test_rejects_inconsistent_scenario_counts() -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate["evaluated_scenarios"] = 99

    with pytest.raises(ValueError, match="must equal"):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


def test_rejects_fractional_count() -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate["failed_scenarios"] = 0.5

    with pytest.raises(ValueError, match="exact non-negative integer"):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


def test_rejects_solve_and_physical_count_disagreement() -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate["solve_count"] = 59
    candidate["solve_rate"] = 0.59
    candidate["solve_rate_requested"] = 0.59

    with pytest.raises(ValueError, match="solve_count must equal"):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


def test_rejects_different_fixed_evaluation_set_sizes() -> None:
    with pytest.raises(ValueError, match="same fixed evaluation set size"):
        accept_candidate(
            new_metrics=_metrics(
                requested_scenarios=200,
                physically_secure_count=120,
            ),
            best_metrics=_metrics(
                requested_scenarios=100,
                physically_secure_count=50,
            ),
            config=_config(),
        )


def test_does_not_mutate_metric_mappings() -> None:
    new_metrics = _metrics(physically_secure_count=60)
    best_metrics = _metrics(physically_secure_count=50)
    new_before = dict(new_metrics)
    best_before = dict(best_metrics)

    accept_candidate(
        new_metrics=new_metrics,
        best_metrics=best_metrics,
        config=_config(),
    )

    assert new_metrics == new_before
    assert best_metrics == best_before


def test_require_metrics_pf_alg_top_level_accepts() -> None:
    require_metrics_pf_alg(
        _metrics(pf_alg=3),
        expected_pf_alg=3,
        source="test",
    )


def test_require_metrics_pf_alg_legacy_task_config_accepts() -> None:
    metrics = _metrics(pf_alg=3)
    metrics.pop("pf_alg")
    require_metrics_pf_alg(
        metrics,
        expected_pf_alg=3,
        source="test",
    )


def test_require_metrics_pf_alg_missing_rejects() -> None:
    metrics = _metrics()
    metrics.pop("pf_alg")
    metrics["task_config"] = {}

    with pytest.raises(ValueError, match="missing"):
        require_metrics_pf_alg(
            metrics,
            expected_pf_alg=3,
            source="bootstrap",
        )


def test_require_metrics_pf_alg_disagreement_rejects() -> None:
    with pytest.raises(ValueError, match="task_config"):
        require_metrics_pf_alg(
            _metrics(
                pf_alg=3,
                task_config={"pf_alg": 1},
            ),
            expected_pf_alg=3,
            source="best",
        )


def test_require_metrics_pf_alg_mismatch_rejects() -> None:
    with pytest.raises(ValueError, match="expected PF_ALG=3"):
        require_metrics_pf_alg(
            _metrics(pf_alg=1),
            expected_pf_alg=3,
            source="candidate",
        )


def test_require_metrics_pf_alg_invalid_rejects() -> None:
    with pytest.raises(
        ValueError,
        match="PF_ALG conflicts with PhysicsConfig",
    ):
        require_metrics_pf_alg(
            _metrics(pf_alg=9),
            expected_pf_alg=3,
            source="candidate",
        )


def test_require_metrics_pf_alg_accepts_exact_float_and_string() -> None:
    require_metrics_pf_alg(
        _metrics(pf_alg=3.0),
        expected_pf_alg="3",  # type: ignore[arg-type]
        source="test",
    )
    require_metrics_pf_alg(
        _metrics(pf_alg="3"),
        expected_pf_alg=3.0,  # type: ignore[arg-type]
        source="test",
    )


def test_require_metrics_pf_alg_rejects_fractional_top_level() -> None:
    with pytest.raises(
        ValueError,
        match="PF_ALG conflicts with PhysicsConfig",
    ):
        require_metrics_pf_alg(
            _metrics(pf_alg=3.5),
            expected_pf_alg=3,
            source="candidate",
        )


def test_require_metrics_pf_alg_rejects_fractional_task_config() -> None:
    with pytest.raises(ValueError, match="exact integer"):
        require_metrics_pf_alg(
            _metrics(
                pf_alg=3,
                task_config={"pf_alg": 3.5},
            ),
            expected_pf_alg=3,
            source="candidate",
        )


def test_legacy_evaluation_metrics_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="legacy artifacts cannot be upgraded safely",
    ):
        accept_candidate(
            new_metrics={PRIMARY_ACCEPTANCE_METRIC: 0.6},
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )


def test_nested_physical_objective_physics_mismatch_is_rejected() -> None:
    candidate = _metrics(physically_secure_count=60)
    candidate["physical_objective_contract"] = physical_objective_contract(
        replace(
            DEFAULT_PHYSICS_CONFIG,
            overload_limit_percent=115.0,
            hard_overload_limit_percent=135.0,
        )
    )

    with pytest.raises(ValueError, match="PhysicsConfig mismatch"):
        accept_candidate(
            new_metrics=candidate,
            best_metrics=_metrics(physically_secure_count=50),
            config=_config(),
        )

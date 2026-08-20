from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

from grid_topology_ai.config import AcceptanceConfig
from grid_topology_ai.config.acceptance import (
    PRIMARY_ACCEPTANCE_METRIC,
)
from grid_topology_ai.config._validation import coerce_exact_int
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import (
    EVALUATION_METRICS_CONTRACT_VERSION,
    require_exact_contract_version,
    require_physics_provenance,
)
from grid_topology_ai.physics.objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
)
from grid_topology_ai.evaluation.paired_results import (
    PAIRED_COMPARISON_VERSION,
    PAIRED_OUTCOME_FIELDS,
)

_COMPARISON_EPSILON = 1e-12

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

_CONFIDENCE_SAFETY_FIELDS = tuple(PAIRED_OUTCOME_FIELDS)


@dataclass(frozen=True, slots=True)
class _ValidatedAcceptanceMetrics:
    requested_scenarios: int
    evaluated_scenarios: int
    failed_scenarios: int

    avg_final_topology_utility: float
    physically_secure_rate_requested: float

    evaluation_coverage_rate: float
    failed_scenario_rate_requested: float
    power_flow_failure_rate_requested: float

    topology_connected_rate_requested: float
    hard_overload_free_rate_requested: float
    voltage_feasible_rate_requested: float
    generator_p_feasible_rate_requested: float
    generator_q_feasible_rate_requested: float


@dataclass(frozen=True, slots=True)
class _ValidatedPairedMetric:
    rate_difference: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True, slots=True)
class _ValidatedPairedContinuousMetric:
    valid_pairs: int
    mean_improvement: float
    ci_lower: float
    ci_upper: float


def _require_count(
    metrics: Mapping[str, object],
    *,
    name: str,
    source: str,
) -> int:
    if name not in metrics:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            f"required count {name!r} is missing. "
            "Regenerate fixed evaluation metrics."
        )

    raw_value = metrics[name]

    try:
        value = coerce_exact_int(
            name,
            raw_value,
        )
    except ValueError:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            f"{name!r} must be an exact non-negative integer, "
            f"got {raw_value!r}. "
            "Regenerate fixed evaluation metrics."
        ) from None

    if value < 0:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            f"{name!r} must be non-negative, got {value}."
        )

    return value


def _require_rate(
    metrics: Mapping[str, object],
    *,
    name: str,
    source: str,
) -> float:
    if name not in metrics:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            f"required rate {name!r} is missing. "
            "Regenerate fixed evaluation metrics."
        )

    raw_value = metrics[name]

    if isinstance(raw_value, bool) or not isinstance(
        raw_value,
        Real,
    ):
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            f"{name!r} must be a finite numeric rate in [0, 1], "
            f"got {raw_value!r}."
        )

    value = float(raw_value)

    if not math.isfinite(value):
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            f"{name!r} must be finite, got {raw_value!r}."
        )

    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            f"{name!r} must be in [0, 1], got {value}."
        )

    return value


def _require_bounded_number(
    values: Mapping[str, object],
    *,
    name: str,
    source: str,
    lower: float,
    upper: float,
) -> float:
    if name not in values:
        raise ValueError(
            f"Invalid {source}: required field {name!r} is missing."
        )

    raw_value = values[name]
    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
        raise ValueError(
            f"Invalid {source}: {name!r} must be a finite number "
            f"in [{lower}, {upper}], got {raw_value!r}."
        )

    value = float(raw_value)
    if not math.isfinite(value) or value < lower or value > upper:
        raise ValueError(
            f"Invalid {source}: {name!r} must be a finite number "
            f"in [{lower}, {upper}], got {raw_value!r}."
        )

    return value


def _require_consistent_rate(
    *,
    name: str,
    observed: float,
    numerator: int,
    denominator: int,
    source: str,
) -> None:
    expected = (
        0.0
        if denominator == 0
        else float(numerator) / float(denominator)
    )

    if abs(observed - expected) > _COMPARISON_EPSILON:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            f"{name!r} is inconsistent with its counts. "
            f"Expected {expected}, observed {observed}."
        )


def _validate_acceptance_metrics(
    metrics: Mapping[str, object],
    *,
    source: str,
) -> _ValidatedAcceptanceMetrics:
    requested_scenarios = _require_count(
        metrics,
        name="requested_scenarios",
        source=source,
    )

    if requested_scenarios <= 0:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            "requested_scenarios must be greater than zero."
        )

    evaluated_scenarios = _require_count(
        metrics,
        name="evaluated_scenarios",
        source=source,
    )
    failed_scenarios = _require_count(
        metrics,
        name="failed_scenarios",
        source=source,
    )

    if (
        evaluated_scenarios + failed_scenarios
        != requested_scenarios
    ):
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            "evaluated_scenarios + failed_scenarios must equal "
            "requested_scenarios. "
            f"Observed {evaluated_scenarios} + {failed_scenarios} "
            f"!= {requested_scenarios}."
        )

    avg_final_topology_utility = _require_bounded_number(
        metrics,
        name="avg_final_topology_utility",
        source=f"acceptance metrics for {source}",
        lower=-1.0,
        upper=1.0,
    )

    solve_count = _require_count(
        metrics,
        name="solve_count",
        source=source,
    )
    power_flow_failure_count = _require_count(
        metrics,
        name="power_flow_failure_count",
        source=source,
    )

    if solve_count > evaluated_scenarios:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            "solve_count cannot exceed evaluated_scenarios."
        )

    if power_flow_failure_count > evaluated_scenarios:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            "power_flow_failure_count cannot exceed "
            "evaluated_scenarios."
        )

    evaluation_coverage_rate = _require_rate(
        metrics,
        name="evaluation_coverage_rate",
        source=source,
    )
    failed_scenario_rate_requested = _require_rate(
        metrics,
        name="failed_scenario_rate_requested",
        source=source,
    )
    solve_rate = _require_rate(
        metrics,
        name="solve_rate",
        source=source,
    )
    solve_rate_requested = _require_rate(
        metrics,
        name="solve_rate_requested",
        source=source,
    )
    power_flow_failure_rate = _require_rate(
        metrics,
        name="power_flow_failure_rate",
        source=source,
    )
    power_flow_failure_rate_requested = _require_rate(
        metrics,
        name="power_flow_failure_rate_requested",
        source=source,
    )

    _require_consistent_rate(
        name="evaluation_coverage_rate",
        observed=evaluation_coverage_rate,
        numerator=evaluated_scenarios,
        denominator=requested_scenarios,
        source=source,
    )
    _require_consistent_rate(
        name="failed_scenario_rate_requested",
        observed=failed_scenario_rate_requested,
        numerator=failed_scenarios,
        denominator=requested_scenarios,
        source=source,
    )
    _require_consistent_rate(
        name="solve_rate",
        observed=solve_rate,
        numerator=solve_count,
        denominator=evaluated_scenarios,
        source=source,
    )
    _require_consistent_rate(
        name="solve_rate_requested",
        observed=solve_rate_requested,
        numerator=solve_count,
        denominator=requested_scenarios,
        source=source,
    )
    _require_consistent_rate(
        name="power_flow_failure_rate",
        observed=power_flow_failure_rate,
        numerator=power_flow_failure_count,
        denominator=evaluated_scenarios,
        source=source,
    )
    _require_consistent_rate(
        name="power_flow_failure_rate_requested",
        observed=power_flow_failure_rate_requested,
        numerator=power_flow_failure_count,
        denominator=requested_scenarios,
        source=source,
    )

    component_counts: dict[str, int] = {}
    component_rates: dict[str, float] = {}
    component_requested_rates: dict[str, float] = {}

    for field in _COMPONENT_FIELDS:
        count_name = f"{field}_count"
        rate_name = f"{field}_rate"
        requested_rate_name = f"{field}_rate_requested"

        count = _require_count(
            metrics,
            name=count_name,
            source=source,
        )

        if count > evaluated_scenarios:
            raise ValueError(
                f"Invalid acceptance metrics for {source}: "
                f"{count_name!r} cannot exceed "
                "evaluated_scenarios."
            )

        evaluated_rate = _require_rate(
            metrics,
            name=rate_name,
            source=source,
        )
        requested_rate = _require_rate(
            metrics,
            name=requested_rate_name,
            source=source,
        )

        _require_consistent_rate(
            name=rate_name,
            observed=evaluated_rate,
            numerator=count,
            denominator=evaluated_scenarios,
            source=source,
        )
        _require_consistent_rate(
            name=requested_rate_name,
            observed=requested_rate,
            numerator=count,
            denominator=requested_scenarios,
            source=source,
        )

        component_counts[field] = count
        component_rates[field] = evaluated_rate
        component_requested_rates[field] = requested_rate

    physically_secure_count = component_counts[
        "physically_secure"
    ]

    if solve_count != physically_secure_count:
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            "solve_count must equal physically_secure_count. "
            f"Observed {solve_count} != "
            f"{physically_secure_count}."
        )

    physically_secure_rate = component_rates[
        "physically_secure"
    ]
    physically_secure_rate_requested = (
        component_requested_rates["physically_secure"]
    )

    if (
        abs(solve_rate - physically_secure_rate)
        > _COMPARISON_EPSILON
    ):
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            "solve_rate must equal physically_secure_rate."
        )

    if (
        abs(
            solve_rate_requested
            - physically_secure_rate_requested
        )
        > _COMPARISON_EPSILON
    ):
        raise ValueError(
            f"Invalid acceptance metrics for {source}: "
            "solve_rate_requested must equal "
            "physically_secure_rate_requested."
        )

    return _ValidatedAcceptanceMetrics(
        requested_scenarios=requested_scenarios,
        evaluated_scenarios=evaluated_scenarios,
        failed_scenarios=failed_scenarios,
        avg_final_topology_utility=avg_final_topology_utility,
        physically_secure_rate_requested=(
            physically_secure_rate_requested
        ),
        evaluation_coverage_rate=evaluation_coverage_rate,
        failed_scenario_rate_requested=(
            failed_scenario_rate_requested
        ),
        power_flow_failure_rate_requested=(
            power_flow_failure_rate_requested
        ),
        topology_connected_rate_requested=(
            component_requested_rates["topology_connected"]
        ),
        hard_overload_free_rate_requested=(
            component_requested_rates["hard_overload_free"]
        ),
        voltage_feasible_rate_requested=(
            component_requested_rates["voltage_feasible"]
        ),
        generator_p_feasible_rate_requested=(
            component_requested_rates["generator_p_feasible"]
        ),
        generator_q_feasible_rate_requested=(
            component_requested_rates["generator_q_feasible"]
        ),
    )


def _require_difference(
    values: Mapping[str, object],
    *,
    name: str,
    source: str,
) -> float:
    return _require_bounded_number(
        values,
        name=name,
        source=source,
        lower=-1.0,
        upper=1.0,
    )


def _validate_paired_metric(
    metrics: Mapping[str, object],
    *,
    name: str,
    scenario_count: int,
) -> _ValidatedPairedMetric:
    raw_metric = metrics.get(name)

    if not isinstance(raw_metric, Mapping):
        raise ValueError(
            "Invalid paired comparison: metric "
            f"{name!r} is missing or is not a mapping."
        )

    source = f"paired comparison metric {name!r}"

    parent_count = _require_count(
        raw_metric,
        name="parent_count",
        source=source,
    )
    candidate_count = _require_count(
        raw_metric,
        name="candidate_count",
        source=source,
    )

    improved = _require_count(
        raw_metric,
        name="improved_scenarios",
        source=source,
    )
    regressed = _require_count(
        raw_metric,
        name="regressed_scenarios",
        source=source,
    )
    unchanged = _require_count(
        raw_metric,
        name="unchanged_scenarios",
        source=source,
    )

    for count_name, count in (
        ("parent_count", parent_count),
        ("candidate_count", candidate_count),
        ("improved_scenarios", improved),
        ("regressed_scenarios", regressed),
        ("unchanged_scenarios", unchanged),
    ):
        if count > scenario_count:
            raise ValueError(
                f"Invalid {source}: {count_name!r} "
                "cannot exceed scenario_count."
            )

    if improved + regressed + unchanged != scenario_count:
        raise ValueError(
            f"Invalid {source}: improved, regressed "
            "and unchanged scenario counts must sum "
            "to scenario_count."
        )

    parent_rate = _require_rate(
        raw_metric,
        name="parent_rate",
        source=source,
    )
    candidate_rate = _require_rate(
        raw_metric,
        name="candidate_rate",
        source=source,
    )

    _require_consistent_rate(
        name="parent_rate",
        observed=parent_rate,
        numerator=parent_count,
        denominator=scenario_count,
        source=source,
    )
    _require_consistent_rate(
        name="candidate_rate",
        observed=candidate_rate,
        numerator=candidate_count,
        denominator=scenario_count,
        source=source,
    )

    rate_difference = _require_difference(
        raw_metric,
        name="rate_difference",
        source=source,
    )
    ci_lower = _require_difference(
        raw_metric,
        name="ci_lower",
        source=source,
    )
    ci_upper = _require_difference(
        raw_metric,
        name="ci_upper",
        source=source,
    )

    expected_difference = (
        float(candidate_count - parent_count)
        / float(scenario_count)
    )

    if (
        abs(rate_difference - expected_difference)
        > _COMPARISON_EPSILON
    ):
        raise ValueError(
            f"Invalid {source}: rate_difference "
            "is inconsistent with parent and "
            "candidate counts."
        )

    paired_difference = (
        float(improved - regressed)
        / float(scenario_count)
    )

    if (
        abs(rate_difference - paired_difference)
        > _COMPARISON_EPSILON
    ):
        raise ValueError(
            f"Invalid {source}: rate_difference "
            "is inconsistent with improved and "
            "regressed scenario counts."
        )

    if ci_lower > ci_upper:
        raise ValueError(
            f"Invalid {source}: ci_lower cannot "
            "exceed ci_upper."
        )

    if (
        rate_difference
        < ci_lower - _COMPARISON_EPSILON
        or rate_difference
        > ci_upper + _COMPARISON_EPSILON
    ):
        raise ValueError(
            f"Invalid {source}: confidence interval "
            "must contain rate_difference."
        )

    return _ValidatedPairedMetric(
        rate_difference=rate_difference,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def _validate_paired_continuous_metric(
    metrics: Mapping[str, object],
    *,
    name: str,
    scenario_count: int,
) -> _ValidatedPairedContinuousMetric:
    raw_metric = metrics.get(name)
    if not isinstance(raw_metric, Mapping):
        raise ValueError(
            "Invalid paired comparison: continuous metric "
            f"{name!r} is missing or is not a mapping."
        )

    source = f"paired continuous metric {name!r}"
    valid_pairs = _require_count(
        raw_metric,
        name="valid_pairs",
        source=source,
    )
    if valid_pairs != scenario_count:
        raise ValueError(
            f"Invalid {source}: valid_pairs must equal scenario_count "
            f"for acceptance, observed {valid_pairs} != {scenario_count}."
        )

    mean_improvement = _require_bounded_number(
        raw_metric,
        name="mean_improvement",
        source=source,
        lower=-2.0,
        upper=2.0,
    )
    ci_lower = _require_bounded_number(
        raw_metric,
        name="ci_lower",
        source=source,
        lower=-2.0,
        upper=2.0,
    )
    ci_upper = _require_bounded_number(
        raw_metric,
        name="ci_upper",
        source=source,
        lower=-2.0,
        upper=2.0,
    )

    if ci_lower > ci_upper:
        raise ValueError(
            f"Invalid {source}: ci_lower cannot exceed ci_upper."
        )
    if (
        mean_improvement < ci_lower - _COMPARISON_EPSILON
        or mean_improvement > ci_upper + _COMPARISON_EPSILON
    ):
        raise ValueError(
            f"Invalid {source}: confidence interval must contain "
            "mean_improvement."
        )

    return _ValidatedPairedContinuousMetric(
        valid_pairs=valid_pairs,
        mean_improvement=mean_improvement,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def passes_confidence_gates(
    *,
    comparison: Mapping[str, object],
    config: AcceptanceConfig,
) -> bool:
    try:
        version = coerce_exact_int(
            "paired_comparison_version",
            comparison.get(
                "paired_comparison_version"
            ),
        )
    except ValueError:
        raise ValueError(
            "Invalid paired comparison version."
        ) from None

    if version != PAIRED_COMPARISON_VERSION:
        raise ValueError(
            "Paired comparison version mismatch: "
            f"expected {PAIRED_COMPARISON_VERSION}, "
            f"observed {version}."
        )

    scenario_count = coerce_exact_int(
        "paired comparison scenario_count",
        comparison.get("scenario_count"),
    )

    if scenario_count <= 0:
        raise ValueError(
            "Paired comparison scenario_count "
            "must be greater than zero."
        )

    confidence_level = _require_rate(
        comparison,
        name="confidence_level",
        source="paired comparison",
    )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "Paired comparison confidence_level "
            "must be strictly between 0 and 1."
        )

    if (
        abs(
            confidence_level
            - config.confidence_level
        )
        > _COMPARISON_EPSILON
    ):
        raise ValueError(
            "Paired comparison confidence_level "
            "does not match acceptance config: "
            f"comparison={confidence_level}, "
            f"config={config.confidence_level}."
        )

    bootstrap_samples = coerce_exact_int(
        "paired comparison bootstrap_samples",
        comparison.get("bootstrap_samples"),
    )

    if bootstrap_samples <= 0:
        raise ValueError(
            "Paired comparison bootstrap_samples "
            "must be greater than zero."
        )

    if bootstrap_samples != config.bootstrap_samples:
        raise ValueError(
            "Paired comparison bootstrap_samples "
            "does not match acceptance config: "
            f"comparison={bootstrap_samples}, "
            f"config={config.bootstrap_samples}."
        )

    policy_mode = comparison.get("policy_mode")

    if (
        not isinstance(policy_mode, str)
        or not policy_mode.strip()
    ):
        raise ValueError(
            "Paired comparison policy_mode "
            "must be a non-empty string."
        )

    raw_metrics = comparison.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError(
            "Paired comparison is missing metrics."
        )

    validated = {
        field: _validate_paired_metric(
            raw_metrics,
            name=field,
            scenario_count=scenario_count,
        )
        for field in PAIRED_OUTCOME_FIELDS
    }

    raw_continuous = comparison.get("continuous_metrics")
    if not isinstance(raw_continuous, Mapping):
        raise ValueError(
            "Paired comparison is missing continuous_metrics."
        )

    primary = _validate_paired_continuous_metric(
        raw_continuous,
        name="final_topology_utility",
        scenario_count=scenario_count,
    )

    if (
        primary.ci_lower
        <= config.min_improvement
        + _COMPARISON_EPSILON
    ):
        return False

    for field in _CONFIDENCE_SAFETY_FIELDS:
        if validated[field].ci_lower < -_COMPARISON_EPSILON:
            return False

    return True


def _coerce_pf_alg(value: object, *, source: str) -> int:
    try:
        pf_alg = coerce_exact_int("PF_ALG", value)
    except ValueError:
        raise ValueError(
            f"PF_ALG mismatch for {source}: expected exact integer PF_ALG value, "
            f"observed PF_ALG={value!r}. Regenerate fixed evaluation metrics "
            "with the configured PF_ALG before running self-play."
        ) from None

    if pf_alg not in {1, 2, 3, 4}:
        raise ValueError(
            f"PF_ALG mismatch for {source}: observed PF_ALG={pf_alg} is invalid; "
            "expected one of 1, 2, 3, or 4. Regenerate fixed evaluation metrics "
            "with the configured PF_ALG before running self-play."
        )
    return pf_alg


def require_metrics_pf_alg(
    metrics: Mapping[str, object],
    *,
    expected_pf_alg: int,
    source: str,
) -> None:
    require_metrics_semantic_versions(metrics, source=source)
    expected = _coerce_pf_alg(expected_pf_alg, source=source)
    top_level = metrics.get("pf_alg")
    task_pf_alg = None
    task_config = metrics.get("task_config")
    if isinstance(task_config, Mapping):
        task_pf_alg = task_config.get("pf_alg")

    if top_level is None and task_pf_alg is None:
        raise ValueError(
            f"PF_ALG mismatch for {source}: expected PF_ALG={expected}, "
            "observed PF_ALG=missing. Regenerate fixed evaluation metrics "
            "with the configured PF_ALG before running self-play."
        )

    observed = (
        _coerce_pf_alg(top_level, source=source)
        if top_level is not None
        else _coerce_pf_alg(task_pf_alg, source=source)
    )

    if task_pf_alg is not None:
        task_observed = _coerce_pf_alg(task_pf_alg, source=source)
        if task_observed != observed:
            raise ValueError(
                f"PF_ALG mismatch for {source}: expected PF_ALG={expected}, "
                f"observed PF_ALG={observed} but task_config PF_ALG={task_observed}. "
                "Regenerate fixed evaluation metrics with the configured PF_ALG "
                "before running self-play."
            )

    if observed != expected:
        raise ValueError(
            f"PF_ALG mismatch for {source}: expected PF_ALG={expected}, "
            f"observed PF_ALG={observed}. Regenerate fixed evaluation metrics "
            "with the configured PF_ALG before running self-play."
        )


def require_metrics_physics_config(
    metrics: Mapping[str, object],
    *,
    expected_physics_config: PhysicsConfig,
    source: str,
) -> None:
    require_metrics_semantic_versions(metrics, source=source)
    require_physics_provenance(
        metrics,
        source=source,
        expected_physics_config=expected_physics_config,
    )


def require_metrics_semantic_versions(
    metrics: Mapping[str, object],
    *,
    source: str,
) -> None:
    require_exact_contract_version(
        metrics.get("evaluation_metrics_contract_version"),
        expected=EVALUATION_METRICS_CONTRACT_VERSION,
        name="evaluation-metrics contract",
        source=source,
        regeneration_command="python -m scripts.evaluation.evaluate_checkpoint ...",
    )
    metrics_physics_config = require_physics_provenance(
        metrics,
        source=source,
    )
    physical_contract = metrics.get("physical_objective_contract")
    physical_version = (
        physical_contract.get("schema_version")
        if isinstance(physical_contract, Mapping)
        else None
    )
    require_exact_contract_version(
        physical_version,
        expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        name="physical-objective contract",
        source=source,
        regeneration_command="python -m scripts.evaluation.evaluate_checkpoint ...",
    )
    if isinstance(physical_contract, Mapping):
        require_physics_provenance(
            physical_contract,
            source=f"{source} physical-objective contract",
            expected_physics_config=metrics_physics_config,
        )


def accept_candidate(
    *,
    new_metrics: Mapping[str, object],
    best_metrics: Mapping[str, object],
    config: AcceptanceConfig,
) -> bool:
    require_metrics_semantic_versions(
        new_metrics,
        source="candidate metrics",
    )
    require_metrics_semantic_versions(
        best_metrics,
        source="best metrics",
    )

    new_physics = require_physics_provenance(
        new_metrics,
        source="candidate metrics",
    )
    require_physics_provenance(
        best_metrics,
        source="best metrics",
        expected_physics_config=new_physics,
    )

    if config.metric != PRIMARY_ACCEPTANCE_METRIC:
        raise ValueError(
            "Candidate acceptance requires metric "
            f"{PRIMARY_ACCEPTANCE_METRIC!r}, "
            f"got {config.metric!r}."
        )

    candidate = _validate_acceptance_metrics(
        new_metrics,
        source="candidate metrics",
    )
    best = _validate_acceptance_metrics(
        best_metrics,
        source="best metrics",
    )

    if (
        candidate.requested_scenarios
        != best.requested_scenarios
    ):
        raise ValueError(
            "Candidate and best metrics must use the same fixed "
            "evaluation set size. "
            f"Candidate requested_scenarios="
            f"{candidate.requested_scenarios}, "
            f"best requested_scenarios="
            f"{best.requested_scenarios}."
        )

    if (
        candidate.failed_scenarios
        > config.reject_if_failed_scenarios_above
    ):
        return False

    if (
        candidate.failed_scenario_rate_requested
        > best.failed_scenario_rate_requested
        + _COMPARISON_EPSILON
    ):
        return False

    if (
        candidate.power_flow_failure_rate_requested
        > best.power_flow_failure_rate_requested
        + _COMPARISON_EPSILON
    ):
        return False

    higher_is_better_gates = (
        (
            "evaluation_coverage_rate",
            candidate.evaluation_coverage_rate,
            best.evaluation_coverage_rate,
        ),
        (
            "topology_connected_rate_requested",
            candidate.topology_connected_rate_requested,
            best.topology_connected_rate_requested,
        ),
        (
            "hard_overload_free_rate_requested",
            candidate.hard_overload_free_rate_requested,
            best.hard_overload_free_rate_requested,
        ),
        (
            "voltage_feasible_rate_requested",
            candidate.voltage_feasible_rate_requested,
            best.voltage_feasible_rate_requested,
        ),
        (
            "generator_p_feasible_rate_requested",
            candidate.generator_p_feasible_rate_requested,
            best.generator_p_feasible_rate_requested,
        ),
        (
            "generator_q_feasible_rate_requested",
            candidate.generator_q_feasible_rate_requested,
            best.generator_q_feasible_rate_requested,
        ),
        (
            "physically_secure_rate_requested",
            candidate.physically_secure_rate_requested,
            best.physically_secure_rate_requested,
        ),
    )

    for _, candidate_value, best_value in higher_is_better_gates:
        if candidate_value + _COMPARISON_EPSILON < best_value:
            return False

    improvement = (
        candidate.avg_final_topology_utility
        - best.avg_final_topology_utility
    )

    if improvement <= _COMPARISON_EPSILON:
        return False

    if (
        improvement + _COMPARISON_EPSILON
        < config.min_improvement
    ):
        return False

    return True
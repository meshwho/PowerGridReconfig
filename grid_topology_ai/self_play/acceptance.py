from __future__ import annotations

from collections.abc import Mapping

from grid_topology_ai.config import AcceptanceConfig
from grid_topology_ai.config._validation import coerce_exact_int
from grid_topology_ai.contracts import (
    EVALUATION_METRICS_CONTRACT_VERSION,
    require_exact_contract_version,
)
from grid_topology_ai.physical_objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
)

_COMPARISON_EPSILON = 1e-12


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

def accept_candidate(
    *,
    new_metrics: Mapping[str, object],
    best_metrics: Mapping[str, object],
    config: AcceptanceConfig,
) -> bool:
    require_metrics_semantic_versions(new_metrics, source="candidate metrics")
    require_metrics_semantic_versions(best_metrics, source="best metrics")
    metric = config.metric

    if metric not in new_metrics:
        raise KeyError(f"Metric {metric!r} not found in new_metrics.")

    if metric not in best_metrics:
        raise KeyError(f"Metric {metric!r} not found in best_metrics.")

    new_value = float(new_metrics[metric])
    best_value = float(best_metrics[metric])
    improvement = new_value - best_value

    if improvement <= _COMPARISON_EPSILON:
        return False

    if improvement + _COMPARISON_EPSILON < config.min_improvement:
        return False

    if (
        "solve_rate_simple" in new_metrics
        and "solve_rate_simple" in best_metrics
    ):
        if (
            float(new_metrics["solve_rate_simple"])
            < float(best_metrics["solve_rate_simple"])
            - config.max_simple_solve_rate_drop
        ):
            return False

    max_failed = config.reject_if_failed_scenarios_above

    if max_failed is not None and "failed_scenarios" in new_metrics:
        if int(new_metrics["failed_scenarios"]) > int(max_failed):
            return False

    return True

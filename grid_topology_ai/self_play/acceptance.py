from __future__ import annotations

from collections.abc import Mapping

from grid_topology_ai.config import AcceptanceConfig

_COMPARISON_EPSILON = 1e-12


def accept_candidate(
    *,
    new_metrics: Mapping[str, object],
    best_metrics: Mapping[str, object],
    config: AcceptanceConfig,
) -> bool:
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

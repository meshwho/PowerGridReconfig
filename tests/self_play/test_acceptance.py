from __future__ import annotations

from dataclasses import replace

import pytest

from grid_topology_ai.config import AcceptanceConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    EVALUATION_METRICS_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.physical_objective import (
    physical_objective_contract,
)
from grid_topology_ai.self_play.acceptance import accept_candidate


def _metrics(**values: object) -> dict[str, object]:
    raw_pf_alg = values.get("pf_alg")
    physics_config = DEFAULT_PHYSICS_CONFIG
    if isinstance(raw_pf_alg, int) and not isinstance(raw_pf_alg, bool):
        if raw_pf_alg in {1, 2, 3, 4}:
            physics_config = replace(physics_config, pf_alg=raw_pf_alg)

    return {
        "evaluation_metrics_contract_version": EVALUATION_METRICS_CONTRACT_VERSION,
        **physics_provenance(physics_config),
        "physical_objective_contract": physical_objective_contract(
            physics_config
        ),
        **values,
    }


def _config(
    *,
    metric: str = "solve_rate",
    min_improvement: float = 0.0,
    max_simple_solve_rate_drop: float = 0.05,
    reject_if_failed_scenarios_above: int | None = None,
) -> AcceptanceConfig:
    return AcceptanceConfig(
        metric=metric,
        min_improvement=min_improvement,
        max_simple_solve_rate_drop=max_simple_solve_rate_drop,
        reject_if_failed_scenarios_above=reject_if_failed_scenarios_above,
    )


def test_accepts_strict_improvement() -> None:
    assert accept_candidate(
        new_metrics=_metrics(solve_rate=0.6),
        best_metrics=_metrics(solve_rate=0.5),
        config=_config(),
    )


def test_rejects_exact_tie() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(solve_rate=0.5),
        best_metrics=_metrics(solve_rate=0.5),
        config=_config(),
    )


def test_rejects_numerical_noise() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(solve_rate=0.5 + 5e-13),
        best_metrics=_metrics(solve_rate=0.5),
        config=_config(),
    )


def test_rejects_improvement_below_required_minimum() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(solve_rate=0.54),
        best_metrics=_metrics(solve_rate=0.5),
        config=_config(min_improvement=0.05),
    )


def test_accepts_improvement_at_required_minimum() -> None:
    assert accept_candidate(
        new_metrics=_metrics(solve_rate=0.55),
        best_metrics=_metrics(solve_rate=0.5),
        config=_config(min_improvement=0.05),
    )


def test_rejects_excessive_simple_solve_rate_drop() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(
            solve_rate=0.6,
            solve_rate_simple=0.79,
        ),
        best_metrics=_metrics(
            solve_rate=0.5,
            solve_rate_simple=0.9,
        ),
        config=_config(max_simple_solve_rate_drop=0.1),
    )


def test_accepts_allowed_simple_solve_rate_drop() -> None:
    assert accept_candidate(
        new_metrics=_metrics(solve_rate=0.6, solve_rate_simple=0.8),
        best_metrics=_metrics(solve_rate=0.5, solve_rate_simple=0.9),
        config=_config(max_simple_solve_rate_drop=0.1),
    )


def test_simple_guard_is_optional_when_metric_missing() -> None:
    assert accept_candidate(
        new_metrics=_metrics(solve_rate=0.6),
        best_metrics=_metrics(solve_rate=0.5, solve_rate_simple=0.9),
        config=_config(max_simple_solve_rate_drop=0.1),
    )


def test_rejects_too_many_failed_scenarios() -> None:
    assert not accept_candidate(
        new_metrics=_metrics(solve_rate=0.6, failed_scenarios=1),
        best_metrics=_metrics(solve_rate=0.5),
        config=_config(reject_if_failed_scenarios_above=0),
    )


def test_accepts_failed_scenarios_at_threshold() -> None:
    assert accept_candidate(
        new_metrics=_metrics(solve_rate=0.6, failed_scenarios=2),
        best_metrics=_metrics(solve_rate=0.5),
        config=_config(reject_if_failed_scenarios_above=2),
    )


def test_failed_scenario_guard_is_optional_when_metric_missing() -> None:
    assert accept_candidate(
        new_metrics=_metrics(solve_rate=0.6),
        best_metrics=_metrics(solve_rate=0.5),
        config=_config(reject_if_failed_scenarios_above=0),
    )


def test_missing_candidate_primary_metric_raises() -> None:
    with pytest.raises(KeyError, match="solve_rate"):
        accept_candidate(
            new_metrics=_metrics(),
            best_metrics=_metrics(solve_rate=0.5),
            config=_config(),
        )


def test_missing_best_primary_metric_raises() -> None:
    with pytest.raises(KeyError, match="solve_rate"):
        accept_candidate(
            new_metrics=_metrics(solve_rate=0.6),
            best_metrics=_metrics(),
            config=_config(),
        )


def test_does_not_mutate_metric_mappings() -> None:
    new_metrics = _metrics(solve_rate=0.6)
    best_metrics = _metrics(solve_rate=0.5)
    new_before = dict(new_metrics)
    best_before = dict(best_metrics)

    accept_candidate(
        new_metrics=new_metrics,
        best_metrics=best_metrics,
        config=_config(),
    )

    assert new_metrics == new_before
    assert best_metrics == best_before

from grid_topology_ai.self_play.acceptance import require_metrics_pf_alg


def test_require_metrics_pf_alg_top_level_accepts() -> None:
    require_metrics_pf_alg(_metrics(pf_alg=3), expected_pf_alg=3, source="test")


def test_require_metrics_pf_alg_legacy_task_config_accepts() -> None:
    require_metrics_pf_alg(_metrics(task_config={"pf_alg": 3}), expected_pf_alg=3, source="test")


def test_require_metrics_pf_alg_missing_rejects() -> None:
    with pytest.raises(ValueError, match="missing"):
        require_metrics_pf_alg(_metrics(), expected_pf_alg=3, source="bootstrap")


def test_require_metrics_pf_alg_disagreement_rejects() -> None:
    with pytest.raises(ValueError, match="task_config"):
        require_metrics_pf_alg(_metrics(pf_alg=3, task_config={"pf_alg": 1}), expected_pf_alg=3, source="best")


def test_require_metrics_pf_alg_mismatch_rejects() -> None:
    with pytest.raises(ValueError, match="expected PF_ALG=3"):
        require_metrics_pf_alg(_metrics(pf_alg=1), expected_pf_alg=3, source="candidate")


def test_require_metrics_pf_alg_invalid_rejects() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        require_metrics_pf_alg(_metrics(pf_alg=9), expected_pf_alg=3, source="candidate")


def test_require_metrics_pf_alg_accepts_exact_float_and_string() -> None:
    require_metrics_pf_alg(_metrics(pf_alg=3.0), expected_pf_alg="3", source="test")  # type: ignore[arg-type]
    require_metrics_pf_alg(_metrics(pf_alg="3"), expected_pf_alg=3.0, source="test")  # type: ignore[arg-type]


def test_require_metrics_pf_alg_rejects_fractional_top_level() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        require_metrics_pf_alg(_metrics(pf_alg=3.5), expected_pf_alg=3, source="candidate")


def test_require_metrics_pf_alg_rejects_fractional_task_config() -> None:
    with pytest.raises(ValueError, match="exact integer"):
        require_metrics_pf_alg(_metrics(task_config={"pf_alg": 3.5}), expected_pf_alg=3, source="candidate")


def test_legacy_evaluation_metrics_are_rejected() -> None:
    with pytest.raises(ValueError, match="legacy artifacts cannot be upgraded safely"):
        accept_candidate(
            new_metrics={"solve_rate": 0.6},
            best_metrics=_metrics(solve_rate=0.5),
            config=_config(),
        )


def test_nested_physical_objective_physics_mismatch_is_rejected() -> None:
    candidate = _metrics(solve_rate=0.6)
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
            best_metrics=_metrics(solve_rate=0.5),
            config=_config(),
        )

from dataclasses import replace

import pytest

from grid_topology_ai.config import (
    AcceptanceConfig,
    EvaluationConfig,
    GenerationConfig,
    ReplayBufferConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from grid_topology_ai.config.acceptance import PRIMARY_ACCEPTANCE_METRIC
from grid_topology_ai.config.checkpoint_selection import CheckpointSelectionConfig


def test_generation_rejects_zero_simulations() -> None:
    with pytest.raises(ValueError, match="simulations"):
        GenerationConfig(simulations=0)


def test_replay_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="fresh_fraction"):
        ReplayBufferConfig(fresh_fraction=1.5)


def test_training_rejects_invalid_device() -> None:
    with pytest.raises(ValueError, match="device"):
        TrainingConfig(device="tpu")


def test_training_has_no_model_architecture_field() -> None:
    assert "model_type" not in TrainingConfig.__dataclass_fields__


def test_evaluation_rejects_unknown_primary_policy_mode() -> None:
    with pytest.raises(ValueError, match="primary_policy_mode"):
        EvaluationConfig(primary_policy_mode="hybrid")


def test_constrained_primary_mode_requires_continuation_gate() -> None:
    with pytest.raises(ValueError, match="requires"):
        EvaluationConfig(
            primary_policy_mode="constrained",
            use_continuation_gate=False,
        )


def test_checkpoint_arena_requires_ungated_primary_mode() -> None:
    with pytest.raises(ValueError, match="arena.primary_policy_mode"):
        CheckpointSelectionConfig(
            arena=EvaluationConfig(primary_policy_mode="constrained")
        )


def test_self_play_requires_ungated_primary_mode() -> None:
    config = SelfPlayConfig.load("configs/self_play_loop_smoke.yaml")
    constrained = replace(
        config.evaluation,
        primary_policy_mode="constrained",
    )
    with pytest.raises(ValueError, match="evaluation.primary_policy_mode"):
        replace(config, evaluation=constrained)


def test_acceptance_defaults_to_topology_utility_metric() -> None:
    config = AcceptanceConfig()
    assert config.confidence_level == 0.95
    assert config.bootstrap_samples == 5000
    assert config.metric == PRIMARY_ACCEPTANCE_METRIC
    assert config.min_improvement == 0.0
    assert config.reject_if_failed_scenarios_above == 0


@pytest.mark.parametrize("metric", ["", "solve_rate", "physically_secure_rate"])
def test_acceptance_rejects_non_primary_metric(metric: str) -> None:
    with pytest.raises(ValueError, match="metric"):
        AcceptanceConfig(metric=metric)


@pytest.mark.parametrize("valid_value", [0.0, 1.01, 2.0])
def test_acceptance_accepts_topology_utility_improvement_threshold(
    valid_value: float,
) -> None:
    assert AcceptanceConfig(min_improvement=valid_value).min_improvement == valid_value


@pytest.mark.parametrize(
    "invalid_value",
    [-0.01, 2.01, float("nan"), float("inf"), float("-inf"), True],
)
def test_acceptance_rejects_invalid_min_improvement(invalid_value: object) -> None:
    with pytest.raises(ValueError, match="min_improvement"):
        AcceptanceConfig(min_improvement=invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_value",
    [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), True],
)
def test_acceptance_rejects_invalid_confidence_level(invalid_value: object) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        AcceptanceConfig(confidence_level=invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_value",
    [0, -1, 0.5, True, float("nan"), float("inf")],
)
def test_acceptance_rejects_invalid_bootstrap_samples(invalid_value: object) -> None:
    with pytest.raises(ValueError, match="bootstrap_samples"):
        AcceptanceConfig(bootstrap_samples=invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_value",
    [-1, 0.5, True, float("nan"), float("inf")],
)
def test_acceptance_rejects_invalid_failed_scenario_threshold(
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match="reject_if_failed_scenarios_above"):
        AcceptanceConfig(
            reject_if_failed_scenarios_above=invalid_value  # type: ignore[arg-type]
        )


def test_acceptance_from_mapping_rejects_legacy_simple_gate() -> None:
    with pytest.raises(ValueError, match="was removed"):
        AcceptanceConfig.from_mapping({"max_simple_solve_rate_drop": 0.05})


def test_acceptance_from_mapping_uses_strict_defaults() -> None:
    config = AcceptanceConfig.from_mapping({})
    assert config.confidence_level == 0.95
    assert config.bootstrap_samples == 5000
    assert config.metric == PRIMARY_ACCEPTANCE_METRIC
    assert config.min_improvement == 0.0
    assert config.reject_if_failed_scenarios_above == 0

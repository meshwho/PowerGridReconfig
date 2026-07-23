import pytest

from grid_topology_ai.config import (
    AcceptanceConfig,
    GenerationConfig,
    ReplayBufferConfig,
    TrainingConfig,
)
from grid_topology_ai.config.acceptance import PRIMARY_ACCEPTANCE_METRIC


def test_generation_rejects_zero_simulations() -> None:
    with pytest.raises(ValueError, match="simulations"):
        GenerationConfig(simulations=0)


def test_replay_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="fresh_fraction"):
        ReplayBufferConfig(fresh_fraction=1.5)


def test_training_rejects_invalid_device() -> None:
    with pytest.raises(ValueError, match="device"):
        TrainingConfig(device="tpu")


def test_training_config_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="training.model_type"):
        TrainingConfig(model_type="graph-v3")


def test_training_config_from_mapping_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="training.model_type"):
        TrainingConfig.from_mapping({"model_type": "graph-v3"})


def test_training_config_accepts_supported_model_types() -> None:
    assert TrainingConfig(model_type="graph_v1").model_type == "graph_v1"
    assert TrainingConfig(model_type="graph_v2").model_type == "graph_v2"


def test_acceptance_defaults_to_requested_physical_metric() -> None:
    config = AcceptanceConfig()

    assert config.metric == PRIMARY_ACCEPTANCE_METRIC
    assert config.min_improvement == 0.0
    assert config.reject_if_failed_scenarios_above == 0


@pytest.mark.parametrize("metric", ["", "solve_rate", "physically_secure_rate"])
def test_acceptance_rejects_non_primary_metric(metric: str) -> None:
    with pytest.raises(ValueError, match="metric"):
        AcceptanceConfig(metric=metric)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    ],
)
def test_acceptance_rejects_invalid_min_improvement(
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match="min_improvement"):
        AcceptanceConfig(
            min_improvement=invalid_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_value",
    [-1, 0.5, True, float("nan"), float("inf")],
)
def test_acceptance_rejects_invalid_failed_scenario_threshold(
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match="reject_if_failed_scenarios_above"):
        AcceptanceConfig(
            reject_if_failed_scenarios_above=invalid_value,  # type: ignore[arg-type]
        )


def test_acceptance_from_mapping_rejects_legacy_simple_gate() -> None:
    with pytest.raises(ValueError, match="was removed"):
        AcceptanceConfig.from_mapping(
            {
                "max_simple_solve_rate_drop": 0.05,
            }
        )


def test_acceptance_from_mapping_uses_strict_defaults() -> None:
    config = AcceptanceConfig.from_mapping({})

    assert config.metric == PRIMARY_ACCEPTANCE_METRIC
    assert config.min_improvement == 0.0
    assert config.reject_if_failed_scenarios_above == 0

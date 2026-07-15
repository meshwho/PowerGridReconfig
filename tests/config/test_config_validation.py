import pytest

from grid_topology_ai.config import (
    AcceptanceConfig,
    GenerationConfig,
    ReplayBufferConfig,
    TrainingConfig,
)


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

def test_acceptance_rejects_empty_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        AcceptanceConfig(metric="")
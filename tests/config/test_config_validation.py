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


def test_acceptance_rejects_empty_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        AcceptanceConfig(metric="")
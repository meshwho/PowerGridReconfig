from pathlib import Path

import pytest

from grid_topology_ai.config import SelfPlayConfig


@pytest.mark.parametrize(
    "path",
    [
        Path("configs/self_play_loop.yaml"),
        Path("configs/self_play_loop_smoke.yaml"),
        Path("configs/self_play_loop_pilot.yaml"),
    ],
)
def test_repository_config_parses(path: Path) -> None:
    config = SelfPlayConfig.load(path)

    assert config.run_name
    assert config.n_iterations > 0
    assert config.n_scenarios_per_iteration > 0


def test_pilot_config_preserves_current_values() -> None:
    config = SelfPlayConfig.load(
        "configs/self_play_loop_pilot.yaml"
    )

    assert config.run_name == "self_play_pilot"
    assert config.n_iterations == 3
    assert config.training.epochs == 3
    assert config.training.examples_per_iteration == 128
    assert config.generation.simulations == 25
    assert config.generation.pf_alg == 3
    assert config.evaluation.simulations == 50
    assert config.replay_buffer.fresh_fraction == 0.70
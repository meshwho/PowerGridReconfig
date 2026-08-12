from pathlib import Path

import pytest
import yaml

from grid_topology_ai.config import SelfPlayConfig


@pytest.mark.parametrize(
    "filename",
    [
        "self_play_loop.yaml",
        "self_play_loop_pilot.yaml",
        "self_play_loop_smoke.yaml",
    ],
)
def test_self_play_configs_use_canonical_newton_raphson(
    filename: str,
) -> None:
    path = Path("configs") / filename
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = SelfPlayConfig.from_mapping(raw)

    assert config.physics.pf_alg == 1
    assert config.generation.pf_alg == 1
    assert config.evaluation.pf_alg == 1
    assert config.checkpoint_selection.arena.pf_alg == 1

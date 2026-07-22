from pathlib import Path

import pytest

from grid_topology_ai.config import GenerationConfig


@pytest.mark.parametrize(
    "key",
    [
        "terminal_unsolved_penalty",
        "terminal_handoff_penalty",
        "terminal_failure_penalty",
        "terminal_penalty_weight",
    ],
)
def test_generation_config_neutralizes_legacy_terminal_penalties(key: str) -> None:
    with pytest.warns(DeprecationWarning):
        config = GenerationConfig.from_mapping({key: 1.0})
    assert getattr(config, key) == 0.0


def test_repository_generation_configs_have_no_terminal_penalties() -> None:
    forbidden = (
        "terminal_unsolved_penalty",
        "terminal_handoff_penalty",
        "terminal_failure_penalty",
        "terminal_penalty_weight",
    )
    for path in (
        Path("configs/self_play_loop.yaml"),
        Path("configs/self_play_loop_pilot.yaml"),
        Path("configs/self_play_loop_smoke.yaml"),
    ):
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden)

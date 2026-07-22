from dataclasses import fields
from pathlib import Path
import warnings

import pytest

from grid_topology_ai.config import GenerationConfig


LEGACY_TERMINAL_PENALTY_FIELDS = (
    "terminal_unsolved_penalty",
    "terminal_handoff_penalty",
    "terminal_failure_penalty",
    "terminal_penalty_weight",
)


def test_generation_config_has_no_legacy_terminal_penalty_fields() -> None:
    field_names = {field.name for field in fields(GenerationConfig)}
    assert set(LEGACY_TERMINAL_PENALTY_FIELDS).isdisjoint(field_names)


@pytest.mark.parametrize("key", LEGACY_TERMINAL_PENALTY_FIELDS)
def test_generation_config_rejects_legacy_terminal_penalties(key: str) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Unsupported legacy generation terminal penalty fields.*"
            "Terminal penalties were removed"
        ),
    ):
        GenerationConfig.from_mapping({key: 1.0})


def test_generation_config_emits_no_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GenerationConfig()

    assert not any(
        issubclass(item.category, DeprecationWarning)
        for item in caught
    )


def test_repository_generation_configs_have_no_terminal_penalties() -> None:
    for path in (
        Path("configs/self_play_loop.yaml"),
        Path("configs/self_play_loop_pilot.yaml"),
        Path("configs/self_play_loop_smoke.yaml"),
    ):
        text = path.read_text(encoding="utf-8")
        assert all(
            token not in text
            for token in LEGACY_TERMINAL_PENALTY_FIELDS
        )

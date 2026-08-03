from __future__ import annotations

from typing import Any

from grid_topology_ai.self_play import _replay_core as _core
from grid_topology_ai.self_play._replay_core import (
    RollingReplayBuffer as _CoreRollingReplayBuffer,
)
from grid_topology_ai.self_play.example_validation import (
    load_and_validate_examples_csv,
)
from grid_topology_ai.self_play.replay_error_sampling import (
    ReplayPredictionErrorMixin,
)


__all__ = (
    "RollingReplayBuffer",
    "load_and_validate_examples_csv",
)


class RollingReplayBuffer(
    ReplayPredictionErrorMixin,
    _CoreRollingReplayBuffer,
):
    """Persistent replay buffer with episode-balanced priority sampling."""


def __getattr__(name: str) -> Any:
    """Keep private replay helpers available to validation tests."""

    return getattr(_core, name)

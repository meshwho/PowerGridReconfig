from __future__ import annotations

from typing import Any

from grid_topology_ai.self_play import _replay_core as _core
from grid_topology_ai.self_play._replay_core import (
    RollingReplayBuffer as _CoreRollingReplayBuffer,
)
from grid_topology_ai.self_play.replay_sampling import EpisodeSamplingMixin


class RollingReplayBuffer(EpisodeSamplingMixin, _CoreRollingReplayBuffer):
    """Persistent replay buffer with episode-balanced batch sampling."""


def __getattr__(name: str) -> Any:
    """Keep private replay helpers available to validation tests."""

    return getattr(_core, name)

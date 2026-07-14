from __future__ import annotations

from dataclasses import dataclass

from grid_topology_ai.config._mapping import ConfigMapping
from grid_topology_ai.config._validation import (
    require_fraction,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class ReplayBufferConfig:
    max_size: int = 50_000
    min_size_to_train: int = 1_000
    fresh_fraction: float = 0.70
    random_seed: int = 42

    def __post_init__(self) -> None:
        require_positive("replay_buffer.max_size", self.max_size)
        require_positive(
            "replay_buffer.min_size_to_train",
            self.min_size_to_train,
        )
        require_fraction(
            "replay_buffer.fresh_fraction",
            self.fresh_fraction,
        )

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "ReplayBufferConfig":
        return cls(
            max_size=int(data.get("max_size", 50_000)),
            min_size_to_train=int(
                data.get("min_size_to_train", 1_000)
            ),
            fresh_fraction=float(
                data.get("fresh_fraction", 0.70)
            ),
            random_seed=int(data.get("random_seed", 42)),
        )
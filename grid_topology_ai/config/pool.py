from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grid_topology_ai.config._mapping import (
    ConfigMapping,
    require_value,
)


@dataclass(frozen=True, slots=True)
class PoolConfig:
    transitions_csv: Path
    raw_dir: Path
    metadata_path: Path

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "PoolConfig":
        return cls(
            transitions_csv=Path(
                require_value(data, "transitions_csv")
            ),
            raw_dir=Path(
                require_value(data, "raw_dir")
            ),
            metadata_path=Path(
                require_value(data, "metadata_path")
            ),
        )
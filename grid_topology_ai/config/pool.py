from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from grid_topology_ai.config._mapping import (
    ConfigMapping,
    get_section,
    require_value,
)
from grid_topology_ai.config._validation import (
    coerce_exact_int,
    require_fraction,
    require_non_negative,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class CurriculumSamplingConfig:
    never_solved_min_fraction: float = 0.20
    hard_min_fraction: float = 0.25
    simple_max_fraction: float = 0.35
    frontier_max_fraction: float = 0.50

    frontier_solve_rate_min: float = 0.25
    frontier_solve_rate_max: float = 0.75

    learning_progress_weight: float = 1.00
    uncertainty_weight: float = 0.75
    staleness_weight: float = 0.50
    frontier_weight: float = 0.35

    stale_after_iterations: int = 3
    priority_floor: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            (
                "never_solved_min_fraction",
                self.never_solved_min_fraction,
            ),
            ("hard_min_fraction", self.hard_min_fraction),
            ("simple_max_fraction", self.simple_max_fraction),
            ("frontier_max_fraction", self.frontier_max_fraction),
            (
                "frontier_solve_rate_min",
                self.frontier_solve_rate_min,
            ),
            (
                "frontier_solve_rate_max",
                self.frontier_solve_rate_max,
            ),
        ):
            require_fraction(name, value)

        if (
            self.frontier_solve_rate_min
            >= self.frontier_solve_rate_max
        ):
            raise ValueError(
                "frontier_solve_rate_min must be smaller than "
                "frontier_solve_rate_max."
            )

        for name, value in (
            (
                "learning_progress_weight",
                self.learning_progress_weight,
            ),
            ("uncertainty_weight", self.uncertainty_weight),
            ("staleness_weight", self.staleness_weight),
            ("frontier_weight", self.frontier_weight),
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite, got {value}."
                )
            require_non_negative(name, value)

        require_positive(
            "stale_after_iterations",
            self.stale_after_iterations,
        )

        if not math.isfinite(self.priority_floor):
            raise ValueError(
                "priority_floor must be finite, "
                f"got {self.priority_floor}."
            )
        require_positive("priority_floor", self.priority_floor)

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "CurriculumSamplingConfig":
        return cls(
            never_solved_min_fraction=float(
                data.get(
                    "never_solved_min_fraction",
                    0.20,
                )
            ),
            hard_min_fraction=float(
                data.get("hard_min_fraction", 0.25)
            ),
            simple_max_fraction=float(
                data.get("simple_max_fraction", 0.35)
            ),
            frontier_max_fraction=float(
                data.get("frontier_max_fraction", 0.50)
            ),
            frontier_solve_rate_min=float(
                data.get(
                    "frontier_solve_rate_min",
                    0.25,
                )
            ),
            frontier_solve_rate_max=float(
                data.get(
                    "frontier_solve_rate_max",
                    0.75,
                )
            ),
            learning_progress_weight=float(
                data.get(
                    "learning_progress_weight",
                    1.00,
                )
            ),
            uncertainty_weight=float(
                data.get("uncertainty_weight", 0.75)
            ),
            staleness_weight=float(
                data.get("staleness_weight", 0.50)
            ),
            frontier_weight=float(
                data.get("frontier_weight", 0.35)
            ),
            stale_after_iterations=coerce_exact_int(
                "stale_after_iterations",
                data.get("stale_after_iterations", 3),
            ),
            priority_floor=float(
                data.get("priority_floor", 0.05)
            ),
        )


@dataclass(frozen=True, slots=True)
class PoolConfig:
    transitions_csv: Path
    raw_dir: Path
    metadata_path: Path
    curriculum: CurriculumSamplingConfig = field(
        default_factory=CurriculumSamplingConfig
    )

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
            curriculum=CurriculumSamplingConfig.from_mapping(
                get_section(
                    data,
                    "curriculum",
                    required=False,
                )
            ),
        )

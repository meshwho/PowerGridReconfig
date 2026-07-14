from __future__ import annotations

from dataclasses import dataclass

from grid_topology_ai.config._mapping import ConfigMapping
from grid_topology_ai.config._validation import (
    require_non_negative,
)


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    metric: str = "solve_rate"
    min_improvement: float = 0.0
    max_simple_solve_rate_drop: float = 0.05
    reject_if_failed_scenarios_above: int | None = None

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError(
                "acceptance.metric must not be empty."
            )

        require_non_negative(
            "acceptance.min_improvement",
            self.min_improvement,
        )
        require_non_negative(
            "acceptance.max_simple_solve_rate_drop",
            self.max_simple_solve_rate_drop,
        )

        if self.reject_if_failed_scenarios_above is not None:
            require_non_negative(
                "acceptance.reject_if_failed_scenarios_above",
                self.reject_if_failed_scenarios_above,
            )

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "AcceptanceConfig":
        max_failed = data.get(
            "reject_if_failed_scenarios_above"
        )

        return cls(
            metric=str(data.get("metric", "solve_rate")),
            min_improvement=float(
                data.get("min_improvement", 0.0)
            ),
            max_simple_solve_rate_drop=float(
                data.get(
                    "max_simple_solve_rate_drop",
                    0.05,
                )
            ),
            reject_if_failed_scenarios_above=(
                None
                if max_failed is None
                else int(max_failed)
            ),
        )
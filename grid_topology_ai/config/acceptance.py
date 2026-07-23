from __future__ import annotations

import math
from dataclasses import dataclass

from grid_topology_ai.config._mapping import ConfigMapping
from grid_topology_ai.config._validation import (
    coerce_exact_int,
    require_non_negative,
)


PRIMARY_ACCEPTANCE_METRIC = "physically_secure_rate_requested"


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    metric: str = PRIMARY_ACCEPTANCE_METRIC
    min_improvement: float = 0.0
    reject_if_failed_scenarios_above: int = 0

    def __post_init__(self) -> None:
        metric = str(self.metric).strip()

        if metric != PRIMARY_ACCEPTANCE_METRIC:
            raise ValueError(
                "acceptance.metric must be exactly "
                f"{PRIMARY_ACCEPTANCE_METRIC!r}, got {metric!r}."
            )

        object.__setattr__(
            self,
            "metric",
            metric,
        )

        if isinstance(self.min_improvement, bool):
            raise ValueError(
                "acceptance.min_improvement must be a finite number "
                "in [0, 1], not a boolean."
            )

        try:
            min_improvement = float(self.min_improvement)
        except (TypeError, ValueError):
            raise ValueError(
                "acceptance.min_improvement must be a finite number "
                "in [0, 1]."
            ) from None

        if (
            not math.isfinite(min_improvement)
            or min_improvement < 0.0
            or min_improvement > 1.0
        ):
            raise ValueError(
                "acceptance.min_improvement must be a finite number "
                f"in [0, 1], got {self.min_improvement!r}."
            )

        object.__setattr__(
            self,
            "min_improvement",
            min_improvement,
        )

        max_failed = coerce_exact_int(
            "acceptance.reject_if_failed_scenarios_above",
            self.reject_if_failed_scenarios_above,
        )

        require_non_negative(
            "acceptance.reject_if_failed_scenarios_above",
            max_failed,
        )

        object.__setattr__(
            self,
            "reject_if_failed_scenarios_above",
            max_failed,
        )

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "AcceptanceConfig":
        if "max_simple_solve_rate_drop" in data:
            raise ValueError(
                "acceptance.max_simple_solve_rate_drop was removed. "
                "Candidate acceptance now uses mandatory physical "
                "non-inferiority gates."
            )

        return cls(
            metric=str(
                data.get(
                    "metric",
                    PRIMARY_ACCEPTANCE_METRIC,
                )
            ),
            min_improvement=float(
                data.get(
                    "min_improvement",
                    0.0,
                )
            ),
            reject_if_failed_scenarios_above=coerce_exact_int(
                "acceptance.reject_if_failed_scenarios_above",
                data.get(
                    "reject_if_failed_scenarios_above",
                    0,
                ),
            ),
        )
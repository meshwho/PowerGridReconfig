from __future__ import annotations

import math
from dataclasses import dataclass

from grid_topology_ai.config._mapping import ConfigMapping
from grid_topology_ai.config._validation import (
    coerce_exact_int,
    require_non_negative,
    require_positive,
)


PRIMARY_ACCEPTANCE_METRIC = "avg_final_topology_utility"


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    metric: str = PRIMARY_ACCEPTANCE_METRIC
    min_improvement: float = 0.0
    reject_if_failed_scenarios_above: int = 0
    confidence_level: float = 0.95
    bootstrap_samples: int = 5000

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
        if isinstance(self.confidence_level, bool):
            raise ValueError(
                "acceptance.confidence_level must be a finite "
                "number strictly between 0 and 1."
            )

        try:
            confidence_level = float(
                self.confidence_level
            )
        except (TypeError, ValueError):
            raise ValueError(
                "acceptance.confidence_level must be a finite "
                "number strictly between 0 and 1."
            ) from None

        if (
                not math.isfinite(confidence_level)
                or confidence_level <= 0.0
                or confidence_level >= 1.0
        ):
            raise ValueError(
                "acceptance.confidence_level must be a finite "
                "number strictly between 0 and 1, "
                f"got {self.confidence_level!r}."
            )

        object.__setattr__(
            self,
            "confidence_level",
            confidence_level,
        )

        bootstrap_samples = coerce_exact_int(
            "acceptance.bootstrap_samples",
            self.bootstrap_samples,
        )

        require_positive(
            "acceptance.bootstrap_samples",
            bootstrap_samples,
        )

        object.__setattr__(
            self,
            "bootstrap_samples",
            bootstrap_samples,
        )
        if isinstance(self.min_improvement, bool):
            raise ValueError(
                "acceptance.min_improvement must be a finite number "
                "in [0, 2], not a boolean."
            )

        try:
            min_improvement = float(self.min_improvement)
        except (TypeError, ValueError):
            raise ValueError(
                "acceptance.min_improvement must be a finite number "
                "in [0, 2]."
            ) from None

        if (
            not math.isfinite(min_improvement)
            or min_improvement < 0.0
            or min_improvement > 2.0
        ):
            raise ValueError(
                "acceptance.min_improvement must be a finite number "
                f"in [0, 2], got {self.min_improvement!r}."
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
            confidence_level=float(
                data.get(
                    "confidence_level",
                    0.95,
                )
            ),
            bootstrap_samples=coerce_exact_int(
                "acceptance.bootstrap_samples",
                data.get(
                    "bootstrap_samples",
                    5000,
                ),
            ),
        )
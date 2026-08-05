from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from grid_topology_ai.config._mapping import (
    ConfigMapping,
    get_section,
)
from grid_topology_ai.config._validation import (
    coerce_exact_int,
    require_positive,
)
from grid_topology_ai.config.evaluation import EvaluationConfig


_ARENA_DEFAULTS = {
    "simulations": 40,
    "depth": 3,
    "max_steps": 5,
    "top_k": 20,
    "exploration_quota": 2,
    "pf_alg": 3,
    "widening_coefficient": 2.0,
    "widening_exponent": 0.5,
    "random_seed": 42,
    "gamma": 1.0,
    "c_puct": 2.0,
    "prior_exponent": 0.5,
    "use_continuation_gate": True,
    "allow_handoff_with_hard_overloads": False,
    "num_workers": 1,
    "batch_size": 5,
    "device": "cpu",
    "output_csv_name": "tuning_results.csv",
    "output_json_name": "tuning_metrics.json",
}


def _default_arena_config() -> EvaluationConfig:
    return EvaluationConfig.from_mapping(_ARENA_DEFAULTS)


@dataclass(frozen=True, slots=True)
class CheckpointSelectionConfig:
    enabled: bool = False
    tuning_csv: Path | None = None
    tuning_raw_dir: Path | None = None

    candidates_per_metric: int = 1
    max_candidates: int = 4
    calibration_bins: int = 10
    metric: str = "physically_secure_rate_requested"

    arena: EvaluationConfig = field(
        default_factory=_default_arena_config
    )

    def __post_init__(self) -> None:
        candidates_per_metric = coerce_exact_int(
            "checkpoint_selection.candidates_per_metric",
            self.candidates_per_metric,
        )
        max_candidates = coerce_exact_int(
            "checkpoint_selection.max_candidates",
            self.max_candidates,
        )
        calibration_bins = coerce_exact_int(
            "checkpoint_selection.calibration_bins",
            self.calibration_bins,
        )

        require_positive(
            "checkpoint_selection.candidates_per_metric",
            candidates_per_metric,
        )
        require_positive(
            "checkpoint_selection.max_candidates",
            max_candidates,
        )
        require_positive(
            "checkpoint_selection.calibration_bins",
            calibration_bins,
        )

        if candidates_per_metric > max_candidates:
            raise ValueError(
                "checkpoint_selection.candidates_per_metric must not "
                "exceed checkpoint_selection.max_candidates."
            )

        metric = str(self.metric).strip()
        if not metric:
            raise ValueError(
                "checkpoint_selection.metric must not be empty."
            )

        has_tuning_csv = self.tuning_csv is not None
        has_tuning_raw_dir = self.tuning_raw_dir is not None
        if has_tuning_csv != has_tuning_raw_dir:
            raise ValueError(
                "checkpoint_selection.tuning_csv and "
                "checkpoint_selection.tuning_raw_dir must be set together."
            )
        if self.enabled and not has_tuning_csv:
            raise ValueError(
                "Enabled checkpoint selection requires tuning_csv and "
                "tuning_raw_dir."
            )

        object.__setattr__(
            self,
            "candidates_per_metric",
            candidates_per_metric,
        )
        object.__setattr__(self, "max_candidates", max_candidates)
        object.__setattr__(self, "calibration_bins", calibration_bins)
        object.__setattr__(self, "metric", metric)

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "CheckpointSelectionConfig":
        tuning_csv = data.get("tuning_csv")
        tuning_raw_dir = data.get("tuning_raw_dir")
        arena_data = {
            **_ARENA_DEFAULTS,
            **get_section(data, "arena", required=False),
        }

        return cls(
            enabled=bool(data.get("enabled", False)),
            tuning_csv=(
                None if tuning_csv is None else Path(tuning_csv)
            ),
            tuning_raw_dir=(
                None
                if tuning_raw_dir is None
                else Path(tuning_raw_dir)
            ),
            candidates_per_metric=coerce_exact_int(
                "checkpoint_selection.candidates_per_metric",
                data.get("candidates_per_metric", 1),
            ),
            max_candidates=coerce_exact_int(
                "checkpoint_selection.max_candidates",
                data.get("max_candidates", 4),
            ),
            calibration_bins=coerce_exact_int(
                "checkpoint_selection.calibration_bins",
                data.get("calibration_bins", 10),
            ),
            metric=str(
                data.get(
                    "metric",
                    "physically_secure_rate_requested",
                )
            ),
            arena=EvaluationConfig.from_mapping(arena_data),
        )

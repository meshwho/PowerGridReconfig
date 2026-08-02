from __future__ import annotations

import math
from dataclasses import dataclass

from grid_topology_ai.config._mapping import ConfigMapping
from grid_topology_ai.config._validation import (
    coerce_exact_int,
    require_choice,
    require_fraction,
    require_non_negative,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    simulations: int = 150
    depth: int = 4
    max_steps: int = 5
    top_k: int = 30
    exploration_quota: int = 2
    pf_alg: int = 3
    widening_coefficient: float = 2.0
    widening_exponent: float = 0.5
    random_seed: int = 42
    gamma: float = 0.95
    c_puct: float = 2.0
    prior_exponent: float = 0.5

    use_continuation_gate: bool = True
    allow_handoff_with_hard_overloads: bool = False

    num_workers: int = 1
    batch_size: int = 5
    device: str = "cpu"

    output_csv_name: str = "eval_results.csv"
    output_json_name: str = "eval_metrics.json"

    def __post_init__(self) -> None:
        require_positive(
            "evaluation.simulations",
            self.simulations,
        )
        require_positive("evaluation.depth", self.depth)
        require_positive(
            "evaluation.max_steps",
            self.max_steps,
        )
        require_positive("evaluation.top_k", self.top_k)
        pf_alg = coerce_exact_int(
            "evaluation.pf_alg",
            self.pf_alg,
        )
        exploration_quota = coerce_exact_int(
            "evaluation.exploration_quota",
            self.exploration_quota,
        )
        object.__setattr__(
            self,
            "exploration_quota",
            exploration_quota,
        )
        require_non_negative(
            "evaluation.exploration_quota",
            exploration_quota,
        )
        random_seed = coerce_exact_int(
            "evaluation.random_seed",
            self.random_seed,
        )
        object.__setattr__(
            self,
            "random_seed",
            random_seed,
        )
        require_non_negative(
            "evaluation.random_seed",
            random_seed,
        )
        object.__setattr__(self, "pf_alg", pf_alg)
        require_choice(
            "evaluation.pf_alg",
            pf_alg,
            {1, 2, 3, 4},
        )
        require_fraction("evaluation.gamma", self.gamma)
        object.__setattr__(
            self,
            "gamma",
            float(self.gamma),
        )

        require_positive(
            "evaluation.c_puct",
            self.c_puct,
        )
        require_positive(
            "evaluation.prior_exponent",
            self.prior_exponent,
        )
        require_non_negative(
            "evaluation.num_workers",
            self.num_workers,
        )
        require_positive(
            "evaluation.batch_size",
            self.batch_size,
        )
        require_choice(
            "evaluation.device",
            self.device,
            {"auto", "cpu", "cuda"},
        )

        if not self.output_csv_name:
            raise ValueError(
                "evaluation.output_csv_name must not be empty."
            )

        if not self.output_json_name:
            raise ValueError(
                "evaluation.output_json_name must not be empty."
            )

        if isinstance(self.widening_coefficient, bool):
            raise ValueError(
                "evaluation.widening_coefficient must "
                "be a finite non-negative number."
            )

        if isinstance(self.widening_exponent, bool):
            raise ValueError(
                "evaluation.widening_exponent must "
                "be a finite number in (0, 1]."
            )

        widening_coefficient = float(self.widening_coefficient)
        widening_exponent = float(self.widening_exponent)

        if (
            not math.isfinite(widening_coefficient)
            or widening_coefficient < 0.0
        ):
            raise ValueError(
                "evaluation.widening_coefficient must "
                "be a finite non-negative number."
            )

        if (
            not math.isfinite(widening_exponent)
            or widening_exponent <= 0.0
            or widening_exponent > 1.0
        ):
            raise ValueError(
                "evaluation.widening_exponent must "
                "be a finite number in (0, 1]."
            )

        object.__setattr__(
            self,
            "widening_coefficient",
            widening_coefficient,
        )
        object.__setattr__(
            self,
            "widening_exponent",
            widening_exponent,
        )

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "EvaluationConfig":
        return cls(
            simulations=int(data.get("simulations", 150)),
            depth=int(data.get("depth", 4)),
            max_steps=int(data.get("max_steps", 5)),
            top_k=int(data.get("top_k", 30)),
            pf_alg=coerce_exact_int(
                "evaluation.pf_alg",
                data.get("pf_alg", 3),
            ),
            widening_coefficient=float(
                data.get("widening_coefficient", 2.0)
            ),
            widening_exponent=float(
                data.get("widening_exponent", 0.5)
            ),
            exploration_quota=coerce_exact_int(
                "evaluation.exploration_quota",
                data.get(
                    "exploration_quota",
                    2,
                ),
            ),
            random_seed=coerce_exact_int(
                "evaluation.random_seed",
                data.get(
                    "random_seed",
                    42,
                ),
            ),
            gamma=float(data.get("gamma", 0.95)),
            c_puct=float(data.get("c_puct", 2.0)),
            prior_exponent=float(
                data.get("prior_exponent", 0.5)
            ),
            use_continuation_gate=bool(
                data.get("use_continuation_gate", True)
            ),
            allow_handoff_with_hard_overloads=bool(
                data.get(
                    "allow_handoff_with_hard_overloads",
                    False,
                )
            ),
            num_workers=int(data.get("num_workers", 1)),
            batch_size=int(data.get("batch_size", 5)),
            device=str(data.get("device", "cpu")),
            output_csv_name=str(
                data.get(
                    "output_csv_name",
                    "eval_results.csv",
                )
            ),
            output_json_name=str(
                data.get(
                    "output_json_name",
                    "eval_metrics.json",
                )
            ),
        )
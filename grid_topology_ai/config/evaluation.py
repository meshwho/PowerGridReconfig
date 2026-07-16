from __future__ import annotations

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
    pf_alg: int = 3

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
        object.__setattr__(self, "pf_alg", pf_alg)
        require_choice(
            "evaluation.pf_alg",
            pf_alg,
            {1, 2, 3, 4},
        )
        require_fraction("evaluation.gamma", self.gamma)
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
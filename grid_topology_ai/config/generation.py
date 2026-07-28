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


_LEGACY_TERMINAL_PENALTY_FIELDS = frozenset(
    f"terminal_{suffix}"
    for suffix in (
        "unsolved_penalty",
        "handoff_penalty",
        "failure_penalty",
        "penalty_weight",
    )
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    simulations: int = 150
    depth: int = 4
    max_steps: int = 5
    top_k: int = 30
    widening_coefficient: float = 2.0
    widening_exponent: float = 0.5
    gamma: float = 0.95
    c_puct: float = 2.0
    prior_exponent: float = 0.5
    selection_temperature: float = 0.0

    use_root_noise: bool = True
    use_continuation_gate: bool = True

    pf_alg: int = 3
    stop_policy: str = "no_hard_overloads"

    def __post_init__(self) -> None:
        require_positive("generation.simulations", self.simulations)
        require_positive("generation.depth", self.depth)
        require_positive("generation.max_steps", self.max_steps)
        require_positive("generation.top_k", self.top_k)
        require_fraction("generation.gamma", self.gamma)
        require_positive("generation.c_puct", self.c_puct)
        require_positive("generation.prior_exponent", self.prior_exponent)
        require_non_negative(
            "generation.selection_temperature",
            self.selection_temperature,
        )
        pf_alg = coerce_exact_int("generation.pf_alg", self.pf_alg)
        object.__setattr__(self, "pf_alg", pf_alg)
        require_choice("generation.pf_alg", pf_alg, {1, 2, 3, 4})
        require_choice(
            "generation.stop_policy",
            self.stop_policy,
            {"never", "solved_only", "no_hard_overloads", "always"},
        )

        if isinstance(self.widening_coefficient, bool):
            raise ValueError(
                "generation.widening_coefficient must "
                "be a finite non-negative number."
            )

        if isinstance(self.widening_exponent, bool):
            raise ValueError(
                "generation.widening_exponent must "
                "be a finite number in (0, 1]."
            )

        widening_coefficient = float(self.widening_coefficient)
        widening_exponent = float(self.widening_exponent)

        if (
            not math.isfinite(widening_coefficient)
            or widening_coefficient < 0.0
        ):
            raise ValueError(
                "generation.widening_coefficient must "
                "be a finite non-negative number."
            )

        if (
            not math.isfinite(widening_exponent)
            or widening_exponent <= 0.0
            or widening_exponent > 1.0
        ):
            raise ValueError(
                "generation.widening_exponent must "
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
    def from_mapping(cls, data: ConfigMapping) -> "GenerationConfig":
        legacy_fields = sorted(_LEGACY_TERMINAL_PENALTY_FIELDS.intersection(data))
        if legacy_fields:
            raise ValueError(
                "Unsupported legacy generation terminal penalty fields: "
                f"{', '.join(legacy_fields)}. Terminal penalties were removed. "
                "Value targets use discounted terminal utility; dense rewards "
                "are diagnostic potential shaping only."
            )

        return cls(
            simulations=int(data.get("simulations", 150)),
            depth=int(data.get("depth", 4)),
            max_steps=int(data.get("max_steps", 5)),
            top_k=int(data.get("top_k", 30)),
            widening_coefficient=float(
                data.get("widening_coefficient", 2.0)
            ),
            widening_exponent=float(
                data.get("widening_exponent", 0.5)
            ),
            gamma=float(data.get("gamma", 0.95)),
            c_puct=float(data.get("c_puct", 2.0)),
            prior_exponent=float(data.get("prior_exponent", 0.5)),
            selection_temperature=float(data.get("selection_temperature", 0.0)),
            use_root_noise=bool(data.get("use_root_noise", True)),
            use_continuation_gate=bool(data.get("use_continuation_gate", True)),
            pf_alg=coerce_exact_int("generation.pf_alg", data.get("pf_alg", 3)),
            stop_policy=str(data.get("stop_policy", "no_hard_overloads")),
        )

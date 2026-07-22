from __future__ import annotations

import warnings
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
class GenerationConfig:
    simulations: int = 150
    depth: int = 4
    max_steps: int = 5
    top_k: int = 30

    gamma: float = 0.95
    c_puct: float = 2.0
    prior_exponent: float = 0.5
    selection_temperature: float = 0.0

    use_root_noise: bool = True
    use_continuation_gate: bool = True

    pf_alg: int = 3
    stop_policy: str = "no_hard_overloads"

    # Deprecated compatibility fields. Self-play generation still reads them,
    # but this config normalizes every value to zero so terminal penalties can
    # no longer create a second optimized return.
    terminal_unsolved_penalty: float = 0.0
    terminal_handoff_penalty: float = 0.0
    terminal_failure_penalty: float = 0.0
    terminal_penalty_weight: float = 0.0

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

        terminal_fields = (
            "terminal_unsolved_penalty",
            "terminal_handoff_penalty",
            "terminal_failure_penalty",
            "terminal_penalty_weight",
        )
        supplied = {}
        for field_name in terminal_fields:
            value = float(getattr(self, field_name))
            require_non_negative(f"generation.{field_name}", value)
            if value != 0.0:
                supplied[field_name] = value
            object.__setattr__(self, field_name, 0.0)
        if supplied:
            warnings.warn(
                "Generation terminal penalties are deprecated and ignored. "
                "Value targets use discounted terminal utility; dense rewards "
                "are diagnostic potential shaping only.",
                DeprecationWarning,
                stacklevel=2,
            )

    @classmethod
    def from_mapping(cls, data: ConfigMapping) -> "GenerationConfig":
        return cls(
            simulations=int(data.get("simulations", 150)),
            depth=int(data.get("depth", 4)),
            max_steps=int(data.get("max_steps", 5)),
            top_k=int(data.get("top_k", 30)),
            gamma=float(data.get("gamma", 0.95)),
            c_puct=float(data.get("c_puct", 2.0)),
            prior_exponent=float(data.get("prior_exponent", 0.5)),
            selection_temperature=float(data.get("selection_temperature", 0.0)),
            use_root_noise=bool(data.get("use_root_noise", True)),
            use_continuation_gate=bool(data.get("use_continuation_gate", True)),
            pf_alg=coerce_exact_int("generation.pf_alg", data.get("pf_alg", 3)),
            stop_policy=str(data.get("stop_policy", "no_hard_overloads")),
            terminal_unsolved_penalty=float(
                data.get("terminal_unsolved_penalty", 0.0)
            ),
            terminal_handoff_penalty=float(
                data.get("terminal_handoff_penalty", 0.0)
            ),
            terminal_failure_penalty=float(
                data.get("terminal_failure_penalty", 0.0)
            ),
            terminal_penalty_weight=float(
                data.get("terminal_penalty_weight", 0.0)
            ),
        )

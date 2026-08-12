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
from grid_topology_ai.topology_actions import ActionSpaceConfig


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
    exploration_quota: int = 2
    gamma: float = 0.95
    c_puct: float = 2.0
    prior_exponent: float = 0.5

    # Positive temperature is used only during the configured
    # early self-play iterations and episode steps.
    selection_temperature: float = 0.0
    temperature_steps: int = 0
    temperature_iterations: int = 0

    use_root_noise: bool = True
    use_continuation_gate: bool = True
    device: str = "cpu"

    pf_alg: int = 3
    stop_policy: str = "no_hard_overloads"

    require_connected_after_switch: bool = True
    min_loading_for_switch_percent: float = 0.0
    closeable_branch_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        require_positive("generation.simulations", self.simulations)
        require_positive("generation.depth", self.depth)
        require_positive("generation.max_steps", self.max_steps)
        require_positive("generation.top_k", self.top_k)
        exploration_quota = coerce_exact_int(
            "generation.exploration_quota",
            self.exploration_quota,
        )
        object.__setattr__(
            self,
            "exploration_quota",
            exploration_quota,
        )
        require_non_negative(
            "generation.exploration_quota",
            exploration_quota,
        )
        require_fraction("generation.gamma", self.gamma)
        object.__setattr__(
            self,
            "gamma",
            float(self.gamma),
        )

        require_positive("generation.c_puct", self.c_puct)
        require_positive("generation.prior_exponent", self.prior_exponent)
        if isinstance(
                self.selection_temperature,
                bool,
        ):
            raise ValueError(
                "generation.selection_temperature must "
                "be a finite non-negative number."
            )

        selection_temperature = float(
            self.selection_temperature
        )

        if (
                not math.isfinite(
                    selection_temperature
                )
                or selection_temperature < 0.0
        ):
            raise ValueError(
                "generation.selection_temperature must "
                "be a finite non-negative number."
            )

        object.__setattr__(
            self,
            "selection_temperature",
            selection_temperature,
        )

        temperature_steps = coerce_exact_int(
            "generation.temperature_steps",
            self.temperature_steps,
        )
        temperature_iterations = coerce_exact_int(
            "generation.temperature_iterations",
            self.temperature_iterations,
        )

        require_non_negative(
            "generation.temperature_steps",
            temperature_steps,
        )
        require_non_negative(
            "generation.temperature_iterations",
            temperature_iterations,
        )

        object.__setattr__(
            self,
            "temperature_steps",
            temperature_steps,
        )
        object.__setattr__(
            self,
            "temperature_iterations",
            temperature_iterations,
        )

        device = str(self.device).strip().lower()
        require_choice(
            "generation.device",
            device,
            {"auto", "cpu", "cuda"},
        )
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        object.__setattr__(self, "device", device)

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

        action_space_config = ActionSpaceConfig(
            require_connected_after_switch=(
                self.require_connected_after_switch
            ),
            min_loading_for_switch_percent=(
                self.min_loading_for_switch_percent
            ),
            closeable_branch_ids=self.closeable_branch_ids,
        )
        object.__setattr__(
            self,
            "require_connected_after_switch",
            action_space_config.require_connected_after_switch,
        )
        object.__setattr__(
            self,
            "min_loading_for_switch_percent",
            action_space_config.min_loading_for_switch_percent,
        )
        object.__setattr__(
            self,
            "closeable_branch_ids",
            action_space_config.closeable_branch_ids,
        )

    @property
    def action_space_config(self) -> ActionSpaceConfig:
        return ActionSpaceConfig(
            require_connected_after_switch=(
                self.require_connected_after_switch
            ),
            min_loading_for_switch_percent=(
                self.min_loading_for_switch_percent
            ),
            closeable_branch_ids=self.closeable_branch_ids,
        )

    @classmethod
    def from_mapping(cls, data: ConfigMapping) -> "GenerationConfig":
        legacy_fields = sorted(_LEGACY_TERMINAL_PENALTY_FIELDS.intersection(data))
        if legacy_fields:
            raise ValueError(
                "Unsupported legacy generation terminal penalty fields: "
                f"{', '.join(legacy_fields)}. Terminal penalties were removed. "
                "Value targets use undiscounted terminal utility; dense rewards "
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
            exploration_quota=coerce_exact_int(
                "generation.exploration_quota",
                data.get(
                    "exploration_quota",
                    2,
                ),
            ),
            gamma=float(data.get("gamma", 0.95)),
            c_puct=float(data.get("c_puct", 2.0)),
            prior_exponent=float(data.get("prior_exponent", 0.5)),
            selection_temperature=float(data.get("selection_temperature", 0.0)),
            temperature_steps=coerce_exact_int(
                "generation.temperature_steps",
                data.get(
                    "temperature_steps",
                    0,
                ),
            ),
            temperature_iterations=coerce_exact_int(
                "generation.temperature_iterations",
                data.get(
                    "temperature_iterations",
                    0,
                ),
            ),
            use_root_noise=bool(data.get("use_root_noise", True)),
            use_continuation_gate=bool(data.get("use_continuation_gate", True)),
            device=str(data.get("device", "cpu")),
            pf_alg=coerce_exact_int("generation.pf_alg", data.get("pf_alg", 3)),
            stop_policy=str(data.get("stop_policy", "no_hard_overloads")),
            require_connected_after_switch=data.get(
                "require_connected_after_switch",
                True,
            ),
            min_loading_for_switch_percent=data.get(
                "min_loading_for_switch_percent",
                0.0,
            ),
            closeable_branch_ids=data.get(
                "closeable_branch_ids",
                (),
            ),
        )
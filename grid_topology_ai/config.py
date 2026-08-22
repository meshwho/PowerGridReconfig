"""Configuration contracts for the Light production pipeline."""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Any

from grid_topology_ai.actions import ActionSpaceConfig

ConfigMapping = Mapping[str, Any]

def require_value(data: ConfigMapping, key: str) -> Any:
    try:
        return data[key]
    except KeyError as exc:
        raise ValueError(f"Missing required configuration key: {key}") from exc

def get_section(
    data: ConfigMapping,
    name: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    value = data.get(name)

    if value is None:
        if required:
            raise ValueError(
                f"Missing required configuration section: {name}"
            )
        return {}

    if not isinstance(value, Mapping):
        raise ValueError(
            f"Configuration section {name!r} must be a mapping."
        )

    return dict(value)

def require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")

def require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(
            f"{name} must be non-negative, got {value}."
        )

def require_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be in [0, 1], got {value}."
        )

def require_choice(
    name: str,
    value: object,
    choices: Collection[object],
) -> None:
    if value not in choices:
        allowed = ", ".join(
            sorted(str(choice) for choice in choices)
        )
        raise ValueError(
            f"{name} must be one of: {allowed}. Got {value!r}."
        )

def coerce_exact_int(
    name: str,
    value: object,
) -> int:
    """Coerce values that unambiguously represent an exact integer."""

    def _error() -> ValueError:
        return ValueError(
            f"{name} must be an exact integer, got {value!r}. "
            "Fractional, non-finite, boolean, empty, or lossy values are not allowed."
        )

    if isinstance(value, bool):
        raise _error()

    if isinstance(value, numbers.Integral):
        return int(value)

    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise _error()
        return int(numeric)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise _error()
        signless = text[1:] if text[:1] in {"+", "-"} else text
        if not signless or not signless.isdecimal():
            raise _error()
        return int(text, 10)

    raise _error()

class QLimitPolicy(StrEnum):
    ENFORCE = "enforce"
    VALIDATE_ONLY = "validate_only"

class IslandPolicy(StrEnum):
    REJECT = "reject"

class ZeroRateAPolicy(StrEnum):
    UNLIMITED = "unlimited"
    ERROR = "error"

def _finite(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    if number <= 0.0 if positive else number < 0.0:
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}.")
    return number

@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    base_mva: float = 100.0
    pf_alg: int = 1
    pf_tolerance: float = 1e-8
    max_iterations: int = 30
    q_limit_policy: QLimitPolicy = QLimitPolicy.ENFORCE
    island_policy: IslandPolicy = IslandPolicy.REJECT
    zero_rate_a_policy: ZeroRateAPolicy = ZeroRateAPolicy.UNLIMITED
    overload_limit_percent: float = 100.0
    hard_overload_limit_percent: float = 120.0
    thermal_tolerance_percent: float = 1e-6
    voltage_tolerance_pu: float = 1e-6
    generator_p_tolerance_mw: float = 1e-6
    generator_q_tolerance_mvar: float = 1e-6
    angle_tolerance_degrees: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.pf_alg, bool) or not isinstance(self.pf_alg, Integral) or self.pf_alg not in {1, 2, 3, 4}:
            raise ValueError("pf_alg must be an exact integer in {1, 2, 3, 4}.")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, Integral) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive exact integer.")
        object.__setattr__(self, "pf_alg", int(self.pf_alg))
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        for name in ("base_mva", "pf_tolerance", "overload_limit_percent"):
            object.__setattr__(self, name, _finite(name, getattr(self, name), positive=True))
        object.__setattr__(self, "hard_overload_limit_percent", _finite("hard_overload_limit_percent", self.hard_overload_limit_percent, positive=True))
        if self.hard_overload_limit_percent < self.overload_limit_percent:
            raise ValueError("hard_overload_limit_percent must be >= overload_limit_percent.")
        for name in ("thermal_tolerance_percent", "voltage_tolerance_pu", "generator_p_tolerance_mw", "generator_q_tolerance_mvar", "angle_tolerance_degrees"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name, enum in (("q_limit_policy", QLimitPolicy), ("island_policy", IslandPolicy), ("zero_rate_a_policy", ZeroRateAPolicy)):
            value = getattr(self, name)
            try:
                object.__setattr__(self, name, enum(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} has unsupported value {value!r}.") from exc

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PhysicsConfig":
        if not isinstance(data, Mapping):
            raise ValueError("physics must be a mapping.")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown physics settings: {sorted(unknown)!r}.")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, object]:
        return {key: (value.value if isinstance(value, StrEnum) else value) for key, value in asdict(self).items()}

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

DEFAULT_PHYSICS_CONFIG = PhysicsConfig()


def physics_config_payload(
    physics_config: PhysicsConfig,
) -> dict[str, object]:
    """Serialize the actual physics configuration."""

    return {"physics_config": physics_config.to_dict()}


def require_physics_config_payload(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_physics_config: PhysicsConfig | None = None,
) -> PhysicsConfig:
    """Validate the actual ``PhysicsConfig`` stored by the current pipeline."""

    raw_config = payload.get("physics_config")
    if isinstance(raw_config, str):
        try:
            raw_config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid physics_config JSON for {source}.") from exc
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"Missing or invalid physics_config for {source}.")
    try:
        observed_config = PhysicsConfig.from_mapping(raw_config)
    except ValueError as exc:
        raise ValueError(f"Invalid physics_config for {source}: {exc}") from exc

    legacy_pf_alg = payload.get("pf_alg")
    if legacy_pf_alg is not None:
        if isinstance(legacy_pf_alg, bool):
            parsed_pf_alg: int | None = None
        elif isinstance(legacy_pf_alg, Integral):
            parsed_pf_alg = int(legacy_pf_alg)
        elif isinstance(legacy_pf_alg, Real) and float(legacy_pf_alg).is_integer():
            parsed_pf_alg = int(legacy_pf_alg)
        elif isinstance(legacy_pf_alg, str) and legacy_pf_alg.strip().isdigit():
            parsed_pf_alg = int(legacy_pf_alg.strip())
        else:
            parsed_pf_alg = None
        if parsed_pf_alg != observed_config.pf_alg:
            raise ValueError(f"PF_ALG conflicts with PhysicsConfig for {source}.")

    if expected_physics_config is not None and observed_config != expected_physics_config:
        raise ValueError(f"PhysicsConfig mismatch for {source}.")
    return observed_config

def resolve_physics_config(
    physics_config: PhysicsConfig | None,
    legacy_pf_alg: int | None,
) -> PhysicsConfig:
    """Resolve compatibility PF_ALG input without creating a second truth."""
    from dataclasses import replace

    if legacy_pf_alg is not None:
        if isinstance(legacy_pf_alg, bool):
            raise ValueError("legacy pf_alg must be an exact integer.")
        if isinstance(legacy_pf_alg, str):
            if not legacy_pf_alg.strip().isdigit():
                raise ValueError("legacy pf_alg must be an exact integer.")
            legacy_pf_alg = int(legacy_pf_alg)
        elif isinstance(legacy_pf_alg, Real) and float(legacy_pf_alg).is_integer():
            legacy_pf_alg = int(legacy_pf_alg)
        elif not isinstance(legacy_pf_alg, Integral):
            raise ValueError("legacy pf_alg must be an exact integer.")

    if physics_config is None:
        return DEFAULT_PHYSICS_CONFIG if legacy_pf_alg is None else replace(
            DEFAULT_PHYSICS_CONFIG, pf_alg=legacy_pf_alg
        )
    if legacy_pf_alg is not None and int(legacy_pf_alg) != physics_config.pf_alg:
        raise ValueError(
            "Legacy pf_alg conflicts with PhysicsConfig: "
            f"{legacy_pf_alg} != {physics_config.pf_alg}."
        )
    return physics_config

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

    pf_alg: int = 1
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
            pf_alg=coerce_exact_int("generation.pf_alg", data.get("pf_alg", 1)),
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

_POLICY_MODES = {"ungated", "constrained"}

@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    simulations: int = 150
    depth: int = 4
    max_steps: int = 5
    top_k: int = 30
    exploration_quota: int = 2
    pf_alg: int = 1
    widening_coefficient: float = 2.0
    widening_exponent: float = 0.5
    random_seed: int = 42
    gamma: float = 0.95
    c_puct: float = 2.0
    prior_exponent: float = 0.5

    policy_mode: str = "ungated"
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

        policy_mode = str(self.policy_mode).strip().lower()
        require_choice(
            "evaluation.policy_mode",
            policy_mode,
            _POLICY_MODES,
        )
        object.__setattr__(
            self,
            "policy_mode",
            policy_mode,
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

    @property
    def primary_policy_mode(self) -> str:
        return self.policy_mode

    @property
    def use_continuation_gate(self) -> bool:
        return self.policy_mode == "constrained"

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
                data.get("pf_alg", 1),
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
            policy_mode=str(data.get("policy_mode", "ungated")),
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

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    examples_per_iteration: int | None = None

    batch_size: int = 64
    learning_rate: float = 3e-4
    value_loss_weight: float = 1.0
    value_huber_delta: float = 1.0
    validation_fraction: float = 0.20
    min_validation_scenarios: int = 1

    num_workers: int = 0
    device: str = "auto"

    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.0

    save_multiple_best: bool = False
    no_tensorboard: bool = True

    def __post_init__(self) -> None:
        require_positive("training.epochs", self.epochs)

        if self.examples_per_iteration is not None:
            require_positive(
                "training.examples_per_iteration",
                self.examples_per_iteration,
            )

        require_positive("training.batch_size", self.batch_size)
        require_positive("training.learning_rate", self.learning_rate)
        require_non_negative(
            "training.value_loss_weight",
            self.value_loss_weight,
        )
        require_positive(
            "training.value_huber_delta",
            self.value_huber_delta,
        )
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("training.validation_fraction must be in (0, 1).")
        require_positive(
            "training.min_validation_scenarios",
            self.min_validation_scenarios,
        )
        require_non_negative("training.num_workers", self.num_workers)
        require_choice(
            "training.device",
            self.device,
            {"auto", "cpu", "cuda"},
        )
        require_positive("training.hidden_dim", self.hidden_dim)
        require_positive("training.num_layers", self.num_layers)
        require_fraction("training.dropout", self.dropout)

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
        *,
        epochs: int = 10,
    ) -> "TrainingConfig":
        examples = data.get("examples_per_iteration")

        return cls(
            epochs=int(epochs),
            examples_per_iteration=(
                None if examples is None else int(examples)
            ),
            batch_size=int(data.get("batch_size", 64)),
            learning_rate=float(data.get("learning_rate", 3e-4)),
            value_loss_weight=float(data.get("value_loss_weight", 1.0)),
            value_huber_delta=float(data.get("value_huber_delta", 1.0)),
            validation_fraction=float(data.get("validation_fraction", 0.20)),
            min_validation_scenarios=int(
                data.get("min_validation_scenarios", 1)
            ),
            num_workers=int(data.get("num_workers", 0)),
            device=str(data.get("device", "auto")),
            hidden_dim=int(data.get("hidden_dim", 128)),
            num_layers=int(data.get("num_layers", 3)),
            dropout=float(data.get("dropout", 0.0)),
            save_multiple_best=bool(
                data.get("save_multiple_best", False)
            ),
            no_tensorboard=bool(data.get("no_tensorboard", True)),
        )

__all__ = [
    "DEFAULT_PHYSICS_CONFIG",
    "EvaluationConfig",
    "GenerationConfig",
    "IslandPolicy",
    "PhysicsConfig",
    "QLimitPolicy",
    "TrainingConfig",
    "ZeroRateAPolicy",
    "resolve_physics_config",
]

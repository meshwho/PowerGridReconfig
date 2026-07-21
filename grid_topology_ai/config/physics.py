"""Immutable contract for AC power-flow and physical-limit semantics."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Any


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
    pf_alg: int = 3
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

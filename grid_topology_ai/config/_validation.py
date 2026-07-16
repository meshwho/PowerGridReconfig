from __future__ import annotations

import math
import numbers
from collections.abc import Collection


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

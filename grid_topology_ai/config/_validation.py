from __future__ import annotations

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
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any

PHYSICAL_LINEAGE_CONTRACT_VERSION = 1
PHYSICAL_LINEAGE_FINGERPRINT_FIELD = "physical_lineage_fingerprint"
_REQUIRED_FIELDS = (
    "base_case_id",
    "load_profile_id",
    "contingency_family_id",
)


def _source_label(source: str | None) -> str:
    text = str(source or "").strip()
    return text or "physical lineage"


def _normalize_identifier(
    value: object,
    *,
    name: str,
    source: str,
) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(
            f"{source} has invalid {name}: {value!r}."
        )

    if isinstance(value, Integral):
        return str(int(value))

    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(
                f"{source} has invalid {name}: {value!r}."
            )
        return str(int(number))

    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "<na>"}:
        raise ValueError(
            f"{source} has invalid {name}: {value!r}."
        )

    try:
        number = Decimal(text)
    except InvalidOperation:
        return text.casefold()

    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(
            f"{source} has invalid {name}: {value!r}."
        )
    return str(int(number))


def _contingency_items(
    value: object,
    *,
    source: str,
) -> list[object]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(
                f"{source} has an empty contingency_family_id."
            )
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source} has invalid contingency_family_id JSON."
                ) from exc
            if not isinstance(decoded, list):
                raise ValueError(
                    f"{source} contingency_family_id JSON must be a list."
                )
            return decoded
        return text.split(",") if "," in text else [text]

    if isinstance(value, Mapping):
        raise ValueError(
            f"{source} contingency_family_id must not be a mapping."
        )

    if isinstance(value, Iterable) and not isinstance(
        value,
        (bytes, bytearray),
    ):
        return list(value)

    return [value]


def normalize_contingency_family(
    value: object,
    *,
    source: str | None = None,
) -> str:
    label = _source_label(source)
    items = _contingency_items(value, source=label)
    normalized = {
        _normalize_identifier(
            item,
            name="contingency_family_id item",
            source=label,
        )
        for item in items
    }
    if not normalized:
        raise ValueError(
            f"{label} has an empty contingency_family_id."
        )
    return ",".join(sorted(normalized))


def _canonical_payload(
    *,
    base_case_id: object,
    load_profile_id: object,
    contingency_family_id: object,
    source: str | None = None,
) -> dict[str, object]:
    label = _source_label(source)
    return {
        "lineage_contract_version": (
            PHYSICAL_LINEAGE_CONTRACT_VERSION
        ),
        "base_case_id": _normalize_identifier(
            base_case_id,
            name="base_case_id",
            source=label,
        ),
        "load_profile_id": _normalize_identifier(
            load_profile_id,
            name="load_profile_id",
            source=label,
        ),
        "contingency_family_id": normalize_contingency_family(
            contingency_family_id,
            source=label,
        ),
    }


def physical_lineage_fingerprint(
    *,
    base_case_id: object,
    load_profile_id: object,
    contingency_family_id: object,
    source: str | None = None,
) -> str:
    payload = _canonical_payload(
        base_case_id=base_case_id,
        load_profile_id=load_profile_id,
        contingency_family_id=contingency_family_id,
        source=source,
    )
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PhysicalLineage:
    base_case_id: str
    load_profile_id: str
    contingency_family_id: str
    fingerprint: str

    @classmethod
    def build(
        cls,
        *,
        base_case_id: object,
        load_profile_id: object,
        contingency_family_id: object,
        source: str | None = None,
    ) -> "PhysicalLineage":
        payload = _canonical_payload(
            base_case_id=base_case_id,
            load_profile_id=load_profile_id,
            contingency_family_id=contingency_family_id,
            source=source,
        )
        fingerprint = physical_lineage_fingerprint(
            base_case_id=payload["base_case_id"],
            load_profile_id=payload["load_profile_id"],
            contingency_family_id=payload[
                "contingency_family_id"
            ],
            source=source,
        )
        return cls(
            base_case_id=str(payload["base_case_id"]),
            load_profile_id=str(payload["load_profile_id"]),
            contingency_family_id=str(
                payload["contingency_family_id"]
            ),
            fingerprint=fingerprint,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "lineage_contract_version": (
                PHYSICAL_LINEAGE_CONTRACT_VERSION
            ),
            "base_case_id": self.base_case_id,
            "load_profile_id": self.load_profile_id,
            "contingency_family_id": (
                self.contingency_family_id
            ),
            PHYSICAL_LINEAGE_FINGERPRINT_FIELD: self.fingerprint,
        }


def physical_lineage_from_row(
    row: Mapping[str, Any],
    *,
    source: str | None = None,
) -> PhysicalLineage:
    label = _source_label(source)
    if not isinstance(row, Mapping):
        raise ValueError(f"{label} row must be a mapping.")

    missing = [
        field
        for field in _REQUIRED_FIELDS
        if field not in row
    ]
    if missing:
        raise ValueError(
            f"{label} is missing physical lineage fields: "
            + ", ".join(missing)
            + "."
        )

    return PhysicalLineage.build(
        base_case_id=row["base_case_id"],
        load_profile_id=row["load_profile_id"],
        contingency_family_id=row[
            "contingency_family_id"
        ],
        source=label,
    )


def _require_fingerprint(
    value: object,
    *,
    source: str,
) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef"
        for character in text
    ):
        raise ValueError(
            f"{source} has invalid "
            f"{PHYSICAL_LINEAGE_FINGERPRINT_FIELD}."
        )
    return text


def require_physical_lineage(
    row: Mapping[str, Any],
    *,
    source: str | None = None,
) -> PhysicalLineage:
    label = _source_label(source)
    lineage = physical_lineage_from_row(
        row,
        source=label,
    )
    if PHYSICAL_LINEAGE_FINGERPRINT_FIELD not in row:
        raise ValueError(
            f"{label} is missing "
            f"{PHYSICAL_LINEAGE_FINGERPRINT_FIELD}."
        )
    declared = _require_fingerprint(
        row[PHYSICAL_LINEAGE_FINGERPRINT_FIELD],
        source=label,
    )
    if declared != lineage.fingerprint:
        raise ValueError(
            f"{label} physical lineage fingerprint mismatch: "
            f"expected {lineage.fingerprint}, observed {declared}."
        )
    return lineage


def require_one_lineage_per_scenario(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str | None = None,
) -> dict[str, str]:
    label = _source_label(source)
    assignments: dict[str, str] = {}

    for index, row in enumerate(rows):
        row_source = f"{label} row {index}"
        if "scenario_id" not in row:
            raise ValueError(
                f"{row_source} is missing scenario_id."
            )
        scenario_id = _normalize_identifier(
            row["scenario_id"],
            name="scenario_id",
            source=row_source,
        )
        lineage = physical_lineage_from_row(
            row,
            source=row_source,
        )
        declared = row.get(
            PHYSICAL_LINEAGE_FINGERPRINT_FIELD
        )
        if declared is not None:
            require_physical_lineage(
                row,
                source=row_source,
            )

        previous = assignments.get(scenario_id)
        if (
            previous is not None
            and previous != lineage.fingerprint
        ):
            raise ValueError(
                f"{label} scenario_id={scenario_id!r} maps "
                "to multiple physical lineages."
            )
        assignments[scenario_id] = lineage.fingerprint

    if not assignments:
        raise ValueError(f"{label} contains no rows.")
    return assignments


def load_physical_lineages(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str | None = None,
) -> dict[str, PhysicalLineage]:
    label = _source_label(source)
    lineages: dict[str, PhysicalLineage] = {}

    for index, row in enumerate(rows):
        row_source = f"{label} row {index}"
        lineage = physical_lineage_from_row(
            row,
            source=row_source,
        )
        declared = row.get(
            PHYSICAL_LINEAGE_FINGERPRINT_FIELD
        )
        if declared is not None:
            require_physical_lineage(
                row,
                source=row_source,
            )
        lineages[lineage.fingerprint] = lineage

    if not lineages:
        raise ValueError(f"{label} contains no rows.")
    return lineages

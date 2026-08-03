from __future__ import annotations

import math

import pytest

from grid_topology_ai.self_play.physical_lineage import (
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
    PhysicalLineage,
    load_physical_lineages,
    normalize_contingency_family,
    physical_lineage_fingerprint,
    physical_lineage_from_row,
    require_one_lineage_per_scenario,
    require_physical_lineage,
)


def _row(
    *,
    scenario_id: object = 1,
    base_case_id: object = "case118",
    load_profile_id: object = 42,
    contingency_family_id: object = ("branch:12",),
) -> dict[str, object]:
    lineage = PhysicalLineage.build(
        base_case_id=base_case_id,
        load_profile_id=load_profile_id,
        contingency_family_id=contingency_family_id,
    )
    return {
        "scenario_id": scenario_id,
        **lineage.as_dict(),
    }


def test_same_physics_has_same_fingerprint_across_scenario_ids() -> None:
    first = _row(scenario_id=1)
    second = _row(scenario_id=99)

    assert (
        first[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
        == second[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
    )


def test_contingency_order_and_duplicates_do_not_change_fingerprint() -> None:
    first = physical_lineage_fingerprint(
        base_case_id="case118",
        load_profile_id=5,
        contingency_family_id=["branch:12", "branch:4"],
    )
    second = physical_lineage_fingerprint(
        base_case_id="CASE118",
        load_profile_id="5.0",
        contingency_family_id=(
            "branch:4",
            "branch:12",
            "branch:4",
        ),
    )

    assert first == second
    assert normalize_contingency_family(
        '["branch:12", "branch:4"]'
    ) == "branch:12,branch:4"


def test_different_load_profile_changes_fingerprint() -> None:
    first = physical_lineage_from_row(_row(load_profile_id=1))
    second = physical_lineage_from_row(_row(load_profile_id=2))

    assert first.fingerprint != second.fingerprint


def test_different_contingency_changes_fingerprint() -> None:
    first = physical_lineage_from_row(
        _row(contingency_family_id=["branch:1"])
    )
    second = physical_lineage_from_row(
        _row(contingency_family_id=["branch:2"])
    )

    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_case_id", ""),
        ("base_case_id", None),
        ("load_profile_id", True),
        ("load_profile_id", math.nan),
        ("contingency_family_id", []),
        ("contingency_family_id", ""),
    ],
)
def test_invalid_lineage_values_are_rejected(
    field: str,
    value: object,
) -> None:
    values = {
        "base_case_id": "case118",
        "load_profile_id": 1,
        "contingency_family_id": ["branch:1"],
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        PhysicalLineage.build(**values)


def test_missing_lineage_field_is_rejected() -> None:
    row = _row()
    del row["load_profile_id"]

    with pytest.raises(
        ValueError,
        match="missing physical lineage fields: load_profile_id",
    ):
        physical_lineage_from_row(row, source="example")


def test_declared_fingerprint_mismatch_is_rejected() -> None:
    row = _row()
    row[PHYSICAL_LINEAGE_FINGERPRINT_FIELD] = "a" * 64

    with pytest.raises(
        ValueError,
        match="fingerprint mismatch",
    ):
        require_physical_lineage(row, source="example")


def test_scenario_cannot_map_to_multiple_lineages() -> None:
    rows = [
        _row(
            scenario_id=7,
            contingency_family_id=["branch:1"],
        ),
        _row(
            scenario_id=7,
            contingency_family_id=["branch:2"],
        ),
    ]

    with pytest.raises(
        ValueError,
        match="maps to multiple physical lineages",
    ):
        require_one_lineage_per_scenario(
            rows,
            source="transitions",
        )


def test_lineage_loaders_accept_multiple_scenarios_per_lineage() -> None:
    rows = [
        _row(scenario_id=1),
        _row(scenario_id=2),
    ]

    assignments = require_one_lineage_per_scenario(rows)
    lineages = load_physical_lineages(rows)

    assert assignments["1"] == assignments["2"]
    assert len(lineages) == 1


def test_fingerprint_is_stable_for_numeric_representations() -> None:
    fingerprints = {
        physical_lineage_fingerprint(
            base_case_id="case118",
            load_profile_id=value,
            contingency_family_id=["branch:1"],
        )
        for value in (42, 42.0, "42", "42.0", "0042")
    }

    assert len(fingerprints) == 1

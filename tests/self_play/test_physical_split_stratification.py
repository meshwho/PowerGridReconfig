from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play.physical_lineage import PhysicalLineage
from grid_topology_ai.self_play.physical_split import (
    PHYSICAL_SPLIT_ASSIGNMENT_STRATEGY,
    assign_physical_split,
    load_physical_split_manifest,
)


def _lineage(index: int) -> PhysicalLineage:
    return PhysicalLineage.build(
        base_case_id="case118",
        load_profile_id=f"load-{index}",
        contingency_family_id=[f"branch:{index}"],
    )


def _row(
    index: int,
    *,
    difficulty: object = "medium",
    outcome: object = "solved",
    step: int = 0,
    scenario_id: int | None = None,
) -> dict[str, object]:
    lineage = _lineage(index)
    return {
        "scenario_id": index if scenario_id is None else scenario_id,
        "state_id": f"state-{index}-{step}",
        "step": step,
        "difficulty_class": difficulty,
        "outcome_class": outcome,
        **lineage.as_dict(),
    }


def _assign(
    frame: pd.DataFrame,
    path: Path,
    *,
    iteration: int = 1,
    fraction: float = 0.2,
) -> dict[str, object]:
    return assign_physical_split(
        frame,
        manifest_path=path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        seed=31,
        validation_fraction=fraction,
        min_validation_lineages=1,
        iteration=iteration,
        source="replay batch",
    )


def _splits(manifest: dict[str, object]) -> dict[str, str]:
    assignments = manifest["assignments"]
    assert isinstance(assignments, dict)
    return {
        fingerprint: str(entry["split"])
        for fingerprint, entry in assignments.items()
    }


def test_each_multi_lineage_stratum_gets_both_splits(
    tmp_path: Path,
) -> None:
    rows = []
    index = 1
    for difficulty in ("simple", "hard"):
        for outcome in ("solved", "handoff_to_redispatch"):
            for _ in range(2):
                rows.append(
                    _row(
                        index,
                        difficulty=difficulty,
                        outcome=outcome,
                    )
                )
                index += 1

    manifest = _assign(pd.DataFrame(rows), tmp_path / "split.json")

    assert manifest["assignment_strategy"] == (
        PHYSICAL_SPLIT_ASSIGNMENT_STRATEGY
    )
    assert len(manifest["strata"]) == 4
    for stratum in manifest["strata"].values():
        assert stratum["lineage_count"] == 2
        assert stratum["train_lineage_count"] == 1
        assert stratum["validation_lineage_count"] == 1


def test_stratified_assignments_ignore_input_row_order(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            index,
            difficulty="hard" if index % 2 else "simple",
            outcome="solved" if index % 3 else "failed",
        )
        for index in range(1, 13)
    ]
    frame = pd.DataFrame(rows)

    first = _assign(frame, tmp_path / "first.json")
    second = _assign(
        frame.sample(frac=1.0, random_state=8).reset_index(drop=True),
        tmp_path / "second.json",
    )

    assert _splits(first) == _splits(second)
    assert first["strata"] == second["strata"]


def test_new_stratum_gets_coverage_without_moving_existing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.json"
    initial = pd.DataFrame(
        [_row(index, difficulty="medium", outcome="solved") for index in range(1, 5)]
    )
    first = _assign(initial, path)
    original = _splits(first)

    new_rows = pd.DataFrame(
        [
            _row(20, difficulty="hard", outcome="failed"),
            _row(21, difficulty="hard", outcome="failed"),
        ]
    )
    second = _assign(new_rows, path, iteration=2)

    for fingerprint, split in original.items():
        assert second["assignments"][fingerprint]["split"] == split

    stratum_id = second["assignments"][_lineage(20).fingerprint]["stratum_id"]
    stratum = second["strata"][stratum_id]
    assert stratum["train_lineage_count"] == 1
    assert stratum["validation_lineage_count"] == 1


def test_singleton_strata_use_global_lineage_target(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        [
            _row(
                index,
                difficulty=f"difficulty-{index}",
                outcome=f"outcome-{index}",
            )
            for index in range(1, 6)
        ]
    )

    manifest = _assign(frame, tmp_path / "split.json")

    assert manifest["lineage_count"] == 5
    assert manifest["validation_lineage_count"] == 1
    assert manifest["train_lineage_count"] == 4


def test_repeated_state_rows_do_not_change_lineage_quota(
    tmp_path: Path,
) -> None:
    rows = [
        _row(index, step=step)
        for index in range(1, 5)
        for step in range(10)
    ]

    manifest = _assign(
        pd.DataFrame(rows),
        tmp_path / "split.json",
        fraction=0.25,
    )

    assert manifest["lineage_count"] == 4
    assert manifest["validation_lineage_count"] == 1


def test_mixed_and_unknown_labels_are_persisted(
    tmp_path: Path,
) -> None:
    mixed = _lineage(1)
    rows = [
        {
            **_row(1, outcome="solved", step=0),
            **mixed.as_dict(),
        },
        {
            **_row(1, outcome="failed", step=1),
            **mixed.as_dict(),
        },
        _row(2, difficulty=float("nan"), outcome=float("nan")),
    ]

    manifest = _assign(pd.DataFrame(rows), tmp_path / "split.json")

    mixed_entry = manifest["assignments"][mixed.fingerprint]
    unknown_entry = manifest["assignments"][_lineage(2).fingerprint]
    assert mixed_entry["outcome_class_at_assignment"] == "mixed"
    assert unknown_entry["difficulty_class"] == "unknown"
    assert unknown_entry["outcome_class_at_assignment"] == "unknown"


def test_existing_outcome_stratum_is_not_rewritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.json"
    first = _assign(
        pd.DataFrame([_row(index) for index in range(1, 5)]),
        path,
    )
    fingerprint = _lineage(1).fingerprint
    original = dict(first["assignments"][fingerprint])

    updated = _assign(
        pd.DataFrame(
            [
                _row(
                    1,
                    outcome="handoff_to_redispatch",
                    scenario_id=101,
                )
            ]
        ),
        path,
        iteration=2,
    )
    entry = updated["assignments"][fingerprint]

    assert entry["split"] == original["split"]
    assert entry["stratum_id"] == original["stratum_id"]
    assert entry["outcome_class_at_assignment"] == (
        original["outcome_class_at_assignment"]
    )
    assert entry["scenario_ids"] == [1, 101]


def test_corrupt_strata_summary_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    _assign(pd.DataFrame([_row(index) for index in range(1, 5)]), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stratum_id = next(iter(payload["strata"]))
    payload["strata"][stratum_id]["lineage_count"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="strata summary mismatch"):
        load_physical_split_manifest(
            path,
            physics_config=DEFAULT_PHYSICS_CONFIG,
            seed=31,
            validation_fraction=0.2,
            min_validation_lineages=1,
        )


def test_assignment_strategy_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    _assign(pd.DataFrame([_row(index) for index in range(1, 5)]), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["assignment_strategy"] = "global_random_v0"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="assignment strategy mismatch"):
        load_physical_split_manifest(
            path,
            physics_config=DEFAULT_PHYSICS_CONFIG,
            seed=31,
            validation_fraction=0.2,
            min_validation_lineages=1,
        )

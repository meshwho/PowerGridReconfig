from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.physical_lineage import PhysicalLineage
from grid_topology_ai.self_play.physical_split import (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    assign_physical_split,
    load_physical_split_manifest,
    split_frame_by_manifest,
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
    scenario_id: object | None = None,
    step: int = 0,
    difficulty: str = "medium",
    outcome: str = "solved",
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


def _frame(count: int = 4) -> pd.DataFrame:
    return pd.DataFrame([_row(index) for index in range(count)])


def _assign(
    frame: pd.DataFrame,
    path: Path,
    *,
    iteration: int = 1,
    seed: int = 17,
    fraction: float = 0.25,
    source_hash: str = "a" * 64,
) -> dict[str, object]:
    return assign_physical_split(
        frame,
        manifest_path=path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        seed=seed,
        validation_fraction=fraction,
        min_validation_lineages=1,
        iteration=iteration,
        source="replay batch",
        source_hashes={"pool_transitions": source_hash},
    )


def _split_assignments(manifest: dict[str, object]) -> dict[str, str]:
    assignments = manifest["assignments"]
    assert isinstance(assignments, dict)
    return {
        fingerprint: str(entry["split"])
        for fingerprint, entry in assignments.items()
    }


def test_initial_assignments_are_independent_of_row_order(
    tmp_path: Path,
) -> None:
    frame = _frame(6)
    first = _assign(frame, tmp_path / "first.json")
    second = _assign(
        frame.sample(frac=1.0, random_state=4).reset_index(drop=True),
        tmp_path / "second.json",
    )

    assert _split_assignments(first) == _split_assignments(second)
    assert first["validation_lineage_count"] == 2
    assert first["train_lineage_count"] == 4


def test_existing_assignments_survive_reload_and_new_lineage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.json"
    first = _assign(_frame(4), path)
    original = _split_assignments(first)

    loaded = load_physical_split_manifest(
        path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        seed=17,
        validation_fraction=0.25,
        min_validation_lineages=1,
        source_hashes={"pool_transitions": "a" * 64},
    )
    assert loaded is not None
    assert _split_assignments(loaded) == original

    updated = _assign(
        pd.DataFrame([_row(4)]),
        path,
        iteration=2,
    )
    for fingerprint, split in original.items():
        assert updated["assignments"][fingerprint]["split"] == split
    assert updated["lineage_count"] == 5
    assert updated["last_updated_iteration"] == 2


def test_same_lineage_accumulates_scenario_ids_without_moving(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.json"
    first = _assign(_frame(4), path)
    lineage = _lineage(0)
    previous_split = first["assignments"][lineage.fingerprint]["split"]
    related = pd.DataFrame(
        [
            {
                **_row(0, scenario_id=99),
                **lineage.as_dict(),
            }
        ]
    )

    updated = _assign(related, path, iteration=2)
    entry = updated["assignments"][lineage.fingerprint]

    assert entry["split"] == previous_split
    assert entry["scenario_ids"] == [0, 99]
    assert entry["assigned_iteration"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": 18}, "seed mismatch"),
        ({"validation_fraction": 0.40}, "validation_fraction mismatch"),
        ({"min_validation_lineages": 2}, "min_validation_lineages mismatch"),
    ],
)
def test_manifest_parameters_cannot_change(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "split.json"
    _assign(_frame(4), path)
    arguments = {
        "physics_config": DEFAULT_PHYSICS_CONFIG,
        "seed": 17,
        "validation_fraction": 0.25,
        "min_validation_lineages": 1,
        "source_hashes": {"pool_transitions": "a" * 64},
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        load_physical_split_manifest(path, **arguments)


def test_manifest_physics_cannot_change(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    _assign(_frame(4), path)
    other = PhysicsConfig(
        overload_limit_percent=115.0,
        hard_overload_limit_percent=135.0,
    )

    with pytest.raises(ValueError, match="PhysicsConfig mismatch"):
        load_physical_split_manifest(
            path,
            physics_config=other,
            seed=17,
            validation_fraction=0.25,
            min_validation_lineages=1,
        )


def test_source_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    _assign(_frame(4), path)

    with pytest.raises(ValueError, match="source hash mismatch"):
        load_physical_split_manifest(
            path,
            physics_config=DEFAULT_PHYSICS_CONFIG,
            seed=17,
            validation_fraction=0.25,
            min_validation_lineages=1,
            source_hashes={"pool_transitions": "b" * 64},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("split", "Invalid physical split"),
        ("rank", "assignment_rank mismatch"),
        ("count", "lineage_count mismatch"),
    ],
)
def test_corrupt_manifest_is_rejected(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = tmp_path / "split.json"
    _assign(_frame(4), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = next(iter(payload["assignments"]))
    if mutation == "split":
        payload["assignments"][fingerprint]["split"] = "test"
    elif mutation == "rank":
        payload["assignments"][fingerprint]["assignment_rank"] = "b" * 64
    else:
        payload["lineage_count"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_physical_split_manifest(
            path,
            physics_config=DEFAULT_PHYSICS_CONFIG,
            seed=17,
            validation_fraction=0.25,
            min_validation_lineages=1,
        )


def test_split_frame_preserves_rows_and_isolates_lineages(
    tmp_path: Path,
) -> None:
    rows = [
        _row(index, step=step)
        for index in range(4)
        for step in range(2)
    ]
    frame = pd.DataFrame(rows)
    manifest = _assign(frame, tmp_path / "split.json")

    train, validation = split_frame_by_manifest(
        frame,
        manifest=manifest,
        source="replay batch",
    )

    assert len(train) + len(validation) == len(frame)
    assert set(train["state_id"]).isdisjoint(validation["state_id"])
    assert set(train["physical_lineage_fingerprint"]).isdisjoint(
        validation["physical_lineage_fingerprint"]
    )
    assert set(train["state_id"]) | set(validation["state_id"]) == set(
        frame["state_id"]
    )


def test_fractional_scenario_id_is_rejected(tmp_path: Path) -> None:
    frame = _frame(4)
    frame["scenario_id"] = frame["scenario_id"].astype(object)
    frame.loc[0, "scenario_id"] = 1.5

    with pytest.raises(ValueError, match="scenario_id"):
        _assign(frame, tmp_path / "split.json")


def test_one_scenario_cannot_have_two_lineages(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            _row(0, scenario_id=7),
            _row(1, scenario_id=7),
            _row(2),
        ]
    )

    with pytest.raises(ValueError, match="multiple physical lineages"):
        _assign(frame, tmp_path / "split.json")


def test_manifest_path_is_stable_at_run_level(tmp_path: Path) -> None:
    paths = SelfPlayPaths(
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        pool_transitions_csv=tmp_path / "pool.csv",
        pool_raw_dir=tmp_path / "pool_raw",
        pool_metadata=tmp_path / "pool.json",
        eval_csv=tmp_path / "eval.csv",
        eval_raw_dir=tmp_path / "eval_raw",
        final_test_csv=tmp_path / "test.csv",
        final_test_raw_dir=tmp_path / "test_raw",
        bootstrap_checkpoint=tmp_path / "bootstrap.pt",
        bootstrap_metrics=tmp_path / "bootstrap.json",
        best_checkpoint=tmp_path / "best.pt",
        best_metrics=tmp_path / "best.json",
    )

    assert paths.physical_split_manifest == (
        tmp_path / "run" / "physical_split_manifest.json"
    )
    assert "iter_" not in str(paths.physical_split_manifest)

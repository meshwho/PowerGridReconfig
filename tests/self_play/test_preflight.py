from __future__ import annotations

import json
from pathlib import Path

import pytest

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play import dataset_isolation as isolation_module
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.physical_lineage import PhysicalLineage
from grid_topology_ai.self_play.preflight import (
    validate_inputs,
    validate_resume_artifacts,
)


def make_paths(tmp_path: Path) -> SelfPlayPaths:
    config = SelfPlayConfig.load(
        "configs/self_play_loop_pilot.yaml"
    )
    return SelfPlayPaths.from_config(config, tmp_path)


def _write_raw_dataset(
    raw_dir: Path,
    scenarios: dict[int, tuple[float, tuple[int, ...]]],
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        str(scenario_id): {
            "load": float(load),
            "outages": list(outages),
        }
        for scenario_id, (load, outages) in scenarios.items()
    }
    (raw_dir / "test_physical_lineages.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def use_test_physical_lineages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_scenario_lineages(
        *,
        raw_dir: str | Path,
        scenario_ids,
    ) -> dict[int, PhysicalLineage]:
        path = Path(raw_dir) / "test_physical_lineages.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[int, PhysicalLineage] = {}
        for value in scenario_ids:
            scenario_id = int(value)
            item = payload.get(str(scenario_id))
            if not isinstance(item, dict):
                raise ValueError(
                    f"Scenario {scenario_id} is missing from physical data: "
                    f"{path}."
                )
            outages = tuple(int(branch) for branch in item["outages"])
            result[scenario_id] = PhysicalLineage.build(
                base_case_id="case118",
                load_profile_id=f"load:{float(item['load']).hex()}",
                contingency_family_id=(
                    tuple(f"branch:{branch}" for branch in outages)
                    or ("none",)
                ),
                source=f"test scenario {scenario_id}",
            )
        return result

    monkeypatch.setattr(
        isolation_module,
        "build_scenario_lineages",
        build_scenario_lineages,
    )


def create_required_inputs(paths: SelfPlayPaths) -> None:
    paths.pool_transitions_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.pool_transitions_csv.write_text(
        "scenario_id\n1\n",
        encoding="utf-8",
    )
    _write_raw_dataset(
        paths.pool_raw_dir,
        {1: (10.0, (1,))},
    )

    paths.eval_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.eval_csv.write_text(
        "scenario_id\n2\n",
        encoding="utf-8",
    )
    _write_raw_dataset(
        paths.eval_raw_dir,
        {2: (20.0, (1,))},
    )

    paths.final_test_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.final_test_csv.write_text(
        "scenario_id\n3\n",
        encoding="utf-8",
    )
    _write_raw_dataset(
        paths.final_test_raw_dir,
        {3: (30.0, (2,))},
    )

    assert paths.tuning_csv is not None
    assert paths.tuning_raw_dir is not None
    paths.tuning_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.tuning_csv.write_text(
        "scenario_id\n4\n",
        encoding="utf-8",
    )
    _write_raw_dataset(
        paths.tuning_raw_dir,
        {4: (40.0, (3,))},
    )


def test_validation_allows_missing_bootstrap_in_plan_mode(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)

    warnings = validate_inputs(
        paths,
        require_bootstrap=False,
    )

    assert len(warnings) == 2


def test_validation_requires_bootstrap_for_real_run(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)

    with pytest.raises(
        FileNotFoundError,
        match="Bootstrap checkpoint",
    ):
        validate_inputs(
            paths,
            require_bootstrap=True,
        )


def test_validation_requires_final_test_csv(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)
    paths.final_test_csv.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Final-test transitions CSV",
    ):
        validate_inputs(
            paths,
            require_bootstrap=False,
        )


def test_validation_requires_tuning_csv(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)
    assert paths.tuning_csv is not None
    paths.tuning_csv.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Tuning transitions CSV",
    ):
        validate_inputs(paths, require_bootstrap=False)


def test_validation_rejects_pool_eval_scenario_overlap(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)
    paths.eval_csv.write_text(
        "scenario_id\n1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Pool and Evaluation",
    ):
        validate_inputs(paths, require_bootstrap=False)


def test_validation_rejects_eval_final_overlap(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)
    paths.final_test_csv.write_text(
        "scenario_id\n2\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Evaluation and final-test",
    ):
        validate_inputs(
            paths,
            require_bootstrap=False,
        )


def test_validation_rejects_pool_final_overlap(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)
    paths.final_test_csv.write_text(
        "scenario_id\n1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Pool and final-test",
    ):
        validate_inputs(
            paths,
            require_bootstrap=False,
        )


@pytest.mark.parametrize(
    ("scenario_id", "message"),
    [
        (1, "Tuning and Pool"),
        (2, "Tuning and Evaluation"),
        (3, "Tuning and final-test"),
    ],
)
def test_validation_rejects_tuning_scenario_overlap(
    tmp_path: Path,
    scenario_id: int,
    message: str,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)
    assert paths.tuning_csv is not None
    paths.tuning_csv.write_text(
        f"scenario_id\n{scenario_id}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        validate_inputs(paths, require_bootstrap=False)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("eval", "Pool and Evaluation physical lineages overlap"),
        ("final_pool", "Pool and final-test physical lineages overlap"),
        (
            "final_eval",
            "Evaluation and final-test physical lineages overlap",
        ),
    ],
)
def test_validation_rejects_physical_overlap_with_different_ids(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)

    if target == "eval":
        _write_raw_dataset(
            paths.eval_raw_dir,
            {2: (10.0, (1,))},
        )
    elif target == "final_pool":
        _write_raw_dataset(
            paths.final_test_raw_dir,
            {3: (10.0, (1,))},
        )
    else:
        _write_raw_dataset(
            paths.final_test_raw_dir,
            {3: (20.0, (1,))},
        )

    with pytest.raises(ValueError, match=message):
        validate_inputs(paths, require_bootstrap=False)


def test_validation_rejects_tuning_physical_overlap(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)
    assert paths.tuning_raw_dir is not None
    _write_raw_dataset(
        paths.tuning_raw_dir,
        {4: (10.0, (1,))},
    )

    with pytest.raises(
        ValueError,
        match="Pool and Tuning physical lineages overlap",
    ):
        validate_inputs(paths, require_bootstrap=False)


def test_validation_does_not_trust_declared_csv_fingerprint(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    create_required_inputs(paths)
    paths.eval_csv.write_text(
        "scenario_id,physical_lineage_fingerprint\n"
        "2,ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\n",
        encoding="utf-8",
    )
    _write_raw_dataset(
        paths.eval_raw_dir,
        {2: (10.0, (1,))},
    )

    with pytest.raises(
        ValueError,
        match="Pool and Evaluation physical lineages overlap",
    ):
        validate_inputs(paths, require_bootstrap=False)


def test_resume_requires_all_runtime_artifacts(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Resume best checkpoint",
    ):
        validate_resume_artifacts(paths)


def test_resume_requires_physical_split_manifest(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    for path in (
        paths.best_checkpoint,
        paths.best_metrics,
        paths.pool_metadata,
        paths.replay_manifest,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        FileNotFoundError,
        match="Resume physical split manifest",
    ):
        validate_resume_artifacts(paths)

    paths.physical_split_manifest.write_text(
        "{}\n",
        encoding="utf-8",
    )
    validate_resume_artifacts(paths)

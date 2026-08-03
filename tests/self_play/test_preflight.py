from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.paths import SelfPlayPaths
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
    bus_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []

    for scenario_id, (load, outages) in scenarios.items():
        bus_rows.extend(
            [
                {
                    "scenario": scenario_id,
                    "bus": 1,
                    "Pd": load,
                    "Qd": load * 0.2,
                },
                {
                    "scenario": scenario_id,
                    "bus": 2,
                    "Pd": load * 0.5,
                    "Qd": load * 0.1,
                },
            ]
        )
        for branch_id, buses in (
            (1, (1, 2)),
            (2, (2, 1)),
        ):
            branch_rows.append(
                {
                    "scenario": scenario_id,
                    "idx": branch_id,
                    "from_bus": buses[0],
                    "to_bus": buses[1],
                    "r": 0.01 * branch_id,
                    "x": 0.02 * branch_id,
                    "b": 0.001 * branch_id,
                    "br_status": (
                        0 if branch_id in outages else 1
                    ),
                }
            )

    pd.DataFrame(bus_rows).to_parquet(
        raw_dir / "bus_data.parquet",
        index=False,
    )
    pd.DataFrame(branch_rows).to_parquet(
        raw_dir / "branch_data.parquet",
        index=False,
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

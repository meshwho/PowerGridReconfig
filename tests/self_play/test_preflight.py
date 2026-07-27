from pathlib import Path

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


def create_required_inputs(paths: SelfPlayPaths) -> None:
    paths.pool_transitions_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.pool_transitions_csv.write_text(
        "scenario_id\n1\n",
        encoding="utf-8",
    )

    paths.pool_raw_dir.mkdir(parents=True)

    paths.eval_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.eval_csv.write_text(
        "scenario_id\n2\n",
        encoding="utf-8",
    )

    paths.eval_raw_dir.mkdir(parents=True)

    paths.final_test_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.final_test_csv.write_text(
        "scenario_id\n3\n",
        encoding="utf-8",
    )

    paths.final_test_raw_dir.mkdir(
        parents=True,
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


def test_resume_requires_all_runtime_artifacts(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Resume best checkpoint",
    ):
        validate_resume_artifacts(paths)

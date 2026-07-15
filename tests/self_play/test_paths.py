from pathlib import Path

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.paths import SelfPlayPaths


def test_resolves_relative_paths_from_project_root(
    tmp_path: Path,
) -> None:
    config = SelfPlayConfig.load(
        "configs/self_play_loop_pilot.yaml"
    )

    paths = SelfPlayPaths.from_config(
        config=config,
        project_root=tmp_path,
    )

    assert paths.run_dir == (
        tmp_path / "runs/self_play_pilot"
    )
    assert paths.replay_dir == (
        tmp_path
        / "runs/self_play_pilot"
        / "replay_buffer"
    )
    assert paths.iteration_dir(2) == (
        tmp_path
        / "runs/self_play_pilot"
        / "iter_002"
    )
    assert paths.best_checkpoint == (
        tmp_path
        / "runs/self_play_pilot"
        / "checkpoints/best.pt"
    )


def test_discover_project_root_from_nested_directory(tmp_path: Path) -> None:
    from grid_topology_ai.self_play.paths import discover_project_root

    project_root = tmp_path / "repo"
    nested = project_root / "a" / "b"
    (project_root / "grid_topology_ai").mkdir(parents=True)
    (project_root / "scripts").mkdir()
    nested.mkdir(parents=True)

    assert discover_project_root(nested) == project_root


def test_discover_project_root_from_file(tmp_path: Path) -> None:
    from grid_topology_ai.self_play.paths import discover_project_root

    project_root = tmp_path / "repo"
    config = project_root / "configs" / "self_play.yaml"
    (project_root / "grid_topology_ai").mkdir(parents=True)
    (project_root / "scripts").mkdir()
    config.parent.mkdir()
    config.write_text("run_name: test\n", encoding="utf-8")

    assert discover_project_root(config) == project_root


def test_discover_project_root_raises_outside_repository(tmp_path: Path) -> None:
    from grid_topology_ai.self_play.paths import discover_project_root

    outside = tmp_path / "outside"
    outside.mkdir()

    try:
        discover_project_root(outside)
    except RuntimeError as exc:
        assert "Could not discover project root" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

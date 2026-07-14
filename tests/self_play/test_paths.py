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

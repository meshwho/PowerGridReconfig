from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grid_topology_ai.config import SelfPlayConfig


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def discover_project_root(start: str | Path | None = None) -> Path:
    """
    Find repository root by walking upward until project markers are found.
    """

    current = Path.cwd() if start is None else Path(start).resolve()

    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (
            (candidate / "grid_topology_ai").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not discover project root. Run from inside PowerGridReconfig."
    )


@dataclass(frozen=True, slots=True)
class SelfPlayPaths:
    project_root: Path
    run_dir: Path

    pool_transitions_csv: Path
    pool_raw_dir: Path
    pool_metadata: Path

    eval_csv: Path
    eval_raw_dir: Path

    bootstrap_checkpoint: Path
    bootstrap_metrics: Path

    best_checkpoint: Path
    best_metrics: Path

    @classmethod
    def from_config(
        cls,
        config: SelfPlayConfig,
        project_root: str | Path,
    ) -> "SelfPlayPaths":
        root = Path(project_root).resolve()

        return cls(
            project_root=root,
            run_dir=_resolve(root, config.checkpoint_dir),
            pool_transitions_csv=_resolve(
                root,
                config.pool.transitions_csv,
            ),
            pool_raw_dir=_resolve(
                root,
                config.pool.raw_dir,
            ),
            pool_metadata=_resolve(
                root,
                config.pool.metadata_path,
            ),
            eval_csv=_resolve(root, config.eval_csv),
            eval_raw_dir=_resolve(root, config.eval_raw_dir),
            bootstrap_checkpoint=_resolve(
                root,
                config.bootstrap_checkpoint,
            ),
            bootstrap_metrics=_resolve(
                root,
                config.bootstrap_eval_metrics,
            ),
            best_checkpoint=_resolve(
                root,
                config.best_checkpoint_path,
            ),
            best_metrics=_resolve(
                root,
                config.best_metrics_path,
            ),
        )

    @property
    def replay_dir(self) -> Path:
        return self.run_dir / "replay_buffer"

    @property
    def replay_manifest(self) -> Path:
        return self.replay_dir / "buffer_manifest.json"

    @property
    def learning_curve(self) -> Path:
        return self.run_dir / "learning_curve.csv"

    @property
    def resolved_config(self) -> Path:
        return self.run_dir / "self_play_loop.resolved.yaml"

    def iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / f"iter_{iteration:03d}"
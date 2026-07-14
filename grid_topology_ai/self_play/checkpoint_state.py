from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from grid_topology_ai.self_play.paths import SelfPlayPaths


@dataclass(frozen=True, slots=True)
class BestState:
    checkpoint: Path
    metrics: dict[str, object]


def _load_metrics(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, Mapping):
        raise ValueError(f"Best metrics JSON must be an object: {path}")

    return dict(payload)


def _save_metrics(path: Path, metrics: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            dict(metrics),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def initialize_best_state(
    *,
    paths: SelfPlayPaths,
) -> BestState:
    paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.best_metrics.parent.mkdir(parents=True, exist_ok=True)

    if not paths.best_checkpoint.exists():
        print("Initializing self-play best checkpoint from bootstrap.")
        print(f"Bootstrap checkpoint: {paths.bootstrap_checkpoint}")
        print(f"Best checkpoint:      {paths.best_checkpoint}")
        shutil.copy2(paths.bootstrap_checkpoint, paths.best_checkpoint)

    if not paths.best_metrics.exists():
        print("Initializing self-play best metrics from bootstrap.")
        print(f"Bootstrap metrics: {paths.bootstrap_metrics}")
        print(f"Best metrics:      {paths.best_metrics}")
        shutil.copy2(paths.bootstrap_metrics, paths.best_metrics)

    return BestState(
        checkpoint=paths.best_checkpoint,
        metrics=_load_metrics(paths.best_metrics),
    )


def promote_candidate(
    *,
    candidate_checkpoint: Path,
    candidate_metrics: Mapping[str, object],
    paths: SelfPlayPaths,
) -> BestState:
    if not candidate_checkpoint.is_file():
        raise FileNotFoundError(
            f"Candidate checkpoint not found: {candidate_checkpoint}"
        )

    paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.best_metrics.parent.mkdir(parents=True, exist_ok=True)

    metrics = dict(candidate_metrics)
    shutil.copy2(candidate_checkpoint, paths.best_checkpoint)
    _save_metrics(paths.best_metrics, metrics)

    return BestState(
        checkpoint=paths.best_checkpoint,
        metrics=metrics,
    )

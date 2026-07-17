from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.acceptance import (
    require_metrics_semantic_versions,
)


@dataclass(frozen=True, slots=True)
class BestState:
    checkpoint: Path
    metrics: dict[str, object]


def initialize_best_state(
    *,
    paths: SelfPlayPaths,
) -> BestState:
    from grid_topology_ai.training.checkpoints import load_checkpoint_payload

    paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.best_metrics.parent.mkdir(parents=True, exist_ok=True)

    if not paths.best_checkpoint.exists():
        load_checkpoint_payload(paths.bootstrap_checkpoint, map_location="cpu")
        print("Initializing self-play best checkpoint from bootstrap.")
        print(f"Bootstrap checkpoint: {paths.bootstrap_checkpoint}")
        print(f"Best checkpoint:      {paths.best_checkpoint}")
        shutil.copy2(paths.bootstrap_checkpoint, paths.best_checkpoint)

    load_checkpoint_payload(paths.best_checkpoint, map_location="cpu")

    if not paths.best_metrics.exists():
        bootstrap_metrics = load_json(paths.bootstrap_metrics)
        require_metrics_semantic_versions(
            bootstrap_metrics,
            source=str(paths.bootstrap_metrics),
        )
        print("Initializing self-play best metrics from bootstrap.")
        print(f"Bootstrap metrics: {paths.bootstrap_metrics}")
        print(f"Best metrics:      {paths.best_metrics}")
        shutil.copy2(paths.bootstrap_metrics, paths.best_metrics)

    best_metrics = load_json(paths.best_metrics)
    require_metrics_semantic_versions(
        best_metrics,
        source=str(paths.best_metrics),
    )

    return BestState(
        checkpoint=paths.best_checkpoint,
        metrics=best_metrics,
    )


def promote_candidate(
    *,
    candidate_checkpoint: Path,
    candidate_metrics: Mapping[str, object],
    paths: SelfPlayPaths,
) -> BestState:
    from grid_topology_ai.training.checkpoints import load_checkpoint_payload

    if not candidate_checkpoint.is_file():
        raise FileNotFoundError(
            f"Candidate checkpoint not found: {candidate_checkpoint}"
        )

    load_checkpoint_payload(candidate_checkpoint, map_location="cpu")
    require_metrics_semantic_versions(
        candidate_metrics,
        source="candidate metrics",
    )

    paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.best_metrics.parent.mkdir(parents=True, exist_ok=True)

    metrics = dict(candidate_metrics)
    shutil.copy2(candidate_checkpoint, paths.best_checkpoint)
    save_json(metrics, paths.best_metrics)

    return BestState(
        checkpoint=paths.best_checkpoint,
        metrics=metrics,
    )

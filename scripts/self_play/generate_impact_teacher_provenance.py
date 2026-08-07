from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from scripts.self_play import generate_impact_teacher_parallel_fast as teacher


_original_make_task_config = teacher.make_task_config
_original_load_scenario_checkpoints = teacher.load_scenario_checkpoints


def make_task_config(args: argparse.Namespace) -> dict[str, Any]:
    task_config = _original_make_task_config(args)
    physics_config = teacher.PhysicsConfig.from_mapping(
        task_config["physics_config"]
    )
    task_config.update(teacher.physics_provenance(physics_config))
    return task_config


def load_scenario_checkpoints(
    checkpoint_path: Path,
    allowed_scenario_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    results = _original_load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=allowed_scenario_ids,
    )
    return {
        scenario_id: result
        for scenario_id, result in results.items()
        if result.get("reason") != "exception"
    }


def main() -> None:
    teacher.make_task_config = make_task_config
    teacher.load_scenario_checkpoints = load_scenario_checkpoints
    teacher.main()


if __name__ == "__main__":
    main()

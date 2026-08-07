from __future__ import annotations

import argparse
from typing import Any

from scripts.self_play import generate_impact_teacher_parallel_fast as teacher


_original_make_task_config = teacher.make_task_config


def make_task_config(args: argparse.Namespace) -> dict[str, Any]:
    task_config = _original_make_task_config(args)
    physics_config = teacher.PhysicsConfig.from_mapping(
        task_config["physics_config"]
    )
    task_config.update(teacher.physics_provenance(physics_config))
    return task_config


def main() -> None:
    teacher.make_task_config = make_task_config
    teacher.main()


if __name__ == "__main__":
    main()

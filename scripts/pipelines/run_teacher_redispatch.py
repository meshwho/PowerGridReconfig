from __future__ import annotations

import sys
from collections.abc import Sequence

from scripts.pipelines import run_teacher_by_difficulty as pipeline


REDISPATCH_TEACHER_MODULE = (
    "scripts.self_play.generate_impact_teacher_redispatch"
)


def _option_value(argv: Sequence[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None

    value_index = index + 1
    if value_index >= len(argv):
        return None
    return str(argv[value_index])


def canonical_argv(argv: Sequence[str]) -> list[str]:
    result = [str(value) for value in argv]

    if "--teacher-module" not in result:
        result.extend(
            [
                "--teacher-module",
                REDISPATCH_TEACHER_MODULE,
            ]
        )

    if "--run-name" not in result:
        dataset_name = _option_value(result, "--dataset-name")
        if dataset_name:
            profile = _option_value(result, "--profile") or "full"
            suffix = (
                "teacher_redispatch_smoke_v1"
                if profile == "smoke"
                else "teacher_redispatch_v1"
            )
            result.extend(
                [
                    "--run-name",
                    f"{dataset_name}_{suffix}",
                ]
            )

    return result


def main() -> None:
    sys.argv[:] = canonical_argv(sys.argv)
    pipeline.main()


if __name__ == "__main__":
    main()

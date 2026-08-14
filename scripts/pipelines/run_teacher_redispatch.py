from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from scripts.pipelines import run_teacher_by_difficulty as pipeline


REDISPATCH_TEACHER_MODULE = (
    "scripts.self_play.generate_impact_teacher_redispatch_staged"
)
_SAFE_AUTO_WORKER_MAX = 4
_SAFE_WORKER_MEMORY_MB = 2400.0
_SAFE_MEMORY_RESERVE_MB = 3072.0
_WORKER_INIT_CONCURRENCY_OPTION = "--worker-init-concurrency"
_WORKER_INIT_CONCURRENCY_ENV = "POWERGRID_TEACHER_INIT_CONCURRENCY"
_DEFAULT_WORKER_INIT_CONCURRENCY = 1
_PF_CACHE_DIR_OPTION = "--pf-cache-dir"
_PF_CACHE_DIR_ENV = "POWERGRID_PF_CACHE_DIR"


def _option_value(argv: Sequence[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None

    value_index = index + 1
    if value_index >= len(argv):
        return None
    return str(argv[value_index])


def _safe_auto_workers() -> int:
    cpu_cap = max(
        min((os.cpu_count() or 2) - 1, _SAFE_AUTO_WORKER_MAX),
        1,
    )

    try:
        import psutil
    except Exception:
        return cpu_cap

    try:
        available_mb = float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        return cpu_cap

    usable_mb = max(available_mb - _SAFE_MEMORY_RESERVE_MB, 0.0)
    memory_cap = max(int(usable_mb // _SAFE_WORKER_MEMORY_MB), 1)
    return max(min(cpu_cap, memory_cap), 1)


def _replace_auto_workers(argv: list[str]) -> None:
    workers = _option_value(argv, "--num-workers")
    safe_workers = str(_safe_auto_workers())

    if workers is None:
        argv.extend(["--num-workers", safe_workers])
        return

    if workers.strip().lower() != "auto":
        return

    value_index = argv.index("--num-workers") + 1
    argv[value_index] = safe_workers


def _pop_worker_init_concurrency(argv: list[str]) -> int:
    try:
        option_index = argv.index(_WORKER_INIT_CONCURRENCY_OPTION)
    except ValueError:
        return _DEFAULT_WORKER_INIT_CONCURRENCY

    value_index = option_index + 1
    if value_index >= len(argv):
        raise ValueError(
            f"{_WORKER_INIT_CONCURRENCY_OPTION} requires a positive integer."
        )

    raw_value = str(argv[value_index]).strip()
    try:
        concurrency = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{_WORKER_INIT_CONCURRENCY_OPTION} requires a positive integer, "
            f"got {raw_value!r}."
        ) from exc

    if concurrency <= 0:
        raise ValueError(
            f"{_WORKER_INIT_CONCURRENCY_OPTION} must be >= 1, got {concurrency}."
        )

    del argv[option_index : value_index + 1]
    return concurrency


def _pop_pf_cache_dir(argv: list[str]) -> str | None:
    option_count = argv.count(_PF_CACHE_DIR_OPTION)
    if option_count == 0:
        return None
    if option_count > 1:
        raise ValueError(f"{_PF_CACHE_DIR_OPTION} may be passed only once.")

    option_index = argv.index(_PF_CACHE_DIR_OPTION)
    value_index = option_index + 1
    if value_index >= len(argv):
        raise ValueError(f"{_PF_CACHE_DIR_OPTION} requires a directory path.")

    cache_dir = str(argv[value_index]).strip()
    if not cache_dir:
        raise ValueError(f"{_PF_CACHE_DIR_OPTION} requires a non-empty directory path.")

    del argv[option_index : value_index + 1]
    return cache_dir


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

    _replace_auto_workers(result)
    return result


def main() -> None:
    argv = canonical_argv(sys.argv)
    try:
        init_concurrency = _pop_worker_init_concurrency(argv)
        pf_cache_dir = _pop_pf_cache_dir(argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    os.environ[_WORKER_INIT_CONCURRENCY_ENV] = str(init_concurrency)
    if pf_cache_dir is not None:
        os.environ[_PF_CACHE_DIR_ENV] = pf_cache_dir

    sys.argv[:] = argv
    pipeline.main()


if __name__ == "__main__":
    main()

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
_PF_WARM_SHADOW_OPTION = "--pf-warm-shadow"
_PF_WARM_SHADOW_ENV = "POWERGRID_PF_WARM_SHADOW"
_PF_WARM_SHADOW_RATE_OPTION = "--pf-warm-shadow-rate"
_PF_WARM_SHADOW_RATE_ENV = "POWERGRID_PF_WARM_SHADOW_RATE"
_PF_WARM_SHADOW_MAX_PAIRS_OPTION = "--pf-warm-shadow-max-pairs"
_PF_WARM_SHADOW_MAX_PAIRS_ENV = "POWERGRID_PF_WARM_SHADOW_MAX_PAIRS"
_PF_WARM_MAX_CANDIDATES_OPTION = "--pf-warm-max-candidates"
_PF_WARM_MAX_CANDIDATES_ENV = "POWERGRID_PF_WARM_MAX_CANDIDATES"
_DEFAULT_PF_WARM_SHADOW_RATE = 0.05
_DEFAULT_PF_WARM_SHADOW_MAX_PAIRS = 50_000
_DEFAULT_PF_WARM_MAX_CANDIDATES = 16


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


def _pop_runtime_value(argv: list[str], option: str) -> str | None:
    count = argv.count(option)
    if count == 0:
        return None
    if count > 1:
        raise ValueError(f"{option} may be passed only once.")

    option_index = argv.index(option)
    value_index = option_index + 1
    if value_index >= len(argv):
        raise ValueError(f"{option} requires a value.")

    value = str(argv[value_index]).strip()
    if not value:
        raise ValueError(f"{option} requires a non-empty value.")

    del argv[option_index : value_index + 1]
    return value


def _pop_runtime_flag(argv: list[str], option: str) -> bool:
    count = argv.count(option)
    if count > 1:
        raise ValueError(f"{option} may be passed only once.")
    if count == 0:
        return False
    argv.remove(option)
    return True


def _pop_pf_cache_dir(argv: list[str]) -> str | None:
    return _pop_runtime_value(argv, _PF_CACHE_DIR_OPTION)


def _pop_warm_shadow_settings(
    argv: list[str],
) -> tuple[bool, float, int, int]:
    enabled = _pop_runtime_flag(argv, _PF_WARM_SHADOW_OPTION)
    rate_raw = _pop_runtime_value(argv, _PF_WARM_SHADOW_RATE_OPTION)
    max_pairs_raw = _pop_runtime_value(argv, _PF_WARM_SHADOW_MAX_PAIRS_OPTION)
    max_candidates_raw = _pop_runtime_value(argv, _PF_WARM_MAX_CANDIDATES_OPTION)

    if not enabled and any(
        value is not None
        for value in (rate_raw, max_pairs_raw, max_candidates_raw)
    ):
        raise ValueError(
            "Warm-shadow tuning options require --pf-warm-shadow."
        )

    if not enabled:
        return (
            False,
            _DEFAULT_PF_WARM_SHADOW_RATE,
            _DEFAULT_PF_WARM_SHADOW_MAX_PAIRS,
            _DEFAULT_PF_WARM_MAX_CANDIDATES,
        )

    try:
        rate = (
            _DEFAULT_PF_WARM_SHADOW_RATE
            if rate_raw is None
            else float(rate_raw)
        )
    except ValueError as exc:
        raise ValueError(
            f"{_PF_WARM_SHADOW_RATE_OPTION} must be a number between 0 and 1."
        ) from exc
    if not 0.0 <= rate <= 1.0:
        raise ValueError(
            f"{_PF_WARM_SHADOW_RATE_OPTION} must be between 0 and 1."
        )

    try:
        max_pairs = (
            _DEFAULT_PF_WARM_SHADOW_MAX_PAIRS
            if max_pairs_raw is None
            else int(max_pairs_raw)
        )
    except ValueError as exc:
        raise ValueError(
            f"{_PF_WARM_SHADOW_MAX_PAIRS_OPTION} must be a positive integer."
        ) from exc
    if max_pairs <= 0:
        raise ValueError(
            f"{_PF_WARM_SHADOW_MAX_PAIRS_OPTION} must be >= 1."
        )

    try:
        max_candidates = (
            _DEFAULT_PF_WARM_MAX_CANDIDATES
            if max_candidates_raw is None
            else int(max_candidates_raw)
        )
    except ValueError as exc:
        raise ValueError(
            f"{_PF_WARM_MAX_CANDIDATES_OPTION} must be a positive integer."
        ) from exc
    if max_candidates <= 0:
        raise ValueError(
            f"{_PF_WARM_MAX_CANDIDATES_OPTION} must be >= 1."
        )

    return True, rate, max_pairs, max_candidates


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
        (
            warm_shadow,
            warm_shadow_rate,
            warm_shadow_max_pairs,
            warm_max_candidates,
        ) = _pop_warm_shadow_settings(argv)

        effective_cache_dir = (
            pf_cache_dir
            if pf_cache_dir is not None
            else os.environ.get(_PF_CACHE_DIR_ENV, "").strip() or None
        )
        if warm_shadow and effective_cache_dir is None:
            raise ValueError(
                f"{_PF_WARM_SHADOW_OPTION} requires {_PF_CACHE_DIR_OPTION}."
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    os.environ[_WORKER_INIT_CONCURRENCY_ENV] = str(init_concurrency)
    if pf_cache_dir is not None:
        os.environ[_PF_CACHE_DIR_ENV] = pf_cache_dir

    if warm_shadow:
        os.environ[_PF_WARM_SHADOW_ENV] = "1"
        os.environ[_PF_WARM_SHADOW_RATE_ENV] = str(warm_shadow_rate)
        os.environ[_PF_WARM_SHADOW_MAX_PAIRS_ENV] = str(warm_shadow_max_pairs)
        os.environ[_PF_WARM_MAX_CANDIDATES_ENV] = str(warm_max_candidates)
    else:
        os.environ.pop(_PF_WARM_SHADOW_ENV, None)
        os.environ.pop(_PF_WARM_SHADOW_RATE_ENV, None)
        os.environ.pop(_PF_WARM_SHADOW_MAX_PAIRS_ENV, None)
        os.environ.pop(_PF_WARM_MAX_CANDIDATES_ENV, None)

    sys.argv[:] = argv
    pipeline.main()


if __name__ == "__main__":
    main()

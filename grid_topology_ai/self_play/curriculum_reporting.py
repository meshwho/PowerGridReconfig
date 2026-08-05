from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any

from grid_topology_ai.config.pool import CurriculumSamplingConfig
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
)
from grid_topology_ai.self_play.pool_sampling import (
    refresh_priorities,
    sample_curriculum_from_pool,
)


@dataclass(frozen=True, slots=True)
class CurriculumSamplingReport:
    scenario_ids: tuple[int, ...]
    report: dict[str, Any]
    path: Path


def _scenario_mapping(
    pool_metadata: Mapping[str, object],
) -> Mapping[str, Mapping[str, Any]] | None:
    scenarios = pool_metadata.get("scenarios")
    if not isinstance(scenarios, Mapping):
        return None
    if not all(isinstance(meta, Mapping) for meta in scenarios.values()):
        return None
    return scenarios  # type: ignore[return-value]


def _difficulty(meta: Mapping[str, Any]) -> str:
    value = str(meta.get("difficulty_class", "unknown")).strip()
    return value or "unknown"


def _fraction(count: int, total: int) -> float:
    return 0.0 if total == 0 else float(count / total)


def _finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if isfinite(number) else 0.0


def _is_unvisited(
    meta: Mapping[str, Any],
    *,
    current_iter: int,
    threshold: int,
) -> bool:
    if int(meta.get("times_attempted", 0)) <= 0:
        return True

    age = max(
        int(current_iter)
        - int(meta.get("last_attempted_iter", 0)),
        0,
    )
    return age >= threshold


def _selected_signal_means(
    scenarios: Mapping[str, Mapping[str, Any]],
    scenario_ids: tuple[int, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in (
        "priority",
        "learning_progress",
        "uncertainty",
        "staleness",
    ):
        values = [
            _finite_float(scenarios[str(scenario_id)].get(field, 0.0))
            for scenario_id in scenario_ids
        ]
        result[field] = 0.0 if not values else float(fmean(values))
    return result


def _enrich_report(
    *,
    report: Mapping[str, Any],
    pool_metadata: Mapping[str, object],
    selected_scenario_ids: tuple[int, ...],
    current_iter: int,
    scenario_sampling_seed: int,
    config: CurriculumSamplingConfig,
) -> dict[str, Any]:
    scenarios = _scenario_mapping(pool_metadata)
    if scenarios is None:
        raise ValueError(
            "Curriculum report requires a scenario metadata mapping."
        )

    selected_keys = [str(value) for value in selected_scenario_ids]
    missing = [key for key in selected_keys if key not in scenarios]
    if missing:
        raise ValueError(
            "Curriculum report references unknown scenario IDs: "
            + ", ".join(missing)
        )

    threshold = int(config.stale_after_iterations)
    unvisited_keys = {
        key
        for key, meta in scenarios.items()
        if _is_unvisited(
            meta,
            current_iter=current_iter,
            threshold=threshold,
        )
    }
    selected_unvisited = [
        key for key in selected_keys if key in unvisited_keys
    ]

    pool_by_difficulty = Counter(
        _difficulty(meta) for meta in scenarios.values()
    )
    selected_by_difficulty = Counter(
        _difficulty(scenarios[key]) for key in selected_keys
    )

    by_difficulty = {
        difficulty: {
            "pool_count": sum(
                key in unvisited_keys
                and _difficulty(meta) == difficulty
                for key, meta in scenarios.items()
            ),
            "selected_count": sum(
                key in unvisited_keys
                and _difficulty(scenarios[key]) == difficulty
                for key in selected_keys
            ),
        }
        for difficulty in sorted(pool_by_difficulty)
    }

    enriched = deepcopy(dict(report))
    enriched.update(
        {
            "iteration": int(current_iter),
            "scenario_sampling_seed": int(scenario_sampling_seed),
            "pool_count": int(len(scenarios)),
            "pool_by_difficulty": dict(
                sorted(pool_by_difficulty.items())
            ),
            "selected_by_difficulty": dict(
                sorted(selected_by_difficulty.items())
            ),
            "unvisited_after_n_iterations": {
                "threshold": threshold,
                "pool_count": int(len(unvisited_keys)),
                "pool_fraction": _fraction(
                    len(unvisited_keys),
                    len(scenarios),
                ),
                "selected_count": int(len(selected_unvisited)),
                "selected_fraction": _fraction(
                    len(selected_unvisited),
                    len(selected_keys),
                ),
                "by_difficulty": by_difficulty,
            },
            "selected_signal_means": _selected_signal_means(
                scenarios,
                selected_scenario_ids,
            ),
        }
    )

    enriched["never_solved"]["min_fraction"] = float(
        config.never_solved_min_fraction
    )
    enriched["hard"]["min_fraction"] = float(
        config.hard_min_fraction
    )
    enriched["simple"]["max_fraction"] = float(
        config.simple_max_fraction
    )
    enriched["frontier"]["max_fraction"] = float(
        config.frontier_max_fraction
    )
    return enriched


def _update_learning_curve(
    row: dict[str, object],
    report: Mapping[str, Any],
) -> None:
    never_solved = report["never_solved"]
    hard = report["hard"]
    simple = report["simple"]
    frontier = report["frontier"]
    unvisited = report["unvisited_after_n_iterations"]
    means = report["selected_signal_means"]

    row.update(
        {
            "curriculum_never_solved_fraction": float(
                never_solved["fraction"]
            ),
            "curriculum_hard_fraction": float(hard["fraction"]),
            "curriculum_simple_fraction": float(simple["fraction"]),
            "curriculum_frontier_fraction": float(
                frontier["fraction"]
            ),
            "curriculum_unvisited_pool_fraction": float(
                unvisited["pool_fraction"]
            ),
            "curriculum_unvisited_selected_fraction": float(
                unvisited["selected_fraction"]
            ),
            "curriculum_never_solved_shortfall": int(
                never_solved["shortfall"]
            ),
            "curriculum_hard_shortfall": int(hard["shortfall"]),
            "curriculum_cap_relaxation_count": len(
                report["cap_relaxations"]
            ),
            "curriculum_mean_priority": float(means["priority"]),
            "curriculum_mean_learning_progress": float(
                means["learning_progress"]
            ),
            "curriculum_mean_uncertainty": float(
                means["uncertainty"]
            ),
            "curriculum_mean_staleness": float(means["staleness"]),
        }
    )


def _print_summary(report: Mapping[str, Any]) -> None:
    never_solved = report["never_solved"]
    hard = report["hard"]
    simple = report["simple"]
    frontier = report["frontier"]
    unvisited = report["unvisited_after_n_iterations"]

    print(f"Curriculum sample: {int(report['selected_count'])} scenarios")
    print(
        "  never-solved: "
        f"{int(never_solved['selected'])} / "
        f"target {int(never_solved['target'])}"
    )
    print(
        "  hard:         "
        f"{int(hard['selected'])} / target {int(hard['target'])}"
    )
    print(
        "  simple:       "
        f"{100.0 * float(simple['fraction']):.1f}% / "
        f"cap {100.0 * float(simple['max_fraction']):.1f}%"
    )
    print(
        "  frontier:     "
        f"{100.0 * float(frontier['fraction']):.1f}% / "
        f"cap {100.0 * float(frontier['max_fraction']):.1f}%"
    )
    print(
        f"  stale >= {int(unvisited['threshold'])}:   "
        f"{100.0 * float(unvisited['selected_fraction']):.1f}%"
    )


def prepare_curriculum_sampling(
    *,
    pool_metadata: dict[str, Any],
    n_scenarios: int,
    current_iter: int,
    scenario_sampling_seed: int,
    config: CurriculumSamplingConfig,
    report_path: Path,
) -> CurriculumSamplingReport | None:
    """Resolve, sample, save, and print pre-generation diagnostics."""

    pool_metadata["curriculum_sampling"] = asdict(config)
    scenarios = _scenario_mapping(pool_metadata)
    if scenarios is None or int(pool_metadata.get("schema_version", -1)) < 3:
        return None

    sampling_state = deepcopy(pool_metadata)
    sample = sample_curriculum_from_pool(
        pool_metadata=sampling_state,
        n=n_scenarios,
        seed=scenario_sampling_seed,
        current_iter=current_iter,
        config=config,
    )
    report = _enrich_report(
        report=sample.report,
        pool_metadata=sampling_state,
        selected_scenario_ids=sample.scenario_ids,
        current_iter=current_iter,
        scenario_sampling_seed=scenario_sampling_seed,
        config=config,
    )
    save_json(report, report_path)
    _print_summary(report)
    print(f"Curriculum report: {report_path}")

    return CurriculumSamplingReport(
        scenario_ids=sample.scenario_ids,
        report=report,
        path=report_path,
    )


def record_curriculum_sampling(
    *,
    prepared: CurriculumSamplingReport | None,
    selected_scenario_ids: tuple[int, ...],
    iteration_metadata_path: Path,
    learning_curve_row: dict[str, object],
) -> dict[str, Any] | None:
    if prepared is None:
        return None

    if prepared.scenario_ids != selected_scenario_ids:
        raise RuntimeError(
            "Curriculum diagnostics do not match the sampled scenario IDs."
        )
    if not iteration_metadata_path.is_file():
        raise FileNotFoundError(
            f"Iteration metadata not found: {iteration_metadata_path}"
        )

    iteration_metadata = load_json(iteration_metadata_path)
    extra = iteration_metadata.get("extra")
    hashes = iteration_metadata.get("hashes")
    if not isinstance(extra, dict) or not isinstance(hashes, dict):
        raise ValueError(
            f"Iteration metadata is incomplete: {iteration_metadata_path}"
        )

    report_sha256 = sha256_file(prepared.path)
    hashes["curriculum_sampling_sha256"] = report_sha256
    extra["curriculum_sampling_path"] = str(prepared.path)
    extra["curriculum_sampling_sha256"] = report_sha256
    extra["curriculum_sampling"] = prepared.report
    save_json(iteration_metadata, iteration_metadata_path)

    _update_learning_curve(learning_curve_row, prepared.report)
    return prepared.report


def persist_curriculum_pool_state(
    *,
    pool_metadata: dict[str, Any],
    current_iter: int,
    config: CurriculumSamplingConfig,
    path: Path,
) -> None:
    scenarios = _scenario_mapping(pool_metadata)
    if scenarios is None or int(pool_metadata.get("schema_version", -1)) < 3:
        return

    refresh_priorities(
        pool_metadata,
        current_iter=current_iter,
        config=config,
    )
    save_json(pool_metadata, path)

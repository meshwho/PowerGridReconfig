from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from grid_topology_ai.config.pool import CurriculumSamplingConfig


_SCHEMA_WITH_LEARNING_SIGNALS = 3
_BETA_UNIFORM_STD = math.sqrt(1.0 / 12.0)


@dataclass(frozen=True, slots=True)
class PriorityBreakdown:
    total: float
    learning_progress: float
    uncertainty: float
    staleness: float
    frontier: float
    difficulty_bonus: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "learning_progress": self.learning_progress,
            "uncertainty": self.uncertainty,
            "staleness": self.staleness,
            "frontier": self.frontier,
            "difficulty_bonus": self.difficulty_bonus,
        }


@dataclass(frozen=True, slots=True)
class PoolSample:
    scenario_ids: tuple[int, ...]
    report: dict[str, Any]


def _finite_unit_interval(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if not math.isfinite(number):
        return 0.0

    return float(np.clip(number, 0.0, 1.0))


def _legacy_priority(
    *,
    solve_rate: float,
    times_attempted: int,
    last_attempted_iter: int,
    current_iter: int,
    difficulty_class: str,
) -> float:
    solve_rate = _finite_unit_interval(solve_rate)
    times_attempted = int(times_attempted)
    last_attempted_iter = int(last_attempted_iter)
    current_iter = int(current_iter)

    frontier_score = 4.0 * solve_rate * (1.0 - solve_rate)
    exploration_bonus = 1.0 if times_attempted == 0 else 0.0

    if times_attempted == 0:
        staleness_bonus = 0.0
    else:
        age = max(current_iter - last_attempted_iter, 0)
        staleness_bonus = 0.3 * min(age, 5) / 5.0

    difficulty_weight = {
        "simple": 0.8,
        "medium": 1.0,
        "hard": 1.2,
    }.get(str(difficulty_class), 1.0)

    raw_priority = (
        frontier_score
        + exploration_bonus
        + staleness_bonus
    ) * difficulty_weight

    return float(max(raw_priority, 0.05))


def _difficulty_bonus(difficulty_class: str) -> float:
    return {
        "simple": 0.0,
        "medium": 0.1,
        "hard": 0.2,
    }.get(str(difficulty_class), 0.1)


def compute_priority_breakdown(
    *,
    solve_rate: float,
    learning_progress: float,
    uncertainty: float,
    staleness: float,
    difficulty_class: str,
    config: CurriculumSamplingConfig | None = None,
) -> PriorityBreakdown:
    config = config or CurriculumSamplingConfig()

    solve_rate = _finite_unit_interval(solve_rate)
    progress_signal = _finite_unit_interval(learning_progress)
    uncertainty_signal = _finite_unit_interval(uncertainty)
    staleness_signal = _finite_unit_interval(staleness)

    frontier_signal = 4.0 * solve_rate * (1.0 - solve_rate)

    progress_component = (
        config.learning_progress_weight * progress_signal
    )
    uncertainty_component = (
        config.uncertainty_weight * uncertainty_signal
    )
    staleness_component = (
        config.staleness_weight * staleness_signal
    )
    frontier_component = (
        config.frontier_weight * frontier_signal
    )
    difficulty_component = _difficulty_bonus(difficulty_class)

    total = (
        config.priority_floor
        + progress_component
        + uncertainty_component
        + staleness_component
        + frontier_component
        + difficulty_component
    )

    return PriorityBreakdown(
        total=float(total),
        learning_progress=float(progress_component),
        uncertainty=float(uncertainty_component),
        staleness=float(staleness_component),
        frontier=float(frontier_component),
        difficulty_bonus=float(difficulty_component),
    )


def compute_priority(
    solve_rate: float,
    times_attempted: int,
    last_attempted_iter: int,
    current_iter: int,
    difficulty_class: str,
    *,
    learning_progress: float | None = None,
    uncertainty: float | None = None,
    staleness: float | None = None,
    config: CurriculumSamplingConfig | None = None,
) -> float:
    """
    Compute a scenario priority.

    Calls without learning signals preserve the legacy formula for compatibility
    with schema-v2 metadata. Schema-v3 callers use the explicit signal inputs.
    """

    if (
        learning_progress is None
        and uncertainty is None
        and staleness is None
        and config is None
    ):
        return _legacy_priority(
            solve_rate=solve_rate,
            times_attempted=times_attempted,
            last_attempted_iter=last_attempted_iter,
            current_iter=current_iter,
            difficulty_class=difficulty_class,
        )

    resolved_config = config or CurriculumSamplingConfig()
    attempts = max(int(times_attempted), 0)

    if uncertainty is None:
        uncertainty = 1.0 if attempts == 0 else 0.0

    if staleness is None:
        if attempts == 0:
            staleness = 1.0
        else:
            age = max(
                int(current_iter) - int(last_attempted_iter),
                0,
            )
            staleness = min(
                age / resolved_config.stale_after_iterations,
                1.0,
            )

    breakdown = compute_priority_breakdown(
        solve_rate=solve_rate,
        learning_progress=(
            0.0 if learning_progress is None else learning_progress
        ),
        uncertainty=uncertainty,
        staleness=staleness,
        difficulty_class=difficulty_class,
        config=resolved_config,
    )
    return breakdown.total


def _metadata_schema_version(pool_metadata: Mapping[str, Any]) -> int:
    try:
        return int(pool_metadata.get("schema_version", -1))
    except (TypeError, ValueError, OverflowError):
        return -1


def _resolve_curriculum_config(
    pool_metadata: dict[str, Any],
    config: CurriculumSamplingConfig | None,
) -> CurriculumSamplingConfig:
    if config is not None:
        pool_metadata["curriculum_sampling"] = asdict(config)
        return config

    stored = pool_metadata.get("curriculum_sampling")
    if isinstance(stored, Mapping):
        return CurriculumSamplingConfig.from_mapping(stored)

    return CurriculumSamplingConfig()


def _posterior_uncertainty(meta: Mapping[str, Any]) -> float:
    attempts = max(int(meta.get("times_attempted", 0)), 0)
    solved = min(
        max(int(meta.get("times_solved", 0)), 0),
        attempts,
    )

    alpha = float(solved + 1)
    beta = float(attempts - solved + 1)
    total = alpha + beta
    variance = alpha * beta / (
        total * total * (total + 1.0)
    )

    return float(
        np.clip(
            math.sqrt(variance) / _BETA_UNIFORM_STD,
            0.0,
            1.0,
        )
    )


def _scenario_staleness(
    meta: Mapping[str, Any],
    *,
    current_iter: int,
    config: CurriculumSamplingConfig,
) -> float:
    attempts = max(int(meta.get("times_attempted", 0)), 0)
    if attempts == 0:
        return 1.0

    age = max(
        int(current_iter)
        - int(meta.get("last_attempted_iter", 0)),
        0,
    )
    return float(
        min(age / config.stale_after_iterations, 1.0)
    )


def _priority_weights(
    scenarios: Mapping[str, Mapping[str, Any]],
    scenario_ids: Sequence[str],
) -> np.ndarray:
    priorities = np.array(
        [
            float(
                scenarios[scenario_id].get(
                    "priority",
                    0.05,
                )
            )
            for scenario_id in scenario_ids
        ],
        dtype=np.float64,
    )
    priorities = np.nan_to_num(
        priorities,
        nan=0.05,
        posinf=0.05,
        neginf=0.05,
    )
    return np.maximum(priorities, 0.05)


def _weighted_pick(
    *,
    scenarios: Mapping[str, Mapping[str, Any]],
    scenario_ids: Sequence[str],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    count = min(max(int(count), 0), len(scenario_ids))
    if count == 0:
        return []

    ordered_ids = list(scenario_ids)
    priorities = _priority_weights(scenarios, ordered_ids)
    total_priority = float(priorities.sum())

    if total_priority <= 0.0:
        probabilities = (
            np.ones_like(priorities) / len(priorities)
        )
    else:
        probabilities = priorities / total_priority

    chosen = rng.choice(
        ordered_ids,
        size=count,
        replace=False,
        p=probabilities,
    )
    return [str(scenario_id) for scenario_id in chosen]


def _is_never_solved(meta: Mapping[str, Any]) -> bool:
    return (
        int(meta.get("times_attempted", 0)) > 0
        and int(meta.get("times_solved", 0)) == 0
    )


def _is_hard(meta: Mapping[str, Any]) -> bool:
    return str(meta.get("difficulty_class", "")).lower() == "hard"


def _is_simple(meta: Mapping[str, Any]) -> bool:
    return str(meta.get("difficulty_class", "")).lower() == "simple"


def _is_frontier(
    meta: Mapping[str, Any],
    config: CurriculumSamplingConfig,
) -> bool:
    solve_rate = _finite_unit_interval(
        meta.get("solve_rate", 0.0)
    )
    return (
        config.frontier_solve_rate_min
        <= solve_rate
        <= config.frontier_solve_rate_max
    )


def _legacy_sample(
    *,
    scenarios: Mapping[str, Mapping[str, Any]],
    n: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    ids = list(scenarios.keys())
    chosen = _weighted_pick(
        scenarios=scenarios,
        scenario_ids=ids,
        count=min(n, len(ids)),
        rng=rng,
    )
    return tuple(int(scenario_id) for scenario_id in chosen)


def _fraction(count: int, total: int) -> float:
    return 0.0 if total == 0 else float(count / total)


def sample_curriculum_from_pool(
    pool_metadata: dict[str, Any],
    n: int,
    seed: int | None = None,
    *,
    current_iter: int | None = None,
    config: CurriculumSamplingConfig | None = None,
) -> PoolSample:
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    scenarios = pool_metadata.get("scenarios", {})
    if not scenarios:
        raise ValueError("Pool metadata contains no scenarios.")

    rng = np.random.default_rng(seed)
    schema_version = _metadata_schema_version(pool_metadata)

    if schema_version < _SCHEMA_WITH_LEARNING_SIGNALS:
        scenario_ids = _legacy_sample(
            scenarios=scenarios,
            n=n,
            rng=rng,
        )
        return PoolSample(
            scenario_ids=scenario_ids,
            report={
                "mode": "legacy",
                "requested_count": n,
                "selected_count": len(scenario_ids),
            },
        )

    resolved_config = _resolve_curriculum_config(
        pool_metadata,
        config,
    )
    if current_iter is None:
        current_iter = (
            int(pool_metadata.get("last_updated_iteration", 0))
            + 1
        )

    refresh_priorities(
        pool_metadata,
        current_iter=int(current_iter),
        config=resolved_config,
    )

    ids = sorted(scenarios.keys(), key=int)
    target_count = min(n, len(ids))

    never_solved_ids = [
        scenario_id
        for scenario_id in ids
        if _is_never_solved(scenarios[scenario_id])
    ]
    hard_ids = [
        scenario_id
        for scenario_id in ids
        if _is_hard(scenarios[scenario_id])
    ]

    never_solved_target = math.ceil(
        target_count
        * resolved_config.never_solved_min_fraction
    )
    hard_target = math.ceil(
        target_count
        * resolved_config.hard_min_fraction
    )

    selected: list[str] = []
    selected_set: set[str] = set()

    def add_candidates(
        candidate_ids: Sequence[str],
        count: int,
    ) -> None:
        available = [
            scenario_id
            for scenario_id in candidate_ids
            if scenario_id not in selected_set
        ]
        picks = _weighted_pick(
            scenarios=scenarios,
            scenario_ids=available,
            count=min(count, target_count - len(selected)),
            rng=rng,
        )
        selected.extend(picks)
        selected_set.update(picks)

    add_candidates(
        never_solved_ids,
        never_solved_target,
    )

    selected_hard = sum(
        _is_hard(scenarios[scenario_id])
        for scenario_id in selected
    )
    add_candidates(
        hard_ids,
        max(hard_target - selected_hard, 0),
    )

    simple_limit = math.floor(
        target_count * resolved_config.simple_max_fraction
    )
    frontier_limit = math.floor(
        target_count * resolved_config.frontier_max_fraction
    )
    enforce_simple_cap = True
    enforce_frontier_cap = True
    cap_relaxations: list[str] = []

    while len(selected) < target_count:
        remaining = [
            scenario_id
            for scenario_id in ids
            if scenario_id not in selected_set
        ]
        if not remaining:
            break

        selected_simple = sum(
            _is_simple(scenarios[scenario_id])
            for scenario_id in selected
        )
        selected_frontier = sum(
            _is_frontier(
                scenarios[scenario_id],
                resolved_config,
            )
            for scenario_id in selected
        )

        eligible: list[str] = []
        for scenario_id in remaining:
            meta = scenarios[scenario_id]

            if (
                enforce_simple_cap
                and _is_simple(meta)
                and selected_simple >= simple_limit
            ):
                continue

            if (
                enforce_frontier_cap
                and _is_frontier(meta, resolved_config)
                and selected_frontier >= frontier_limit
            ):
                continue

            eligible.append(scenario_id)

        if not eligible:
            if (
                enforce_frontier_cap
                and any(
                    _is_frontier(
                        scenarios[scenario_id],
                        resolved_config,
                    )
                    for scenario_id in remaining
                )
            ):
                enforce_frontier_cap = False
                cap_relaxations.append(
                    "frontier_max_fraction"
                )
                continue

            if (
                enforce_simple_cap
                and any(
                    _is_simple(scenarios[scenario_id])
                    for scenario_id in remaining
                )
            ):
                enforce_simple_cap = False
                cap_relaxations.append(
                    "simple_max_fraction"
                )
                continue

            eligible = remaining

        add_candidates(eligible, 1)

    selected_never_solved = sum(
        _is_never_solved(scenarios[scenario_id])
        for scenario_id in selected
    )
    selected_hard = sum(
        _is_hard(scenarios[scenario_id])
        for scenario_id in selected
    )
    selected_simple = sum(
        _is_simple(scenarios[scenario_id])
        for scenario_id in selected
    )
    selected_frontier = sum(
        _is_frontier(
            scenarios[scenario_id],
            resolved_config,
        )
        for scenario_id in selected
    )
    selected_by_difficulty = Counter(
        str(
            scenarios[scenario_id].get(
                "difficulty_class",
                "unknown",
            )
        )
        for scenario_id in selected
    )

    report = {
        "mode": "curriculum",
        "requested_count": n,
        "target_count": target_count,
        "selected_count": len(selected),
        "never_solved": {
            "available": len(never_solved_ids),
            "target": never_solved_target,
            "selected": selected_never_solved,
            "shortfall": max(
                never_solved_target - selected_never_solved,
                0,
            ),
            "fraction": _fraction(
                selected_never_solved,
                len(selected),
            ),
        },
        "hard": {
            "available": len(hard_ids),
            "target": hard_target,
            "selected": selected_hard,
            "shortfall": max(
                hard_target - selected_hard,
                0,
            ),
            "fraction": _fraction(
                selected_hard,
                len(selected),
            ),
        },
        "simple": {
            "limit": simple_limit,
            "selected": selected_simple,
            "fraction": _fraction(
                selected_simple,
                len(selected),
            ),
        },
        "frontier": {
            "limit": frontier_limit,
            "selected": selected_frontier,
            "fraction": _fraction(
                selected_frontier,
                len(selected),
            ),
        },
        "selected_by_difficulty": dict(
            sorted(selected_by_difficulty.items())
        ),
        "cap_relaxations": cap_relaxations,
    }

    return PoolSample(
        scenario_ids=tuple(
            int(scenario_id)
            for scenario_id in selected
        ),
        report=report,
    )


def sample_from_pool(
    pool_metadata: dict[str, Any],
    n: int,
    seed: int | None = None,
    *,
    current_iter: int | None = None,
    config: CurriculumSamplingConfig | None = None,
) -> list[int]:
    sample = sample_curriculum_from_pool(
        pool_metadata=pool_metadata,
        n=n,
        seed=seed,
        current_iter=current_iter,
        config=config,
    )
    return list(sample.scenario_ids)


def refresh_priorities(
    pool_metadata: dict[str, Any],
    *,
    current_iter: int,
    config: CurriculumSamplingConfig | None = None,
) -> dict[str, Any]:
    scenarios = pool_metadata.get("scenarios", {})
    if not scenarios:
        return pool_metadata

    if _metadata_schema_version(pool_metadata) < (
        _SCHEMA_WITH_LEARNING_SIGNALS
    ):
        for meta in scenarios.values():
            meta["priority"] = _legacy_priority(
                solve_rate=float(meta.get("solve_rate", 0.0)),
                times_attempted=int(
                    meta.get("times_attempted", 0)
                ),
                last_attempted_iter=int(
                    meta.get("last_attempted_iter", 0)
                ),
                current_iter=int(current_iter),
                difficulty_class=str(
                    meta.get("difficulty_class", "unknown")
                ),
            )
        return pool_metadata

    resolved_config = _resolve_curriculum_config(
        pool_metadata,
        config,
    )

    for meta in scenarios.values():
        uncertainty = (
            _finite_unit_interval(meta["uncertainty"])
            if "uncertainty" in meta
            else _posterior_uncertainty(meta)
        )
        staleness = _scenario_staleness(
            meta,
            current_iter=current_iter,
            config=resolved_config,
        )
        meta["uncertainty"] = uncertainty
        meta["staleness"] = staleness

        breakdown = compute_priority_breakdown(
            solve_rate=float(meta.get("solve_rate", 0.0)),
            learning_progress=float(
                meta.get("learning_progress", 0.0)
            ),
            uncertainty=uncertainty,
            staleness=staleness,
            difficulty_class=str(
                meta.get("difficulty_class", "unknown")
            ),
            config=resolved_config,
        )

        meta["priority"] = breakdown.total
        meta["priority_components"] = breakdown.as_dict()

    return pool_metadata

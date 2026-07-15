from __future__ import annotations

from typing import Any

import numpy as np


def compute_priority(
    solve_rate: float,
    times_attempted: int,
    last_attempted_iter: int,
    current_iter: int,
    difficulty_class: str,
) -> float:
    """
    Compute scenario sampling priority.

    Highest priority is assigned to frontier scenarios:
    solve_rate around 0.5.

    Never-solved and always-solved scenarios still keep a small non-zero chance.
    """

    solve_rate = float(np.clip(solve_rate, 0.0, 1.0))
    times_attempted = int(times_attempted)
    last_attempted_iter = int(last_attempted_iter)
    current_iter = int(current_iter)

    frontier_score = 4.0 * solve_rate * (1.0 - solve_rate)

    if times_attempted == 0:
        exploration_bonus = 1.0
    else:
        exploration_bonus = 0.0

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

def sample_from_pool(
    pool_metadata: dict[str, Any],
    n: int,
    seed: int | None = None,
) -> list[int]:
    """
    Prioritized sampling without replacement.

    Returns scenario_id values as integers.
    """

    scenarios = pool_metadata.get("scenarios", {})

    if not scenarios:
        raise ValueError("Pool metadata contains no scenarios.")

    ids = list(scenarios.keys())

    priorities = np.array(
        [
            float(scenarios[scenario_id].get("priority", 0.05))
            for scenario_id in ids
        ],
        dtype=np.float64,
    )

    priorities = np.nan_to_num(
        priorities,
        nan=0.05,
        posinf=0.05,
        neginf=0.05,
    )

    priorities = np.maximum(priorities, 0.05)

    total_priority = float(priorities.sum())

    if total_priority <= 0.0:
        probabilities = np.ones_like(priorities) / len(priorities)
    else:
        probabilities = priorities / total_priority

    rng = np.random.default_rng(seed)

    chosen = rng.choice(
        ids,
        size=min(int(n), len(ids)),
        replace=False,
        p=probabilities,
    )

    return [int(scenario_id) for scenario_id in chosen]



def refresh_priorities(
    pool_metadata: dict[str, Any],
    *,
    current_iter: int,
) -> dict[str, Any]:
    scenarios = pool_metadata.get("scenarios", {})
    for meta in scenarios.values():
        meta["priority"] = compute_priority(
            solve_rate=float(meta.get("solve_rate", 0.0)),
            times_attempted=int(meta.get("times_attempted", 0)),
            last_attempted_iter=int(meta.get("last_attempted_iter", 0)),
            current_iter=int(current_iter),
            difficulty_class=str(meta.get("difficulty_class", "unknown")),
        )
    return pool_metadata

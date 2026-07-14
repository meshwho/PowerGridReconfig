from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 2


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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


def _read_transition_pool(transitions_csv: str | Path) -> pd.DataFrame:
    transitions_csv = Path(transitions_csv)

    if not transitions_csv.exists():
        raise FileNotFoundError(f"Transitions CSV not found: {transitions_csv}")

    df = pd.read_csv(transitions_csv)

    if "scenario_id" not in df.columns:
        raise ValueError(
            f"Transitions CSV must contain scenario_id column: {transitions_csv}"
        )

    if "difficulty_class" not in df.columns:
        df = df.copy()
        df["difficulty_class"] = "unknown"

    pool = (
        df[["scenario_id", "difficulty_class"]]
        .drop_duplicates(subset=["scenario_id"])
        .copy()
    )

    pool["scenario_id"] = pool["scenario_id"].astype(int)
    pool["difficulty_class"] = pool["difficulty_class"].astype(str)

    pool = pool.sort_values("scenario_id", ascending=True).reset_index(drop=True)

    return pool


def initialize_pool_metadata(
    transitions_csv: str | Path,
    path: str | Path,
    *,
    current_iter: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create or load pool_metadata.json.

    The scenario pool itself is fixed. Only per-scenario statistics are updated
    after self-play iterations.
    """

    path = Path(path)

    if path.exists() and not overwrite:
        metadata = load_json(path)

        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported pool metadata schema_version: "
                f"{metadata.get('schema_version')}. Expected {SCHEMA_VERSION}."
            )

        if "scenarios" not in metadata:
            raise ValueError(f"Pool metadata has no scenarios field: {path}")

        return metadata

    pool = _read_transition_pool(transitions_csv)

    scenarios: dict[str, dict[str, Any]] = {}

    for row in pool.itertuples(index=False):
        scenario_id = int(row.scenario_id)
        difficulty_class = str(row.difficulty_class)

        priority = compute_priority(
            solve_rate=0.0,
            times_attempted=0,
            last_attempted_iter=0,
            current_iter=current_iter,
            difficulty_class=difficulty_class,
        )

        scenarios[str(scenario_id)] = {
            "difficulty_class": difficulty_class,
            "times_attempted": 0,
            "times_solved": 0,
            "solve_rate": 0.0,
            "last_attempted_iter": 0,
            "last_solved_iter": None,
            "avg_steps_when_solved": None,
            "priority": priority,
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "transitions_csv": str(Path(transitions_csv)),
        "last_updated_iteration": int(current_iter),
        "scenarios": scenarios,
    }

    save_json(metadata, path)

    return metadata


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


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()

    return text in {"1", "true", "yes", "y"}


def _extract_episode_results(
    episode_results: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(episode_results, pd.DataFrame):
        df = episode_results.copy()
    else:
        df = pd.DataFrame(episode_results)

    if df.empty:
        return df

    if "scenario_id" not in df.columns:
        raise ValueError("Episode results must contain scenario_id column.")

    if "solved" not in df.columns:
        raise ValueError("Episode results must contain solved column.")

    if "steps" not in df.columns:
        if "step" in df.columns:
            per_scenario_steps = (
                df.groupby("scenario_id")["step"]
                .max()
                .reset_index()
            )
            per_scenario_steps["steps"] = per_scenario_steps["step"].astype(int) + 1

            solved = (
                df.groupby("scenario_id")["solved"]
                .max()
                .reset_index()
            )

            df = solved.merge(
                per_scenario_steps[["scenario_id", "steps"]],
                on="scenario_id",
                how="left",
            )
        else:
            df = df.copy()
            df["steps"] = np.nan

    return df


def update_pool_metadata(
    pool_metadata: dict[str, Any],
    episode_results: list[dict[str, Any]] | pd.DataFrame,
    current_iter: int,
    *,
    ema_alpha: float = 0.30,
) -> dict[str, Any]:
    """
    Update per-scenario solve statistics and priorities after one iteration.

    This function mutates and returns pool_metadata.
    """

    df = _extract_episode_results(episode_results)

    if df.empty:
        pool_metadata["last_updated_iteration"] = int(current_iter)
        return pool_metadata

    scenarios = pool_metadata.get("scenarios", {})

    if not scenarios:
        raise ValueError("Pool metadata contains no scenarios.")

    df = df.copy()
    df["scenario_id"] = df["scenario_id"].astype(int)
    df["solved_bool"] = df["solved"].map(_safe_bool)

    grouped = df.groupby("scenario_id", sort=True)

    for scenario_id, group in grouped:
        scenario_key = str(int(scenario_id))

        if scenario_key not in scenarios:
            # Ignore scenario IDs that are not part of the fixed pool.
            continue

        meta = scenarios[scenario_key]

        attempts = int(len(group))
        solved_count = int(group["solved_bool"].sum())

        old_solve_rate = float(meta.get("solve_rate", 0.0))
        iteration_solve_rate = solved_count / attempts if attempts > 0 else 0.0

        if int(meta.get("times_attempted", 0)) == 0:
            updated_solve_rate = iteration_solve_rate
        else:
            updated_solve_rate = (
                (1.0 - float(ema_alpha)) * old_solve_rate
                + float(ema_alpha) * iteration_solve_rate
            )

        meta["times_attempted"] = int(meta.get("times_attempted", 0)) + attempts
        meta["times_solved"] = int(meta.get("times_solved", 0)) + solved_count
        meta["solve_rate"] = float(np.clip(updated_solve_rate, 0.0, 1.0))
        meta["last_attempted_iter"] = int(current_iter)

        if solved_count > 0:
            meta["last_solved_iter"] = int(current_iter)

            solved_steps = group.loc[group["solved_bool"], "steps"]

            if len(solved_steps) > 0 and solved_steps.notna().any():
                new_avg_steps = float(solved_steps.dropna().astype(float).mean())
                old_avg_steps = meta.get("avg_steps_when_solved")

                if old_avg_steps is None:
                    meta["avg_steps_when_solved"] = new_avg_steps
                else:
                    meta["avg_steps_when_solved"] = (
                        (1.0 - float(ema_alpha)) * float(old_avg_steps)
                        + float(ema_alpha) * new_avg_steps
                    )

        meta["priority"] = compute_priority(
            solve_rate=float(meta["solve_rate"]),
            times_attempted=int(meta["times_attempted"]),
            last_attempted_iter=int(meta["last_attempted_iter"]),
            current_iter=int(current_iter),
            difficulty_class=str(meta.get("difficulty_class", "unknown")),
        )

    # Priority depends on current iteration because it includes
    # a staleness bonus. Therefore priorities must be refreshed
    # for every scenario, not only for scenarios attempted now.
    for meta in scenarios.values():
        meta["priority"] = compute_priority(
            solve_rate=float(meta.get("solve_rate", 0.0)),
            times_attempted=int(meta.get("times_attempted", 0)),
            last_attempted_iter=int(meta.get("last_attempted_iter", 0)),
            current_iter=int(current_iter),
            difficulty_class=str(
                meta.get("difficulty_class", "unknown")
            ),
        )

    pool_metadata["last_updated_iteration"] = int(current_iter)

    return pool_metadata

def update_and_save_pool_metadata(
    pool_metadata: dict[str, Any],
    episode_results: list[dict[str, Any]] | pd.DataFrame,
    current_iter: int,
    path: str | Path,
    *,
    ema_alpha: float = 0.30,
) -> dict[str, Any]:
    """
    Convenience wrapper: update metadata and write it to disk.
    """

    updated = update_pool_metadata(
        pool_metadata=pool_metadata,
        episode_results=episode_results,
        current_iter=current_iter,
        ema_alpha=ema_alpha,
    )

    save_json(updated, path)

    return updated
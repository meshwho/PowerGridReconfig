from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.outcome_contract import TerminalOutcomeEvidence
from grid_topology_ai.self_play.pool_sampling import (
    compute_priority,
    refresh_priorities,
)


SCHEMA_VERSION = 4
PREVIOUS_SCHEMA_VERSION = 3
LEGACY_SCHEMA_VERSION = 2
DEFAULT_STALE_AFTER_ITERATIONS = 3
_BETA_UNIFORM_STD = math.sqrt(1.0 / 12.0)


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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _optional_utility(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return float(np.clip(number, -1.0, 1.0))


def _validate_stale_after_iterations(value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(
            "stale_after_iterations must be positive, "
            f"got {value}."
        )
    return value


def _uncertainty(times_attempted: int, times_solved: int) -> float:
    attempts = max(int(times_attempted), 0)
    solved = min(max(int(times_solved), 0), attempts)

    alpha = float(solved + 1)
    beta = float(attempts - solved + 1)
    total = alpha + beta
    variance = alpha * beta / (total * total * (total + 1.0))

    return float(
        np.clip(
            math.sqrt(variance) / _BETA_UNIFORM_STD,
            0.0,
            1.0,
        )
    )


def _staleness(
    *,
    times_attempted: int,
    last_attempted_iter: int,
    current_iter: int,
    stale_after_iterations: int,
) -> float:
    if int(times_attempted) <= 0:
        return 1.0

    age = max(int(current_iter) - int(last_attempted_iter), 0)
    return float(min(age / stale_after_iterations, 1.0))


def _refresh_scenario_signals(
    meta: dict[str, Any],
    *,
    current_iter: int,
    stale_after_iterations: int,
) -> bool:
    before = (
        meta.get("last_iteration_solve_rate"),
        meta.get("solve_rate_delta"),
        meta.get("topology_utility"),
        meta.get("last_iteration_topology_utility"),
        meta.get("topology_utility_delta"),
        meta.get("topology_learning_progress"),
        meta.get("learning_progress"),
        meta.get("uncertainty"),
        meta.get("staleness"),
    )

    last_rate = meta.get("last_iteration_solve_rate")
    if last_rate is not None:
        last_rate = float(
            np.clip(_finite_float(last_rate), 0.0, 1.0)
        )

    meta["last_iteration_solve_rate"] = last_rate
    meta["solve_rate_delta"] = float(
        np.clip(
            _finite_float(meta.get("solve_rate_delta")),
            -1.0,
            1.0,
        )
    )

    topology_utility = _optional_utility(
        meta.get("topology_utility")
    )
    last_topology_utility = _optional_utility(
        meta.get("last_iteration_topology_utility")
    )
    topology_delta = float(
        np.clip(
            _finite_float(meta.get("topology_utility_delta")),
            -2.0,
            2.0,
        )
    )
    topology_progress = float(
        np.clip(
            _finite_float(meta.get("topology_learning_progress")),
            0.0,
            1.0,
        )
    )

    meta["topology_utility"] = topology_utility
    meta["last_iteration_topology_utility"] = (
        last_topology_utility
    )
    meta["topology_utility_delta"] = topology_delta
    meta["topology_learning_progress"] = topology_progress

    if topology_utility is None:
        learning_progress = float(
            np.clip(
                _finite_float(meta.get("learning_progress")),
                0.0,
                1.0,
            )
        )
    else:
        learning_progress = topology_progress

    meta["learning_progress"] = learning_progress
    meta["uncertainty"] = _uncertainty(
        int(meta.get("times_attempted", 0)),
        int(meta.get("times_solved", 0)),
    )
    meta["staleness"] = _staleness(
        times_attempted=int(meta.get("times_attempted", 0)),
        last_attempted_iter=int(
            meta.get("last_attempted_iter", 0)
        ),
        current_iter=current_iter,
        stale_after_iterations=stale_after_iterations,
    )

    after = (
        meta["last_iteration_solve_rate"],
        meta["solve_rate_delta"],
        meta["topology_utility"],
        meta["last_iteration_topology_utility"],
        meta["topology_utility_delta"],
        meta["topology_learning_progress"],
        meta["learning_progress"],
        meta["uncertainty"],
        meta["staleness"],
    )
    return before != after


def _ensure_current_schema(
    metadata: dict[str, Any],
    *,
    current_iter: int,
    stale_after_iterations: int,
) -> bool:
    try:
        version = int(metadata.get("schema_version", -1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Pool metadata schema_version must be an integer."
        ) from exc

    supported_versions = {
        LEGACY_SCHEMA_VERSION,
        PREVIOUS_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }
    if version not in supported_versions:
        expected = ", ".join(
            str(value)
            for value in sorted(supported_versions)
        )
        raise ValueError(
            "Unsupported pool metadata schema_version: "
            f"{metadata.get('schema_version')}. "
            f"Expected one of: {expected}."
        )

    scenarios = metadata.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("Pool metadata has no scenarios mapping.")

    effective_iter = max(
        int(current_iter),
        int(metadata.get("last_updated_iteration", current_iter)),
    )
    changed = version != SCHEMA_VERSION

    for scenario_id, meta in scenarios.items():
        if not isinstance(meta, dict):
            raise ValueError(
                "Pool scenario metadata must be a mapping: "
                f"scenario_id={scenario_id}."
            )

        if version < SCHEMA_VERSION:
            meta.setdefault("topology_utility", None)
            meta.setdefault(
                "last_iteration_topology_utility",
                None,
            )
            meta.setdefault("topology_utility_delta", 0.0)
            meta.setdefault(
                "topology_learning_progress",
                0.0,
            )

        changed |= _refresh_scenario_signals(
            meta,
            current_iter=effective_iter,
            stale_after_iterations=stale_after_iterations,
        )

    metadata["schema_version"] = SCHEMA_VERSION
    return changed


def _refresh_all_signals(
    metadata: dict[str, Any],
    *,
    current_iter: int,
    stale_after_iterations: int,
) -> None:
    for meta in metadata["scenarios"].values():
        _refresh_scenario_signals(
            meta,
            current_iter=current_iter,
            stale_after_iterations=stale_after_iterations,
        )


def _read_transition_pool(
    transitions_csv: str | Path,
) -> pd.DataFrame:
    transitions_csv = Path(transitions_csv)

    if not transitions_csv.exists():
        raise FileNotFoundError(
            f"Transitions CSV not found: {transitions_csv}"
        )

    df = pd.read_csv(transitions_csv)

    if "scenario_id" not in df.columns:
        raise ValueError(
            "Transitions CSV must contain scenario_id column: "
            f"{transitions_csv}"
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

    return pool.sort_values("scenario_id").reset_index(drop=True)


def initialize_pool_metadata(
    transitions_csv: str | Path,
    path: str | Path,
    *,
    current_iter: int = 0,
    overwrite: bool = False,
    stale_after_iterations: int = DEFAULT_STALE_AFTER_ITERATIONS,
) -> dict[str, Any]:
    """
    Create or load pool_metadata.json.

    Schema v2/v3 metadata is migrated in place without losing
    historical solve statistics.
    """

    path = Path(path)
    stale_after_iterations = _validate_stale_after_iterations(
        stale_after_iterations
    )

    if path.exists() and not overwrite:
        metadata = load_json(path)
        changed = _ensure_current_schema(
            metadata,
            current_iter=current_iter,
            stale_after_iterations=stale_after_iterations,
        )
        if changed:
            save_json(metadata, path)
        return metadata

    pool = _read_transition_pool(transitions_csv)
    scenarios: dict[str, dict[str, Any]] = {}

    for row in pool.itertuples(index=False):
        scenario_id = int(row.scenario_id)
        difficulty = str(row.difficulty_class)

        scenarios[str(scenario_id)] = {
            "difficulty_class": difficulty,
            "times_attempted": 0,
            "times_solved": 0,
            "solve_rate": 0.0,
            "last_attempted_iter": 0,
            "last_solved_iter": None,
            "avg_steps_when_solved": None,
            "last_iteration_solve_rate": None,
            "solve_rate_delta": 0.0,
            "topology_utility": None,
            "last_iteration_topology_utility": None,
            "topology_utility_delta": 0.0,
            "topology_learning_progress": 0.0,
            "learning_progress": 0.0,
            "uncertainty": 1.0,
            "staleness": 1.0,
            "priority": compute_priority(
                solve_rate=0.0,
                times_attempted=0,
                last_attempted_iter=0,
                current_iter=current_iter,
                difficulty_class=difficulty,
            ),
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "transitions_csv": str(Path(transitions_csv)),
        "last_updated_iteration": int(current_iter),
        "scenarios": scenarios,
    }
    save_json(metadata, path)
    return metadata


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    return bool(missing) if isinstance(
        missing,
        (bool, np.bool_),
    ) else False


def _strict_utility(value: Any, *, source: str) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{source} must be a finite number in [-1, 1]."
        )

    try:
        utility = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"{source} must be a finite number in [-1, 1]."
        ) from None

    if (
        not math.isfinite(utility)
        or utility < -1.0
        or utility > 1.0
    ):
        raise ValueError(
            f"{source} must be a finite number in [-1, 1]."
        )
    return utility


def _row_topology_utility(
    row: pd.Series,
) -> float | None:
    evidence_json = row.get(
        "terminal_outcome_evidence_json"
    )
    if not _is_missing_scalar(evidence_json):
        evidence = TerminalOutcomeEvidence.from_json(
            evidence_json
        )
        return float(evidence.topology_utility)

    explicit_utility = row.get("topology_utility")
    if not _is_missing_scalar(explicit_utility):
        return _strict_utility(
            explicit_utility,
            source="topology_utility",
        )

    outcome_target = row.get("outcome_value_target")
    if not _is_missing_scalar(outcome_target):
        return _strict_utility(
            outcome_target,
            source="outcome_value_target",
        )

    return None


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
        raise ValueError(
            "Episode results must contain scenario_id column."
        )
    if "solved" not in df.columns:
        raise ValueError(
            "Episode results must contain solved column."
        )

    df = df.copy()
    df["_topology_utility"] = [
        (
            np.nan
            if utility is None
            else float(utility)
        )
        for utility in (
            _row_topology_utility(row)
            for _, row in df.iterrows()
        )
    ]

    if "steps" not in df.columns:
        if "step" in df.columns:
            steps = (
                df.groupby("scenario_id")["step"]
                .max()
                .reset_index()
            )
            steps["steps"] = (
                steps["step"].astype(int) + 1
            )
            solved = (
                df.groupby("scenario_id")["solved"]
                .max()
                .reset_index()
            )
            utilities = (
                df.groupby("scenario_id")[
                    "_topology_utility"
                ]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "_topology_utility": (
                            "topology_utility"
                        )
                    }
                )
            )
            df = (
                solved.merge(
                    steps[["scenario_id", "steps"]],
                    on="scenario_id",
                    how="left",
                )
                .merge(
                    utilities,
                    on="scenario_id",
                    how="left",
                )
            )
        else:
            df["steps"] = np.nan
            df["topology_utility"] = (
                df["_topology_utility"]
            )
            df = df.drop(
                columns=["_topology_utility"]
            )
    else:
        df["topology_utility"] = (
            df["_topology_utility"]
        )
        df = df.drop(columns=["_topology_utility"])

    return df


def _record_missing_result_attempt(
    meta: dict[str, Any],
    current_iter: int,
) -> None:
    meta["times_attempted"] = (
        int(meta.get("times_attempted", 0)) + 1
    )
    meta["last_attempted_iter"] = int(current_iter)
    meta["solve_rate_delta"] = 0.0
    meta["topology_utility_delta"] = 0.0


def _solve_rate_learning_progress(
    *,
    meta: dict[str, Any],
    delta: float,
    old_attempts: int,
    ema_alpha: float,
) -> float:
    old_progress = _finite_float(
        meta.get("learning_progress")
    )
    if (
        old_attempts == 0
        or meta.get("last_iteration_solve_rate") is None
    ):
        return abs(delta)

    return (
        (1.0 - ema_alpha) * old_progress
        + ema_alpha * abs(delta)
    )


def _update_topology_learning_progress(
    *,
    meta: dict[str, Any],
    iteration_utility: float,
    ema_alpha: float,
) -> None:
    old_utility = _optional_utility(
        meta.get("topology_utility")
    )

    if old_utility is None:
        updated_utility = iteration_utility
        delta = 0.0
        progress = 0.0
    else:
        delta = iteration_utility - old_utility
        instantaneous_progress = min(
            abs(delta) / 2.0,
            1.0,
        )
        old_progress = float(
            np.clip(
                _finite_float(
                    meta.get(
                        "topology_learning_progress"
                    )
                ),
                0.0,
                1.0,
            )
        )

        if (
            meta.get(
                "last_iteration_topology_utility"
            )
            is None
        ):
            progress = instantaneous_progress
        else:
            progress = (
                (1.0 - ema_alpha) * old_progress
                + ema_alpha * instantaneous_progress
            )

        updated_utility = (
            (1.0 - ema_alpha) * old_utility
            + ema_alpha * iteration_utility
        )

    meta["topology_utility"] = float(
        np.clip(updated_utility, -1.0, 1.0)
    )
    meta["last_iteration_topology_utility"] = float(
        iteration_utility
    )
    meta["topology_utility_delta"] = float(
        np.clip(delta, -2.0, 2.0)
    )
    meta["topology_learning_progress"] = float(
        np.clip(progress, 0.0, 1.0)
    )
    meta["learning_progress"] = (
        meta["topology_learning_progress"]
    )


def update_pool_metadata(
    pool_metadata: dict[str, Any],
    episode_results: list[dict[str, Any]] | pd.DataFrame,
    current_iter: int,
    *,
    selected_scenario_ids: list[int] | tuple[int, ...] | None = None,
    ema_alpha: float = 0.30,
    stale_after_iterations: int = DEFAULT_STALE_AFTER_ITERATIONS,
) -> dict[str, Any]:
    """
    Update solved coverage and curriculum learning signals.

    New self-play artifacts drive learning progress from terminal
    topology utility. Legacy rows without topology utility retain the
    historical solve-rate learning signal.
    """

    stale_after_iterations = _validate_stale_after_iterations(
        stale_after_iterations
    )
    _ensure_current_schema(
        pool_metadata,
        current_iter=current_iter,
        stale_after_iterations=stale_after_iterations,
    )

    scenarios = pool_metadata["scenarios"]
    if not scenarios:
        raise ValueError("Pool metadata contains no scenarios.")

    df = _extract_episode_results(episode_results)
    selected_ids = {
        int(value) for value in selected_scenario_ids or ()
    }

    if df.empty:
        for scenario_id in sorted(selected_ids):
            meta = scenarios.get(str(scenario_id))
            if meta is not None:
                _record_missing_result_attempt(
                    meta,
                    current_iter,
                )
    else:
        df = df.copy()
        df["scenario_id"] = df["scenario_id"].astype(int)
        df["solved_bool"] = df["solved"].map(_safe_bool)
        attempted_ids: set[int] = set()

        for scenario_id, group in df.groupby(
            "scenario_id",
            sort=True,
        ):
            key = str(int(scenario_id))
            if key not in scenarios:
                continue

            attempted_ids.add(int(scenario_id))
            meta = scenarios[key]
            attempts = int(len(group))
            solved_count = int(
                group["solved_bool"].sum()
            )
            old_attempts = int(
                meta.get("times_attempted", 0)
            )
            old_rate = float(
                meta.get("solve_rate", 0.0)
            )
            iteration_rate = solved_count / attempts
            solve_delta = iteration_rate - old_rate

            if old_attempts == 0:
                updated_rate = iteration_rate
            else:
                updated_rate = (
                    (1.0 - ema_alpha) * old_rate
                    + ema_alpha * iteration_rate
                )

            meta["times_attempted"] = (
                old_attempts + attempts
            )
            meta["times_solved"] = (
                int(meta.get("times_solved", 0))
                + solved_count
            )
            meta["solve_rate"] = float(
                np.clip(updated_rate, 0.0, 1.0)
            )
            meta["last_attempted_iter"] = int(
                current_iter
            )
            meta["last_iteration_solve_rate"] = float(
                iteration_rate
            )
            meta["solve_rate_delta"] = float(
                np.clip(solve_delta, -1.0, 1.0)
            )

            utility_values = pd.to_numeric(
                group["topology_utility"],
                errors="coerce",
            ).to_numpy(dtype=float)
            utility_values = utility_values[
                np.isfinite(utility_values)
            ]

            if utility_values.size:
                iteration_utility = float(
                    utility_values.mean()
                )
                _update_topology_learning_progress(
                    meta=meta,
                    iteration_utility=iteration_utility,
                    ema_alpha=ema_alpha,
                )
            elif meta.get("topology_utility") is None:
                progress = _solve_rate_learning_progress(
                    meta=meta,
                    delta=solve_delta,
                    old_attempts=old_attempts,
                    ema_alpha=ema_alpha,
                )
                meta["learning_progress"] = float(
                    np.clip(progress, 0.0, 1.0)
                )
            else:
                meta["topology_utility_delta"] = 0.0
                meta["learning_progress"] = float(
                    np.clip(
                        _finite_float(
                            meta.get(
                                "topology_learning_progress"
                            )
                        ),
                        0.0,
                        1.0,
                    )
                )

            if solved_count > 0:
                meta["last_solved_iter"] = int(
                    current_iter
                )
                solved_steps = group.loc[
                    group["solved_bool"],
                    "steps",
                ]

                if solved_steps.notna().any():
                    new_avg = float(
                        solved_steps.dropna()
                        .astype(float)
                        .mean()
                    )
                    old_avg = meta.get(
                        "avg_steps_when_solved"
                    )
                    if old_avg is None:
                        meta[
                            "avg_steps_when_solved"
                        ] = new_avg
                    else:
                        meta[
                            "avg_steps_when_solved"
                        ] = (
                            (1.0 - ema_alpha)
                            * float(old_avg)
                            + ema_alpha * new_avg
                        )

        for scenario_id in sorted(
            selected_ids - attempted_ids
        ):
            meta = scenarios.get(str(scenario_id))
            if meta is not None:
                _record_missing_result_attempt(
                    meta,
                    current_iter,
                )

    _refresh_all_signals(
        pool_metadata,
        current_iter=current_iter,
        stale_after_iterations=stale_after_iterations,
    )
    refresh_priorities(
        pool_metadata,
        current_iter=current_iter,
    )

    pool_metadata["schema_version"] = SCHEMA_VERSION
    pool_metadata["last_updated_iteration"] = int(
        current_iter
    )
    return pool_metadata


def update_and_save_pool_metadata(
    pool_metadata: dict[str, Any],
    episode_results: list[dict[str, Any]] | pd.DataFrame,
    current_iter: int,
    path: str | Path,
    *,
    selected_scenario_ids: list[int] | tuple[int, ...] | None = None,
    ema_alpha: float = 0.30,
    stale_after_iterations: int = DEFAULT_STALE_AFTER_ITERATIONS,
) -> dict[str, Any]:
    """Update metadata and write it to disk."""

    updated = update_pool_metadata(
        pool_metadata=pool_metadata,
        episode_results=episode_results,
        current_iter=current_iter,
        selected_scenario_ids=selected_scenario_ids,
        ema_alpha=ema_alpha,
        stale_after_iterations=stale_after_iterations,
    )
    save_json(updated, path)
    return updated

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


PAIRED_COMPARISON_VERSION = 2
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 5000


PAIRED_OUTCOME_FIELDS = (
    "evaluation_success",
    "power_flow_converged",
    "all_values_finite",
    "topology_connected",
    "thermal_solved",
    "thermal_feasible",
    "hard_overload_free",
    "voltage_feasible",
    "generator_p_feasible",
    "generator_q_feasible",
    "angle_difference_feasible",
    "physically_secure",
)


def compare_evaluation_results(
    *,
    parent_csv: str | Path,
    candidate_csv: str | Path,
    policy_mode: str,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    confidence_level = float(confidence_level)

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    if isinstance(bootstrap_samples, bool):
        raise ValueError(
            "bootstrap_samples must be a positive integer."
        )

    parsed_bootstrap_samples = int(bootstrap_samples)

    if (
        parsed_bootstrap_samples <= 0
        or parsed_bootstrap_samples != bootstrap_samples
    ):
        raise ValueError(
            "bootstrap_samples must be a positive integer."
        )

    policy_mode = str(policy_mode).strip()

    if not policy_mode:
        raise ValueError("policy_mode must not be empty.")

    parent = _load_policy_results(
        Path(parent_csv),
        policy_mode=policy_mode,
    )
    candidate = _load_policy_results(
        Path(candidate_csv),
        policy_mode=policy_mode,
    )

    if not parent.index.equals(candidate.index):
        parent_ids = set(parent.index)
        candidate_ids = set(candidate.index)

        parent_only = sorted(
            parent_ids - candidate_ids
        )
        candidate_only = sorted(
            candidate_ids - parent_ids
        )

        raise ValueError(
            "Parent and candidate evaluation results contain "
            "different scenario IDs. "
            f"Parent only: {parent_only[:20]}; "
            f"candidate only: {candidate_only[:20]}."
        )

    scenario_count = int(len(parent))

    if scenario_count == 0:
        raise ValueError(
            "Paired comparison contains no scenarios."
        )

    rng = np.random.default_rng(int(seed))
    metric_results: dict[str, dict[str, object]] = {}

    for field in PAIRED_OUTCOME_FIELDS:
        parent_values = parent[field].to_numpy(
            dtype=np.int8
        )
        candidate_values = candidate[field].to_numpy(
            dtype=np.int8
        )

        differences = (
            candidate_values - parent_values
        )

        lower, upper = _bootstrap_interval(
            differences,
            confidence_level=confidence_level,
            bootstrap_samples=parsed_bootstrap_samples,
            rng=rng,
        )

        parent_count = int(parent_values.sum())
        candidate_count = int(candidate_values.sum())

        metric_results[field] = {
            "parent_count": parent_count,
            "candidate_count": candidate_count,
            "parent_rate": (
                float(parent_count)
                / float(scenario_count)
            ),
            "candidate_rate": (
                float(candidate_count)
                / float(scenario_count)
            ),
            "rate_difference": float(
                differences.mean()
            ),
            "ci_lower": lower,
            "ci_upper": upper,
            "improved_scenarios": int(
                np.count_nonzero(differences == 1)
            ),
            "regressed_scenarios": int(
                np.count_nonzero(differences == -1)
            ),
            "unchanged_scenarios": int(
                np.count_nonzero(differences == 0)
            ),
        }

    utility_differences = (
        candidate["final_topology_utility"].to_numpy(dtype=float)
        - parent["final_topology_utility"].to_numpy(dtype=float)
    )
    utility_lower, utility_upper = _bootstrap_interval(
        utility_differences,
        confidence_level=confidence_level,
        bootstrap_samples=parsed_bootstrap_samples,
        rng=rng,
    )

    continuous_metrics: dict[str, dict[str, object]] = {
        "final_topology_utility": _continuous_metric_summary(
            parent_values=parent["final_topology_utility"].to_numpy(dtype=float),
            candidate_values=(
                candidate["final_topology_utility"].to_numpy(dtype=float)
            ),
            improvements=utility_differences,
            ci_lower=utility_lower,
            ci_upper=utility_upper,
        )
    }

    parent_J = parent["Jfinal"].to_numpy(dtype=float)
    candidate_J = candidate["Jfinal"].to_numpy(dtype=float)
    valid_J = np.isfinite(parent_J) & np.isfinite(candidate_J)
    if bool(valid_J.any()):
        J_improvements = parent_J[valid_J] - candidate_J[valid_J]
        J_lower, J_upper = _bootstrap_interval(
            J_improvements,
            confidence_level=confidence_level,
            bootstrap_samples=parsed_bootstrap_samples,
            rng=rng,
        )
        continuous_metrics["Jfinal"] = _continuous_metric_summary(
            parent_values=parent_J[valid_J],
            candidate_values=candidate_J[valid_J],
            improvements=J_improvements,
            ci_lower=J_lower,
            ci_upper=J_upper,
        )
    else:
        continuous_metrics["Jfinal"] = {
            "valid_pairs": 0,
            "parent_mean": None,
            "candidate_mean": None,
            "mean_improvement": None,
            "ci_lower": None,
            "ci_upper": None,
            "improved_scenarios": 0,
            "regressed_scenarios": 0,
            "unchanged_scenarios": 0,
        }

    return {
        "paired_comparison_version": (
            PAIRED_COMPARISON_VERSION
        ),
        "policy_mode": policy_mode,
        "scenario_count": scenario_count,
        "confidence_level": confidence_level,
        "bootstrap_samples": parsed_bootstrap_samples,
        "seed": int(seed),
        "metrics": metric_results,
        "continuous_metrics": continuous_metrics,
    }


def _load_policy_results(
    path: Path,
    *,
    policy_mode: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation results CSV not found: {path}"
        )

    frame = pd.read_csv(path)

    required_columns = {
        "scenario_id",
        "policy_mode",
        "evaluation_failed",
        "final_topology_utility",
        "Jfinal",
        *PAIRED_OUTCOME_FIELDS[1:],
    }
    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Evaluation results are missing columns in "
            f"{path}: {missing_columns}"
        )

    frame = frame.loc[
        frame["policy_mode"].astype(str)
        == policy_mode
    ].copy()

    if frame.empty:
        raise ValueError(
            f"Evaluation results contain no rows for "
            f"policy_mode={policy_mode!r}: {path}"
        )

    scenario_values = pd.to_numeric(
        frame["scenario_id"],
        errors="raise",
    )
    scenario_array = scenario_values.to_numpy(
        dtype=float
    )

    if not np.isfinite(scenario_array).all():
        raise ValueError(
            f"scenario_id contains non-finite values: "
            f"{path}"
        )

    if not np.equal(
        scenario_array,
        np.floor(scenario_array),
    ).all():
        raise ValueError(
            f"scenario_id must contain exact integers: "
            f"{path}"
        )

    frame["scenario_id"] = scenario_values.astype(
        np.int64
    )

    duplicate_rows = frame.duplicated(
        subset=["scenario_id"],
        keep=False,
    )

    if duplicate_rows.any():
        duplicate_ids = sorted(
            frame.loc[
                duplicate_rows,
                "scenario_id",
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Evaluation results contain duplicate rows "
            f"for policy_mode={policy_mode!r}: "
            f"{duplicate_ids[:20]}"
        )

    frame = frame.set_index(
        "scenario_id"
    ).sort_index()

    evaluation_failed = _coerce_bool_series(
        frame["evaluation_failed"],
        field="evaluation_failed",
        path=path,
    )

    result = pd.DataFrame(index=frame.index)
    result["evaluation_success"] = (
        ~evaluation_failed
    )

    for field in PAIRED_OUTCOME_FIELDS[1:]:
        values = _coerce_bool_series(
            frame[field],
            field=field,
            path=path,
        )

        if bool(values.loc[evaluation_failed].any()):
            raise ValueError(
                f"Failed evaluation rows must not mark "
                f"{field} as true: {path}"
            )

        result[field] = values

    utility = pd.to_numeric(
        frame["final_topology_utility"],
        errors="coerce",
    ).astype(float)
    utility.loc[evaluation_failed] = -1.0
    valid_utility = utility.loc[~evaluation_failed]
    if (
        valid_utility.isna().any()
        or not np.isfinite(valid_utility.to_numpy(dtype=float)).all()
        or bool(((valid_utility < -1.0) | (valid_utility > 1.0)).any())
    ):
        raise ValueError(
            "final_topology_utility must be finite and in [-1, 1] "
            f"for successful evaluation rows: {path}"
        )
    result["final_topology_utility"] = utility

    Jfinal = pd.to_numeric(
        frame["Jfinal"],
        errors="coerce",
    ).astype(float)
    finite_J = Jfinal[np.isfinite(Jfinal.to_numpy(dtype=float))]
    if bool((finite_J < 0.0).any()):
        raise ValueError(
            f"Jfinal must be non-negative when finite: {path}"
        )
    result["Jfinal"] = Jfinal

    return result


def _continuous_metric_summary(
    *,
    parent_values: np.ndarray,
    candidate_values: np.ndarray,
    improvements: np.ndarray,
    ci_lower: float,
    ci_upper: float,
) -> dict[str, object]:
    tolerance = 1e-12
    return {
        "valid_pairs": int(improvements.size),
        "parent_mean": float(parent_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "mean_improvement": float(improvements.mean()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "improved_scenarios": int(
            np.count_nonzero(improvements > tolerance)
        ),
        "regressed_scenarios": int(
            np.count_nonzero(improvements < -tolerance)
        ),
        "unchanged_scenarios": int(
            np.count_nonzero(np.abs(improvements) <= tolerance)
        ),
    }


def _coerce_bool_series(
    series: pd.Series,
    *,
    field: str,
    path: Path,
) -> pd.Series:
    if series.isna().any():
        raise ValueError(
            f"{field} contains missing values: {path}"
        )

    if is_bool_dtype(series.dtype):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    values = {
        "true": True,
        "1": True,
        "1.0": True,
        "false": False,
        "0": False,
        "0.0": False,
    }

    invalid = ~normalized.isin(values)

    if invalid.any():
        observed = sorted(
            normalized.loc[invalid]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{field} contains invalid boolean values "
            f"in {path}: {observed[:20]}"
        )

    return normalized.map(values).astype(bool)


def _bootstrap_interval(
    differences: np.ndarray,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if differences.ndim != 1:
        raise ValueError(
            "differences must be one-dimensional."
        )

    scenario_count = int(differences.size)

    if scenario_count == 0:
        raise ValueError(
            "differences must not be empty."
        )

    point_estimate = float(differences.mean())

    if np.all(differences == differences[0]):
        return point_estimate, point_estimate

    sample_means = np.empty(
        bootstrap_samples,
        dtype=float,
    )
    batch_size = min(64, bootstrap_samples)

    for start in range(
        0,
        bootstrap_samples,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            bootstrap_samples,
        )
        count = stop - start

        indices = rng.integers(
            0,
            scenario_count,
            size=(count, scenario_count),
        )

        sample_means[start:stop] = (
            differences[indices].mean(axis=1)
        )

    alpha = (
        1.0 - confidence_level
    ) / 2.0

    lower, upper = np.quantile(
        sample_means,
        [alpha, 1.0 - alpha],
    )

    return float(lower), float(upper)
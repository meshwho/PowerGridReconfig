from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.evaluation.paired_results import (
    PAIRED_COMPARISON_VERSION,
    PAIRED_OUTCOME_FIELDS,
    compare_evaluation_results,
)


def _row(
    scenario_id: int,
    *,
    secure: bool,
    utility: float | None = None,
    Jfinal: float | None = None,
    failed: bool = False,
    policy_mode: str = "ungated",
) -> dict[str, object]:
    if utility is None:
        utility = 1.0 if secure else 0.0
    if Jfinal is None:
        Jfinal = 0.0 if secure else 500.0

    row: dict[str, object] = {
        "scenario_id": scenario_id,
        "policy_mode": policy_mode,
        "evaluation_failed": failed,
        "final_topology_utility": (
            float("nan") if failed else float(utility)
        ),
        "Jfinal": float("nan") if failed else float(Jfinal),
    }

    for field in PAIRED_OUTCOME_FIELDS:
        if field == "evaluation_success":
            continue
        row[field] = bool(secure and not failed)

    return row


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_tied_results_have_zero_continuous_and_boolean_difference(
    tmp_path: Path,
) -> None:
    rows = [
        _row(1, secure=True),
        _row(2, secure=False, utility=0.20, Jfinal=330.0),
        _row(3, secure=True),
    ]
    parent = _write_csv(tmp_path / "parent.csv", rows)
    candidate = _write_csv(tmp_path / "candidate.csv", rows)

    comparison = compare_evaluation_results(
        parent_csv=parent,
        candidate_csv=candidate,
        policy_mode="ungated",
        bootstrap_samples=200,
        seed=7,
    )

    assert comparison["paired_comparison_version"] == PAIRED_COMPARISON_VERSION
    assert PAIRED_COMPARISON_VERSION == 2
    secure = comparison["metrics"]["physically_secure"]
    utility = comparison["continuous_metrics"]["final_topology_utility"]
    Jfinal = comparison["continuous_metrics"]["Jfinal"]
    assert secure["rate_difference"] == 0.0
    assert utility["mean_improvement"] == 0.0
    assert utility["ci_lower"] == 0.0
    assert utility["ci_upper"] == 0.0
    assert Jfinal["mean_improvement"] == 0.0
    assert utility["unchanged_scenarios"] == 3


def test_unsolved_candidate_can_improve_utility_and_J(
    tmp_path: Path,
) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [
            _row(index, secure=False, utility=-0.30, Jfinal=900.0)
            for index in range(1, 9)
        ],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [
            _row(index, secure=False, utility=0.20, Jfinal=330.0)
            for index in range(1, 9)
        ],
    )

    comparison = compare_evaluation_results(
        parent_csv=parent,
        candidate_csv=candidate,
        policy_mode="ungated",
        bootstrap_samples=200,
        seed=11,
    )

    secure = comparison["metrics"]["physically_secure"]
    utility = comparison["continuous_metrics"]["final_topology_utility"]
    Jfinal = comparison["continuous_metrics"]["Jfinal"]

    assert secure["rate_difference"] == 0.0
    assert utility["mean_improvement"] == pytest.approx(0.50)
    assert utility["ci_lower"] == pytest.approx(0.50)
    assert utility["improved_scenarios"] == 8
    assert Jfinal["mean_improvement"] == pytest.approx(570.0)
    assert Jfinal["improved_scenarios"] == 8


def test_boolean_solved_improvement_is_still_reported(
    tmp_path: Path,
) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [_row(index, secure=False) for index in range(1, 9)],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [_row(index, secure=True) for index in range(1, 9)],
    )

    comparison = compare_evaluation_results(
        parent_csv=parent,
        candidate_csv=candidate,
        policy_mode="ungated",
        bootstrap_samples=200,
        seed=11,
    )

    secure = comparison["metrics"]["physically_secure"]
    assert secure["rate_difference"] == 1.0
    assert secure["ci_lower"] == 1.0
    assert secure["improved_scenarios"] == 8


def test_failed_scenario_is_worst_utility_and_excluded_from_J_pairs(
    tmp_path: Path,
) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [
            _row(1, secure=False, utility=0.30, Jfinal=270.0),
            _row(2, secure=False, utility=0.20, Jfinal=330.0),
        ],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [
            _row(1, secure=False, utility=0.40, Jfinal=215.0),
            _row(2, secure=False, failed=True),
        ],
    )

    comparison = compare_evaluation_results(
        parent_csv=parent,
        candidate_csv=candidate,
        policy_mode="ungated",
        bootstrap_samples=200,
        seed=5,
    )

    evaluation_success = comparison["metrics"]["evaluation_success"]
    utility = comparison["continuous_metrics"]["final_topology_utility"]
    Jfinal = comparison["continuous_metrics"]["Jfinal"]

    assert comparison["scenario_count"] == 2
    assert evaluation_success["regressed_scenarios"] == 1
    assert utility["valid_pairs"] == 2
    assert utility["regressed_scenarios"] == 1
    assert Jfinal["valid_pairs"] == 1
    assert Jfinal["mean_improvement"] == pytest.approx(55.0)


def test_comparison_requires_same_scenarios(tmp_path: Path) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [_row(1, secure=True), _row(2, secure=False)],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [_row(1, secure=True), _row(3, secure=False)],
    )

    with pytest.raises(ValueError, match="different scenario IDs"):
        compare_evaluation_results(
            parent_csv=parent,
            candidate_csv=candidate,
            policy_mode="ungated",
        )


def test_comparison_rejects_duplicate_scenario_rows(tmp_path: Path) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [_row(1, secure=True), _row(1, secure=False)],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [_row(1, secure=True)],
    )

    with pytest.raises(ValueError, match="duplicate rows"):
        compare_evaluation_results(
            parent_csv=parent,
            candidate_csv=candidate,
            policy_mode="ungated",
        )


def test_successful_rows_require_bounded_utility(tmp_path: Path) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [_row(1, secure=False, utility=0.0)],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [_row(1, secure=False, utility=1.2)],
    )

    with pytest.raises(ValueError, match="final_topology_utility"):
        compare_evaluation_results(
            parent_csv=parent,
            candidate_csv=candidate,
            policy_mode="ungated",
        )


def test_negative_finite_J_is_rejected(tmp_path: Path) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [_row(1, secure=False, Jfinal=10.0)],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [_row(1, secure=False, Jfinal=-1.0)],
    )

    with pytest.raises(ValueError, match="Jfinal"):
        compare_evaluation_results(
            parent_csv=parent,
            candidate_csv=candidate,
            policy_mode="ungated",
        )


def test_bootstrap_is_reproducible(tmp_path: Path) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [
            _row(1, secure=False, utility=-0.2, Jfinal=750.0),
            _row(2, secure=True),
            _row(3, secure=False, utility=0.0, Jfinal=500.0),
            _row(4, secure=True),
        ],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [
            _row(1, secure=False, utility=0.1, Jfinal=410.0),
            _row(2, secure=True),
            _row(3, secure=False, utility=-0.1, Jfinal=610.0),
            _row(4, secure=True),
        ],
    )

    first = compare_evaluation_results(
        parent_csv=parent,
        candidate_csv=candidate,
        policy_mode="ungated",
        bootstrap_samples=300,
        seed=17,
    )
    second = compare_evaluation_results(
        parent_csv=parent,
        candidate_csv=candidate,
        policy_mode="ungated",
        bootstrap_samples=300,
        seed=17,
    )

    assert first == second

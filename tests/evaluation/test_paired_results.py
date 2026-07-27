from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.evaluation.paired_results import (
    PAIRED_OUTCOME_FIELDS,
    compare_evaluation_results,
)


def _row(
    scenario_id: int,
    *,
    secure: bool,
    failed: bool = False,
    policy_mode: str = "ungated",
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario_id": scenario_id,
        "policy_mode": policy_mode,
        "evaluation_failed": failed,
    }

    for field in PAIRED_OUTCOME_FIELDS:
        if field == "evaluation_success":
            continue

        row[field] = bool(
            secure and not failed
        )

    return row


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )
    return path


def test_tied_results_have_zero_difference(
    tmp_path: Path,
) -> None:
    rows = [
        _row(1, secure=True),
        _row(2, secure=False),
        _row(3, secure=True),
    ]
    parent = _write_csv(
        tmp_path / "parent.csv",
        rows,
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        rows,
    )

    comparison = compare_evaluation_results(
        parent_csv=parent,
        candidate_csv=candidate,
        policy_mode="ungated",
        bootstrap_samples=200,
        seed=7,
    )

    secure = comparison["metrics"][
        "physically_secure"
    ]

    assert secure["rate_difference"] == 0.0
    assert secure["ci_lower"] == 0.0
    assert secure["ci_upper"] == 0.0
    assert secure["unchanged_scenarios"] == 3


def test_clear_improvement_has_positive_interval(
    tmp_path: Path,
) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [
            _row(index, secure=False)
            for index in range(1, 9)
        ],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [
            _row(index, secure=True)
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

    secure = comparison["metrics"][
        "physically_secure"
    ]

    assert secure["rate_difference"] == 1.0
    assert secure["ci_lower"] == 1.0
    assert secure["ci_upper"] == 1.0
    assert secure["improved_scenarios"] == 8
    assert secure["regressed_scenarios"] == 0


def test_failed_scenario_remains_in_comparison(
    tmp_path: Path,
) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [
            _row(1, secure=True),
            _row(2, secure=True),
        ],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [
            _row(1, secure=True),
            _row(
                2,
                secure=False,
                failed=True,
            ),
        ],
    )

    comparison = compare_evaluation_results(
        parent_csv=parent,
        candidate_csv=candidate,
        policy_mode="ungated",
        bootstrap_samples=200,
        seed=5,
    )

    evaluation_success = comparison["metrics"][
        "evaluation_success"
    ]
    physically_secure = comparison["metrics"][
        "physically_secure"
    ]

    assert comparison["scenario_count"] == 2
    assert (
        evaluation_success["regressed_scenarios"]
        == 1
    )
    assert (
        physically_secure["regressed_scenarios"]
        == 1
    )


def test_comparison_requires_same_scenarios(
    tmp_path: Path,
) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [
            _row(1, secure=True),
            _row(2, secure=False),
        ],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [
            _row(1, secure=True),
            _row(3, secure=False),
        ],
    )

    with pytest.raises(
        ValueError,
        match="different scenario IDs",
    ):
        compare_evaluation_results(
            parent_csv=parent,
            candidate_csv=candidate,
            policy_mode="ungated",
        )


def test_comparison_rejects_duplicate_scenario_rows(
    tmp_path: Path,
) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [
            _row(1, secure=True),
            _row(1, secure=False),
        ],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [
            _row(1, secure=True),
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate rows",
    ):
        compare_evaluation_results(
            parent_csv=parent,
            candidate_csv=candidate,
            policy_mode="ungated",
        )


def test_bootstrap_is_reproducible(
    tmp_path: Path,
) -> None:
    parent = _write_csv(
        tmp_path / "parent.csv",
        [
            _row(1, secure=False),
            _row(2, secure=True),
            _row(3, secure=False),
            _row(4, secure=True),
        ],
    )
    candidate = _write_csv(
        tmp_path / "candidate.csv",
        [
            _row(1, secure=True),
            _row(2, secure=True),
            _row(3, secure=False),
            _row(4, secure=False),
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
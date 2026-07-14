from __future__ import annotations

from pathlib import Path

from grid_topology_ai.self_play.learning_curve import (
    LearningCurveRow,
    load_learning_curve,
    save_learning_curve,
    upsert_iteration_row,
)


def test_missing_learning_curve_loads_as_empty(tmp_path: Path) -> None:
    assert load_learning_curve(tmp_path / "learning_curve.csv") == []


def test_zero_byte_learning_curve_loads_as_empty(
    tmp_path: Path,
) -> None:
    path = tmp_path / "learning_curve.csv"
    path.write_text("", encoding="utf-8")

    assert load_learning_curve(path) == []


def test_header_only_learning_curve_loads_as_empty(
    tmp_path: Path,
) -> None:
    path = tmp_path / "learning_curve.csv"
    path.write_text("iteration,status\n", encoding="utf-8")

    assert load_learning_curve(path) == []


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "learning_curve.csv"
    rows: list[LearningCurveRow] = [
        {
            "iteration": 1,
            "accepted": True,
            "status": "ACCEPTED",
            "candidate_metric": 0.8,
            "candidate_solve_rate": 0.8,
        },
        {
            "iteration": 2,
            "accepted": False,
            "status": "REJECTED",
            "candidate_metric": 0.7,
            "candidate_solve_rate": 0.7,
        },
    ]

    save_learning_curve(rows=rows, path=path)

    loaded = load_learning_curve(path)
    assert len(loaded) == 2
    assert [int(row["iteration"]) for row in loaded] == [1, 2]
    assert [row["status"] for row in loaded] == [
        "ACCEPTED",
        "REJECTED",
    ]
    assert float(loaded[0]["candidate_metric"]) == 0.8
    assert float(loaded[1]["candidate_solve_rate"]) == 0.7


def test_save_preserves_preferred_column_order(tmp_path: Path) -> None:
    path = tmp_path / "learning_curve.csv"
    save_learning_curve(
        rows=[
            {
                "custom_metric": 3.14,
                "status": "ACCEPTED",
                "best_metric_after": 0.9,
                "candidate_metric": 0.8,
                "accepted": True,
                "iteration": 1,
            }
        ],
        path=path,
    )

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:5] == [
        "iteration",
        "accepted",
        "status",
        "candidate_metric",
        "best_metric_after",
    ]
    assert header.split(",")[-1] == "custom_metric"


def test_upsert_appends_new_iteration() -> None:
    rows: list[LearningCurveRow] = [
        {"iteration": 1, "status": "ACCEPTED"},
    ]

    updated = upsert_iteration_row(
        rows=rows,
        row={"iteration": 2, "status": "REJECTED"},
    )

    assert rows == [{"iteration": 1, "status": "ACCEPTED"}]
    assert [row["iteration"] for row in updated] == [1, 2]


def test_upsert_replaces_existing_iteration() -> None:
    rows: list[LearningCurveRow] = [
        {"iteration": 1, "status": "OLD"},
        {"iteration": 2, "status": "KEPT"},
    ]

    updated = upsert_iteration_row(
        rows=rows,
        row={"iteration": 1, "status": "NEW"},
    )

    assert [row["iteration"] for row in updated] == [2, 1]
    assert [row["status"] for row in updated] == ["KEPT", "NEW"]
    assert rows[0]["status"] == "OLD"


def test_save_empty_curve_can_be_loaded(tmp_path: Path) -> None:
    path = tmp_path / "learning_curve.csv"

    save_learning_curve(rows=[], path=path)

    assert load_learning_curve(path) == []


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "learning_curve.csv"

    save_learning_curve(
        rows=[{"iteration": 1, "status": "ACCEPTED"}],
        path=path,
    )

    assert path.is_file()

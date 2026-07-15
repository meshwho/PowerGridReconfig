from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

LearningCurveRow = dict[str, object]

_PREFERRED_COLUMNS = [
    "iteration",
    "accepted",
    "status",
    "candidate_metric",
    "best_metric_after",
    "n_sampled_scenarios",
    "n_raw_examples",
    "n_train_examples",
    "n_fresh",
    "n_old",
    "candidate_checkpoint",
    "best_checkpoint_after",
]


def _fieldnames(rows: list[LearningCurveRow]) -> list[str]:
    fieldnames: list[str] = []

    for key in _PREFERRED_COLUMNS:
        if any(key in row for row in rows):
            fieldnames.append(key)

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    return fieldnames


def load_learning_curve(path: Path) -> list[LearningCurveRow]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    df = pd.read_csv(path)

    if df.empty:
        return []

    return df.to_dict(orient="records")


def save_learning_curve(
    *,
    rows: list[LearningCurveRow],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=_fieldnames(rows),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def upsert_iteration_row(
    *,
    rows: list[LearningCurveRow],
    row: LearningCurveRow,
) -> list[LearningCurveRow]:
    iteration = int(row["iteration"])

    updated = [
        item
        for item in rows
        if int(item.get("iteration", -1)) != iteration
    ]
    updated.append(row)

    return updated

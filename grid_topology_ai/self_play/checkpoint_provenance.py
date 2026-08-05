from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
)

CHECKPOINT_SELECTION_REPORT = Path(
    "checkpoint_selection/checkpoint_selection.json"
)
CHECKPOINT_SELECTION_HASH_KEY = "checkpoint_selection_sha256"


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _summary(report: Mapping[str, object]) -> dict[str, object]:
    candidates = report.get("candidates")
    return {
        "selection_method": report.get("selection_method"),
        "metric": report.get("metric"),
        "metric_direction": report.get("metric_direction"),
        "selected_source_checkpoint": report.get(
            "selected_source_checkpoint"
        ),
        "selected_archived_checkpoint": report.get(
            "selected_archived_checkpoint"
        ),
        "selected_metric_value": report.get("selected_metric_value"),
        "candidate_count": (
            len(candidates) if isinstance(candidates, list) else 0
        ),
    }


def _validate_report(
    *,
    metadata_path: Path,
    report_path: Path,
    expected_hash: str | None = None,
) -> dict[str, object]:
    _require_file(report_path, label="checkpoint selection report")
    actual_hash = sha256_file(report_path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(
            f"Corrupt checkpoint selection report: {report_path}"
        )

    report = load_json(report_path)
    if report.get("selection_method") != "closed_loop_tuning_arena":
        raise ValueError(
            f"Invalid checkpoint selection method: {report_path}"
        )

    metadata = load_json(metadata_path)
    candidate_checkpoint = Path(
        str(metadata.get("candidate_checkpoint", ""))
    )
    selected_checkpoint = Path(
        str(report.get("selected_checkpoint", ""))
    )
    if candidate_checkpoint.resolve() != selected_checkpoint.resolve():
        raise ValueError(
            "Checkpoint selection report does not reference the iteration "
            f"candidate: {report_path}"
        )
    _require_file(
        candidate_checkpoint,
        label="selected candidate checkpoint",
    )
    if sha256_file(candidate_checkpoint) != report.get(
        "selected_checkpoint_sha256"
    ):
        raise ValueError(
            "Selected checkpoint does not match arena report: "
            f"{candidate_checkpoint}"
        )

    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            f"Checkpoint selection report has no candidates: {report_path}"
        )
    for item in candidates:
        if not isinstance(item, Mapping):
            raise ValueError(
                "Checkpoint selection candidate must be an object: "
                f"{report_path}"
            )
        archived = Path(str(item.get("archived_checkpoint", "")))
        _require_file(
            archived,
            label="archived checkpoint candidate",
        )
        if sha256_file(archived) != item.get("checkpoint_sha256"):
            raise ValueError(
                f"Corrupt archived checkpoint candidate: {archived}"
            )

    return report


def _update_learning_curve(
    *,
    path: Path,
    iteration: int,
    report_path: Path,
    report: Mapping[str, object],
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())

    for name in (
        "checkpoint_arena_metric",
        "checkpoint_arena_metric_value",
        "checkpoint_arena_candidate_count",
        "checkpoint_selection_report",
        CHECKPOINT_SELECTION_HASH_KEY,
    ):
        if name not in fieldnames:
            fieldnames.append(name)

    report_hash = sha256_file(report_path)
    updated = False
    for row in rows:
        try:
            row_iteration = int(row.get("iteration", ""))
        except ValueError:
            continue
        if row_iteration != int(iteration):
            continue

        candidates = report.get("candidates")
        row["checkpoint_selection_metric"] = "closed_loop_arena"
        row["checkpoint_arena_metric"] = str(
            report.get("metric", "")
        )
        row["checkpoint_arena_metric_value"] = str(
            report.get("selected_metric_value", "")
        )
        row["checkpoint_arena_candidate_count"] = str(
            len(candidates) if isinstance(candidates, list) else 0
        )
        row["checkpoint_selection_report"] = str(report_path)
        row[CHECKPOINT_SELECTION_HASH_KEY] = report_hash
        updated = True
        break

    if not updated:
        raise ValueError(
            f"learning_curve.csv is missing iteration {iteration}: {path}"
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def attach_checkpoint_selection_provenance(
    *,
    metadata_path: Path,
    learning_curve_path: Path,
    iteration: int,
) -> Path | None:
    report_path = metadata_path.parent / CHECKPOINT_SELECTION_REPORT
    if not report_path.is_file():
        return None

    report = _validate_report(
        metadata_path=metadata_path,
        report_path=report_path,
    )
    report_hash = sha256_file(report_path)
    metadata = load_json(metadata_path)
    hashes = metadata.setdefault("hashes", {})
    extra = metadata.setdefault("extra", {})
    if not isinstance(hashes, dict) or not isinstance(extra, dict):
        raise ValueError(
            "metadata.json has invalid provenance containers: "
            f"{metadata_path}"
        )

    hashes[CHECKPOINT_SELECTION_HASH_KEY] = report_hash
    extra[CHECKPOINT_SELECTION_HASH_KEY] = report_hash
    extra["checkpoint_selection_path"] = str(report_path)
    extra["checkpoint_selection"] = _summary(report)
    save_json(metadata, metadata_path)

    _update_learning_curve(
        path=learning_curve_path,
        iteration=iteration,
        report_path=report_path,
        report=report,
    )
    return report_path


def validate_checkpoint_selection_provenance(
    metadata_path: Path,
) -> Path | None:
    metadata = load_json(metadata_path)
    report_path = metadata_path.parent / CHECKPOINT_SELECTION_REPORT
    hashes = metadata.get("hashes")
    extra = metadata.get("extra")
    has_fields = (
        isinstance(hashes, Mapping)
        and CHECKPOINT_SELECTION_HASH_KEY in hashes
    ) or (
        isinstance(extra, Mapping)
        and any(
            key in extra
            for key in (
                CHECKPOINT_SELECTION_HASH_KEY,
                "checkpoint_selection_path",
                "checkpoint_selection",
            )
        )
    )
    if not report_path.exists() and not has_fields:
        return None
    if not isinstance(hashes, Mapping) or not isinstance(extra, Mapping):
        raise ValueError(
            "metadata.json is missing checkpoint selection provenance: "
            f"{metadata_path}"
        )

    expected_hash = hashes.get(CHECKPOINT_SELECTION_HASH_KEY)
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "metadata.json is missing checkpoint selection hash: "
            f"{metadata_path}"
        )
    if extra.get(CHECKPOINT_SELECTION_HASH_KEY) != expected_hash:
        raise ValueError(
            "metadata.json checkpoint selection hashes disagree: "
            f"{metadata_path}"
        )

    stored_path = extra.get("checkpoint_selection_path")
    if (
        not isinstance(stored_path, str)
        or Path(stored_path).name != report_path.name
    ):
        raise ValueError(
            "metadata.json has invalid checkpoint selection path: "
            f"{metadata_path}"
        )

    report = _validate_report(
        metadata_path=metadata_path,
        report_path=report_path,
        expected_hash=expected_hash,
    )
    if extra.get("checkpoint_selection") != _summary(report):
        raise ValueError(
            "metadata.json checkpoint selection summary is stale: "
            f"{metadata_path}"
        )
    return report_path

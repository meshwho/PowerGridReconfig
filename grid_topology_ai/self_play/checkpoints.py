from __future__ import annotations

import csv
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

from grid_topology_ai.config.checkpoint_selection import (
    CheckpointSelectionConfig,
)
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.evaluation.checkpoint import load_scenario_ids
from grid_topology_ai.evaluation.policy_comparison import (
    PolicyMode,
    require_policy_mode_metrics,
    require_primary_policy_mode as require_evaluation_primary_policy_mode,
)
from grid_topology_ai.self_play.acceptance import (
    require_metrics_semantic_versions,
)
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
)
from grid_topology_ai.self_play.provenance import (
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
)
from grid_topology_ai.training.checkpoints import (
    load_checkpoint_payload,
    make_json_safe,
)

if TYPE_CHECKING:
    from grid_topology_ai.self_play.paths import SelfPlayPaths


_PRIMARY_POLICY_MODE = PolicyMode.UNGATED.value
CHECKPOINT_SELECTION_REPORT = Path(
    "checkpoint_selection/checkpoint_selection.json"
)
CHECKPOINT_SELECTION_HASH_KEY = "checkpoint_selection_sha256"
_ARENA_SCHEMA_VERSION = 3
_RANKING_METRICS = (
    ("validation_loss", "loss"),
    ("validation_policy_loss", "policy_loss"),
    ("validation_value_loss", "value_loss"),
    ("validation_value_calibration_error", "value_calibration_error"),
)


@dataclass(frozen=True, slots=True)
class BestState:
    checkpoint: Path
    metrics: dict[str, object]


@dataclass(frozen=True, slots=True)
class CheckpointArenaResult:
    checkpoint: Path
    report_path: Path
    selected_source: Path
    metric_name: str
    metric_value: float
    candidate_count: int


def _require_primary_policy_mode(
    metrics: Mapping[str, object],
    *,
    source: str,
) -> None:
    require_evaluation_primary_policy_mode(
        metrics,
        PolicyMode.UNGATED,
        source=source,
    )

    task_config = metrics.get("task_config")
    configured_mode = (
        task_config.get("primary_policy_mode")
        if isinstance(task_config, Mapping)
        else None
    )

    if configured_mode != _PRIMARY_POLICY_MODE:
        raise ValueError(
            "Incompatible evaluation primary policy mode for "
            f"{source}: expected {_PRIMARY_POLICY_MODE!r}, "
            f"observed task_config={configured_mode!r}. Regenerate fixed "
            "evaluation metrics with the current ungated policy contract."
        )


def initialize_best_state(
    *,
    paths: SelfPlayPaths,
) -> BestState:
    paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.best_metrics.parent.mkdir(parents=True, exist_ok=True)

    if not paths.best_checkpoint.exists():
        load_checkpoint_payload(paths.bootstrap_checkpoint, map_location="cpu")
        print("Initializing self-play best checkpoint from bootstrap.")
        print(f"Bootstrap checkpoint: {paths.bootstrap_checkpoint}")
        print(f"Best checkpoint:      {paths.best_checkpoint}")
        shutil.copy2(paths.bootstrap_checkpoint, paths.best_checkpoint)

    load_checkpoint_payload(paths.best_checkpoint, map_location="cpu")

    if not paths.best_metrics.exists():
        bootstrap_metrics = load_json(paths.bootstrap_metrics)
        require_metrics_semantic_versions(
            bootstrap_metrics,
            source=str(paths.bootstrap_metrics),
        )
        _require_primary_policy_mode(
            bootstrap_metrics,
            source=str(paths.bootstrap_metrics),
        )
        print("Initializing self-play best metrics from bootstrap.")
        print(f"Bootstrap metrics: {paths.bootstrap_metrics}")
        print(f"Best metrics:      {paths.best_metrics}")
        shutil.copy2(paths.bootstrap_metrics, paths.best_metrics)

    best_metrics = load_json(paths.best_metrics)
    require_metrics_semantic_versions(
        best_metrics,
        source=str(paths.best_metrics),
    )
    _require_primary_policy_mode(
        best_metrics,
        source=str(paths.best_metrics),
    )

    return BestState(
        checkpoint=paths.best_checkpoint,
        metrics=best_metrics,
    )


def promote_candidate(
    *,
    candidate_checkpoint: Path,
    candidate_metrics: Mapping[str, object],
    paths: SelfPlayPaths,
) -> BestState:
    if not candidate_checkpoint.is_file():
        raise FileNotFoundError(
            f"Candidate checkpoint not found: {candidate_checkpoint}"
        )

    load_checkpoint_payload(candidate_checkpoint, map_location="cpu")
    require_metrics_semantic_versions(
        candidate_metrics,
        source="candidate metrics",
    )
    _require_primary_policy_mode(
        candidate_metrics,
        source="candidate metrics",
    )

    paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.best_metrics.parent.mkdir(parents=True, exist_ok=True)

    metrics = dict(candidate_metrics)
    shutil.copy2(candidate_checkpoint, paths.best_checkpoint)
    save_json(metrics, paths.best_metrics)

    return BestState(
        checkpoint=paths.best_checkpoint,
        metrics=metrics,
    )


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _checkpoint_selection_required(
    metadata: Mapping[str, object],
) -> bool:
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        return False
    selection = config.get("checkpoint_selection")
    if not isinstance(selection, Mapping):
        return False
    enabled = selection.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "metadata.json checkpoint_selection.enabled must be a boolean."
        )
    return enabled


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
    metadata = load_json(metadata_path)
    if not report_path.is_file():
        if _checkpoint_selection_required(metadata):
            _require_file(
                report_path,
                label="checkpoint selection report",
            )
        return None

    report = _validate_report(
        metadata_path=metadata_path,
        report_path=report_path,
    )
    report_hash = sha256_file(report_path)
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
        if _checkpoint_selection_required(metadata):
            _require_file(
                report_path,
                label="checkpoint selection report",
            )
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


def _finite_metric(
    metrics: Mapping[str, object],
    name: str,
    *,
    source: str,
) -> float:
    if name not in metrics:
        raise ValueError(
            f"Checkpoint arena metric {name!r} is missing: {source}"
        )
    try:
        value = float(metrics[name])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Checkpoint arena metric {name!r} is not numeric: {source}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Checkpoint arena metric {name!r} is not finite: {source}"
        )
    return value


def _candidate_paths(canonical_checkpoint: Path) -> tuple[Path, ...]:
    variants = sorted(
        canonical_checkpoint.parent.glob(
            f"{canonical_checkpoint.stem}_*{canonical_checkpoint.suffix}"
        )
    )
    paths = [canonical_checkpoint, *variants]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint selection candidate not found: {path}"
            )
        seen.add(resolved)
        unique.append(path)
    return tuple(unique)


def _load_candidates(
    *,
    canonical_checkpoint: Path,
    physics_config: PhysicsConfig,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in _candidate_paths(canonical_checkpoint):
        payload = load_checkpoint_payload(
            path,
            map_location="cpu",
            expected_physics_config=physics_config,
        )
        raw_metrics = payload.get("val_metrics")
        val_metrics = (
            dict(raw_metrics)
            if isinstance(raw_metrics, Mapping)
            else {}
        )
        candidates.append(
            {
                "path": path,
                "sha256": sha256_file(path),
                "payload": payload,
                "val_metrics": val_metrics,
                "training_selector": payload.get(
                    "checkpoint_selection_metric"
                ),
                "saved_epoch": int(
                    payload.get(
                        "saved_epoch",
                        payload.get("best_epoch", 0),
                    )
                ),
                "ranking_sources": [],
            }
        )
    return candidates


def _select_candidate_pool(
    candidates: Sequence[dict[str, Any]],
    config: CheckpointSelectionConfig,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()

    for ranking_name, metric_name in _RANKING_METRICS:
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in candidates:
            metrics = candidate["val_metrics"]
            if metric_name not in metrics:
                continue
            try:
                value = float(metrics[metric_name])
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(value):
                continue
            ranked.append((value, str(candidate["path"]), candidate))

        ranked.sort(key=lambda item: (item[0], item[1]))
        for _, _, candidate in ranked[: config.candidates_per_metric]:
            candidate["ranking_sources"].append(ranking_name)
            digest = str(candidate["sha256"])
            if digest in selected_hashes:
                continue
            selected.append(candidate)
            selected_hashes.add(digest)
            if len(selected) >= config.max_candidates:
                return selected

    if not selected:
        canonical = candidates[0]
        canonical["ranking_sources"].append("canonical_fallback")
        selected.append(canonical)

    return selected[: config.max_candidates]


def _archive_candidates(
    candidates: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    archive_dir = output_dir / "candidates"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        source = Path(candidate["path"])
        destination = archive_dir / f"candidate_{index:02d}_{source.name}"
        if destination.exists():
            raise FileExistsError(
                f"Checkpoint arena candidate archive already exists: {destination}"
            )
        shutil.copy2(source, destination)
        digest = sha256_file(destination)
        if digest != str(candidate["sha256"]):
            raise RuntimeError(
                "Checkpoint arena candidate changed while being archived: "
                f"{source}"
            )
        archived.append(
            {
                **candidate,
                "source_path": source,
                "archived_path": destination,
            }
        )

    return archived


def _scenario_ids(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint selection transitions CSV not found: {path}"
        )
    return {
        int(value)
        for value in load_scenario_ids(path, limit=None)
    }


def _lineage_fingerprints(path: Path) -> set[str] | None:
    frame = pd.read_csv(path)
    if PHYSICAL_LINEAGE_FINGERPRINT_FIELD not in frame.columns:
        return None
    values = {
        str(value).strip().lower()
        for value in frame[PHYSICAL_LINEAGE_FINGERPRINT_FIELD].tolist()
        if str(value).strip()
    }
    return values or None


def _validate_tuning_independence(
    *,
    tuning_csv: Path,
    excluded_csvs: Mapping[str, Path],
) -> tuple[int, ...]:
    tuning_ids = _scenario_ids(tuning_csv)
    if not tuning_ids:
        raise ValueError(
            f"Checkpoint tuning set contains no scenarios: {tuning_csv}"
        )

    tuning_resolved = tuning_csv.resolve()
    tuning_sha256 = sha256_file(tuning_csv)
    tuning_lineages = _lineage_fingerprints(tuning_csv)
    for role, path in excluded_csvs.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint selection comparison CSV not found: {path}"
            )
        if tuning_resolved == path.resolve():
            raise ValueError(
                "Checkpoint tuning CSV must be independent from "
                f"{role}: {tuning_csv}"
            )
        if tuning_sha256 == sha256_file(path):
            raise ValueError(
                "Checkpoint tuning CSV duplicates "
                f"{role}: {tuning_csv}"
            )

        overlap_ids = tuning_ids & _scenario_ids(path)
        if overlap_ids:
            preview = sorted(overlap_ids)[:20]
            raise ValueError(
                "Checkpoint tuning scenario-ID leakage detected "
                f"against {role}: count={len(overlap_ids)}, "
                f"examples={preview}."
            )

        other_lineages = _lineage_fingerprints(path)
        if tuning_lineages is None or other_lineages is None:
            continue
        overlap = tuning_lineages & other_lineages
        if overlap:
            preview = sorted(overlap)[:5]
            raise ValueError(
                "Checkpoint tuning physical-lineage leakage detected "
                f"against {role}: count={len(overlap)}, "
                f"examples={preview}."
            )

    return tuple(sorted(tuning_ids))


def _annotate_selected_checkpoint(
    *,
    source: Path,
    destination: Path,
    metric_name: str,
    metric_value: float,
    report_path: Path,
    candidate_count: int,
    physics_config: PhysicsConfig,
) -> None:
    payload = load_checkpoint_payload(
        source,
        map_location="cpu",
        expected_physics_config=physics_config,
    )
    training_selector = payload.get("checkpoint_selection_metric")
    payload["training_checkpoint_selection_metric"] = training_selector
    payload["checkpoint_selection_metric"] = "closed_loop_arena"
    payload["checkpoint_arena_policy_mode"] = PolicyMode.UNGATED.value
    payload["checkpoint_arena_metric"] = metric_name
    payload["checkpoint_arena_metric_value"] = float(metric_value)
    payload["checkpoint_arena_source_checkpoint"] = str(source)
    payload["checkpoint_arena_report"] = str(report_path)
    payload["checkpoint_arena_candidate_count"] = int(candidate_count)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def _arena_sort_value(
    value: float,
    direction: str,
) -> float:
    return -value if direction == "maximize" else value


def select_checkpoint_in_tuning_arena(
    *,
    canonical_checkpoint: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
    config: CheckpointSelectionConfig,
    physics_config: PhysicsConfig,
    tuning_csv: str | Path,
    tuning_raw_dir: str | Path,
    excluded_csvs: Mapping[str, Path],
    evaluate: Callable[..., dict[str, Any]],
) -> CheckpointArenaResult:
    if not config.enabled:
        raise ValueError(
            "Checkpoint tuning arena requires checkpoint_selection.enabled=true."
        )

    canonical = Path(canonical_checkpoint)
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    tuning_csv = Path(tuning_csv)
    tuning_raw_dir = Path(tuning_raw_dir)
    if not tuning_raw_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint tuning raw directory not found: {tuning_raw_dir}"
        )

    scenario_ids = _validate_tuning_independence(
        tuning_csv=tuning_csv,
        excluded_csvs=excluded_csvs,
    )
    loaded = _load_candidates(
        canonical_checkpoint=canonical,
        physics_config=physics_config,
    )
    candidates = _select_candidate_pool(loaded, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _archive_candidates(candidates, output_dir)
    evaluated: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        checkpoint = Path(candidate["archived_path"])
        source_checkpoint = Path(candidate["source_path"])
        candidate_dir = output_dir / (
            f"candidate_{index:02d}_{source_checkpoint.stem}"
        )
        metrics = evaluate(
            project_root=project_root,
            checkpoint=checkpoint,
            eval_csv=tuning_csv,
            eval_raw_dir=tuning_raw_dir,
            output_dir=candidate_dir,
            config=config.arena,
            physics_config=physics_config,
            scenario_ids=scenario_ids,
        )
        require_evaluation_primary_policy_mode(
            metrics,
            PolicyMode.UNGATED,
            source=str(candidate_dir),
        )
        ungated_metrics = require_policy_mode_metrics(
            metrics,
            PolicyMode.UNGATED,
            source=str(candidate_dir),
        )
        metric_value = _finite_metric(
            ungated_metrics,
            config.metric,
            source=str(candidate_dir),
        )
        failed_scenarios = int(
            ungated_metrics.get("failed_scenarios", 0)
        )
        evaluated.append(
            {
                **candidate,
                "arena_metrics": dict(metrics),
                "arena_metric_value": metric_value,
                "failed_scenarios": failed_scenarios,
                "output_dir": candidate_dir,
            }
        )

    evaluated.sort(
        key=lambda item: (
            _arena_sort_value(
                float(item["arena_metric_value"]),
                config.metric_direction,
            ),
            int(item["failed_scenarios"]),
            str(item["source_path"]),
        )
    )
    winner = evaluated[0]
    report_path = output_dir / "checkpoint_selection.json"
    selected_source = Path(winner["source_path"])
    selected_archive = Path(winner["archived_path"])

    _annotate_selected_checkpoint(
        source=selected_archive,
        destination=canonical,
        metric_name=config.metric,
        metric_value=float(winner["arena_metric_value"]),
        report_path=report_path,
        candidate_count=len(evaluated),
        physics_config=physics_config,
    )

    report = {
        "schema_version": _ARENA_SCHEMA_VERSION,
        "selection_method": "closed_loop_tuning_arena",
        "policy_mode": PolicyMode.UNGATED.value,
        "metric": config.metric,
        "metric_direction": config.metric_direction,
        "tuning_csv": str(tuning_csv),
        "tuning_csv_sha256": sha256_file(tuning_csv),
        "tuning_raw_dir": str(tuning_raw_dir),
        "tuning_scenario_count": len(scenario_ids),
        "tuning_scenario_ids": list(scenario_ids),
        "candidates_per_metric": config.candidates_per_metric,
        "max_candidates": config.max_candidates,
        "arena_config": make_json_safe(asdict(config.arena)),
        "selected_source_checkpoint": str(selected_source),
        "selected_archived_checkpoint": str(selected_archive),
        "selected_checkpoint": str(canonical),
        "selected_checkpoint_sha256": sha256_file(canonical),
        "selected_metric_value": float(winner["arena_metric_value"]),
        "candidates": [
            {
                "source_checkpoint": str(item["source_path"]),
                "archived_checkpoint": str(item["archived_path"]),
                "checkpoint_sha256": str(item["sha256"]),
                "saved_epoch": int(item["saved_epoch"]),
                "training_selector": item["training_selector"],
                "ranking_sources": list(item["ranking_sources"]),
                "validation_metrics": make_json_safe(
                    item["val_metrics"]
                ),
                "arena_metric_value": float(
                    item["arena_metric_value"]
                ),
                "failed_scenarios": int(item["failed_scenarios"]),
                "arena_output_dir": str(item["output_dir"]),
                "arena_metrics": make_json_safe(
                    item["arena_metrics"]
                ),
            }
            for item in evaluated
        ],
    }
    save_json(report, report_path)

    return CheckpointArenaResult(
        checkpoint=canonical,
        report_path=report_path,
        selected_source=selected_source,
        metric_name=config.metric,
        metric_value=float(winner["arena_metric_value"]),
        candidate_count=len(evaluated),
    )

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import (
    physics_provenance,
    require_exact_contract_version,
    require_physics_provenance,
)
from grid_topology_ai.self_play.lineage_artifacts import (
    validate_lineage_columns,
)
from grid_topology_ai.self_play.physical_lineage import (
    PHYSICAL_LINEAGE_CONTRACT_VERSION,
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
    PhysicalLineage,
    require_one_lineage_per_scenario,
    require_physical_lineage,
)

PHYSICAL_SPLIT_MANIFEST_SCHEMA_VERSION = 1
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
_VALID_SPLITS = {TRAIN_SPLIT, VALIDATION_SPLIT}


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _require_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be in (0, 1).")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be in (0, 1).") from exc
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")
    return number


def _exact_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
        raise ValueError(f"{name} must be an integer.")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be an integer.")
    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer.")
    canonical = str(int(number))
    if text not in {canonical, f"{canonical}.0"}:
        raise ValueError(f"{name} must be an unambiguous integer.")
    return int(number)


def _require_positive_integer(value: object, *, name: str) -> int:
    number = _exact_integer(value, name=name)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return number


def _require_non_negative_integer(value: object, *, name: str) -> int:
    number = _exact_integer(value, name=name)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return number


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef"
        for character in text
    ):
        raise ValueError(f"{name} must be a SHA-256 digest.")
    return text


def _assignment_rank(seed: int, fingerprint: str) -> str:
    payload = f"{int(seed)}:{fingerprint}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lineage_summary(
    frame: pd.DataFrame,
    *,
    source: str | Path,
) -> dict[str, dict[str, Any]]:
    validate_lineage_columns(frame, source=source)
    if "scenario_id" not in frame.columns:
        raise ValueError(f"{source} is missing scenario_id.")
    require_one_lineage_per_scenario(
        frame.to_dict(orient="records"),
        source=str(source),
    )

    normalized = frame.copy()
    normalized[PHYSICAL_LINEAGE_FINGERPRINT_FIELD] = (
        normalized[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    summaries: dict[str, dict[str, Any]] = {}
    for fingerprint, group in normalized.groupby(
        PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
        sort=False,
    ):
        first = group.iloc[0].to_dict()
        lineage = require_physical_lineage(
            first,
            source=f"{source} lineage {fingerprint}",
        )
        scenarios = sorted(
            {
                _require_non_negative_integer(
                    value,
                    name="scenario_id",
                )
                for value in group["scenario_id"].tolist()
            }
        )
        difficulties = {
            str(value).strip()
            for value in group.get(
                "difficulty_class",
                pd.Series(dtype=object),
            ).dropna()
            if str(value).strip()
        }
        outcomes = {
            str(value).strip()
            for value in group.get(
                "outcome_class",
                pd.Series(dtype=object),
            ).dropna()
            if str(value).strip()
        }
        summaries[lineage.fingerprint] = {
            **lineage.as_dict(),
            "scenario_ids": scenarios,
            "difficulty_class": (
                next(iter(difficulties))
                if len(difficulties) == 1
                else "mixed" if difficulties else "unknown"
            ),
            "outcome_class_at_assignment": (
                next(iter(outcomes))
                if len(outcomes) == 1
                else "mixed" if outcomes else "unknown"
            ),
        }
    if not summaries:
        raise ValueError(f"{source} contains no physical lineages.")
    return summaries


def _new_manifest(
    *,
    physics_config: PhysicsConfig,
    seed: int,
    validation_fraction: float,
    min_validation_lineages: int,
    source_hashes: Mapping[str, str] | None,
) -> dict[str, Any]:
    hashes = {
        str(name): _require_sha256(value, name=f"source hash {name}")
        for name, value in (source_hashes or {}).items()
    }
    return {
        "schema_version": PHYSICAL_SPLIT_MANIFEST_SCHEMA_VERSION,
        "lineage_contract_version": PHYSICAL_LINEAGE_CONTRACT_VERSION,
        **physics_provenance(physics_config),
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "min_validation_lineages": int(min_validation_lineages),
        "source_hashes": dict(sorted(hashes.items())),
        "assignments": {},
    }


def _validate_assignment(
    fingerprint: str,
    raw: object,
    *,
    source: Path,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Physical split assignment must be an object: {source}."
        )
    entry = dict(raw)
    lineage = require_physical_lineage(
        entry,
        source=f"{source} assignment {fingerprint}",
    )
    if fingerprint != lineage.fingerprint:
        raise ValueError(
            f"Physical split assignment key mismatch in {source}: "
            f"{fingerprint} != {lineage.fingerprint}."
        )
    split = str(entry.get("split", "")).strip()
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"Invalid physical split {split!r} in {source}."
        )
    assigned_iteration = _require_positive_integer(
        entry.get("assigned_iteration"),
        name="assigned_iteration",
    )
    raw_scenarios = entry.get("scenario_ids")
    if not isinstance(raw_scenarios, list):
        raise ValueError(
            f"Physical split scenario_ids must be a list in {source}."
        )
    scenario_ids = sorted(
        {
            _require_non_negative_integer(
                value,
                name="scenario_id",
            )
            for value in raw_scenarios
        }
    )
    if not scenario_ids:
        raise ValueError(
            f"Physical split scenario_ids must not be empty in {source}."
        )
    return {
        **lineage.as_dict(),
        "split": split,
        "assigned_iteration": assigned_iteration,
        "scenario_ids": scenario_ids,
        "difficulty_class": str(
            entry.get("difficulty_class", "unknown")
        ).strip() or "unknown",
        "outcome_class_at_assignment": str(
            entry.get("outcome_class_at_assignment", "unknown")
        ).strip() or "unknown",
        "assignment_rank": _require_sha256(
            entry.get("assignment_rank"),
            name="assignment_rank",
        ),
    }


def load_physical_split_manifest(
    path: str | Path,
    *,
    physics_config: PhysicsConfig,
    seed: int,
    validation_fraction: float,
    min_validation_lineages: int,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Physical split manifest must be an object: {manifest_path}."
        )
    require_exact_contract_version(
        payload.get("schema_version"),
        expected=PHYSICAL_SPLIT_MANIFEST_SCHEMA_VERSION,
        name="physical split manifest schema",
        source=str(manifest_path),
        regeneration_command=(
            "remove the incompatible run directory and restart self-play"
        ),
    )
    require_exact_contract_version(
        payload.get("lineage_contract_version"),
        expected=PHYSICAL_LINEAGE_CONTRACT_VERSION,
        name="physical lineage contract",
        source=str(manifest_path),
        regeneration_command=(
            "regenerate physical lineage artifacts"
        ),
    )
    require_physics_provenance(
        payload,
        source=str(manifest_path),
        expected_physics_config=physics_config,
    )
    expected_fraction = _require_fraction(
        validation_fraction,
        name="validation_fraction",
    )
    expected_minimum = _require_positive_integer(
        min_validation_lineages,
        name="min_validation_lineages",
    )
    expected_seed = _require_non_negative_integer(seed, name="seed")
    observed_seed = _require_non_negative_integer(
        payload.get("seed"),
        name="manifest seed",
    )
    if observed_seed != expected_seed:
        raise ValueError(
            f"Physical split seed mismatch for {manifest_path}."
        )
    observed_fraction = _require_fraction(
        payload.get("validation_fraction"),
        name="manifest validation_fraction",
    )
    if not math.isclose(
        observed_fraction,
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Physical split validation_fraction mismatch for {manifest_path}."
        )
    observed_minimum = _require_positive_integer(
        payload.get("min_validation_lineages"),
        name="manifest min_validation_lineages",
    )
    if observed_minimum != expected_minimum:
        raise ValueError(
            f"Physical split min_validation_lineages mismatch for {manifest_path}."
        )

    recorded_hashes = payload.get("source_hashes", {})
    if not isinstance(recorded_hashes, Mapping):
        raise ValueError(
            f"Physical split source_hashes must be an object: {manifest_path}."
        )
    normalized_recorded_hashes = {
        str(name): _require_sha256(value, name=f"source hash {name}")
        for name, value in recorded_hashes.items()
    }
    expected_hashes = {
        str(name): _require_sha256(value, name=f"source hash {name}")
        for name, value in (source_hashes or {}).items()
    }
    for name, expected_hash in expected_hashes.items():
        observed = normalized_recorded_hashes.get(name)
        if observed is not None and observed != expected_hash:
            raise ValueError(
                f"Physical split source hash mismatch for {name}: "
                f"{manifest_path}."
            )

    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, Mapping):
        raise ValueError(
            f"Physical split assignments must be an object: {manifest_path}."
        )
    assignments = {
        str(fingerprint): _validate_assignment(
            str(fingerprint),
            raw,
            source=manifest_path,
        )
        for fingerprint, raw in raw_assignments.items()
    }
    for fingerprint, entry in assignments.items():
        expected_rank = _assignment_rank(expected_seed, fingerprint)
        if entry["assignment_rank"] != expected_rank:
            raise ValueError(
                f"Physical split assignment_rank mismatch for "
                f"{fingerprint} in {manifest_path}."
            )

    expected_counts = {
        "lineage_count": len(assignments),
        "train_lineage_count": sum(
            entry["split"] == TRAIN_SPLIT
            for entry in assignments.values()
        ),
        "validation_lineage_count": sum(
            entry["split"] == VALIDATION_SPLIT
            for entry in assignments.values()
        ),
    }
    for name, expected_count in expected_counts.items():
        observed_count = _require_non_negative_integer(
            payload.get(name),
            name=f"manifest {name}",
        )
        if observed_count != expected_count:
            raise ValueError(
                f"Physical split {name} mismatch for {manifest_path}."
            )
    _require_positive_integer(
        payload.get("last_updated_iteration"),
        name="manifest last_updated_iteration",
    )
    if assignments and (
        expected_counts["train_lineage_count"] == 0
        or expected_counts["validation_lineage_count"] == 0
    ):
        raise ValueError(
            f"Physical split manifest must contain both splits: {manifest_path}."
        )

    payload["source_hashes"] = dict(sorted(normalized_recorded_hashes.items()))
    payload["assignments"] = assignments
    return payload


def _validation_target(
    total_lineages: int,
    *,
    validation_fraction: float,
    min_validation_lineages: int,
) -> int:
    if total_lineages < 2:
        raise ValueError(
            "Physical split requires at least two lineages."
        )
    target = max(
        int(min_validation_lineages),
        int(math.ceil(total_lineages * validation_fraction)),
    )
    if target >= total_lineages:
        raise ValueError(
            "Requested physical validation split leaves no training lineage: "
            f"total={total_lineages}, validation={target}."
        )
    return target


def assign_physical_split(
    frame: pd.DataFrame,
    *,
    manifest_path: str | Path,
    physics_config: PhysicsConfig,
    seed: int,
    validation_fraction: float,
    min_validation_lineages: int,
    iteration: int,
    source: str | Path,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    fraction = _require_fraction(
        validation_fraction,
        name="validation_fraction",
    )
    resolved_seed = _require_non_negative_integer(seed, name="seed")
    minimum = _require_positive_integer(
        min_validation_lineages,
        name="min_validation_lineages",
    )
    assigned_iteration = _require_positive_integer(
        iteration,
        name="iteration",
    )
    summaries = _lineage_summary(frame, source=source)
    path = Path(manifest_path)
    manifest = load_physical_split_manifest(
        path,
        physics_config=physics_config,
        seed=resolved_seed,
        validation_fraction=fraction,
        min_validation_lineages=minimum,
        source_hashes=source_hashes,
    )
    if manifest is None:
        manifest = _new_manifest(
            physics_config=physics_config,
            seed=resolved_seed,
            validation_fraction=fraction,
            min_validation_lineages=minimum,
            source_hashes=source_hashes,
        )

    assignments = dict(manifest["assignments"])
    unseen = sorted(set(summaries) - set(assignments))
    total_after = len(assignments) + len(unseen)
    target_validation = _validation_target(
        total_after,
        validation_fraction=fraction,
        min_validation_lineages=minimum,
    )
    current_validation = sum(
        entry["split"] == VALIDATION_SPLIT
        for entry in assignments.values()
    )
    validation_needed = max(0, target_validation - current_validation)
    ranked_unseen = sorted(
        unseen,
        key=lambda fingerprint: _assignment_rank(
            resolved_seed,
            fingerprint,
        ),
    )

    for position, fingerprint in enumerate(ranked_unseen):
        summary = summaries[fingerprint]
        split = (
            VALIDATION_SPLIT
            if position < validation_needed
            else TRAIN_SPLIT
        )
        assignments[fingerprint] = {
            **summary,
            "split": split,
            "assigned_iteration": assigned_iteration,
            "assignment_rank": _assignment_rank(
                resolved_seed,
                fingerprint,
            ),
        }

    for fingerprint, summary in summaries.items():
        entry = assignments[fingerprint]
        entry["scenario_ids"] = sorted(
            set(entry["scenario_ids"]) | set(summary["scenario_ids"])
        )

    merged_hashes = dict(manifest.get("source_hashes", {}))
    for name, value in (source_hashes or {}).items():
        merged_hashes[str(name)] = _require_sha256(
            value,
            name=f"source hash {name}",
        )
    manifest["source_hashes"] = dict(sorted(merged_hashes.items()))
    manifest["assignments"] = dict(sorted(assignments.items()))
    manifest["lineage_count"] = len(assignments)
    manifest["train_lineage_count"] = sum(
        entry["split"] == TRAIN_SPLIT
        for entry in assignments.values()
    )
    manifest["validation_lineage_count"] = sum(
        entry["split"] == VALIDATION_SPLIT
        for entry in assignments.values()
    )
    manifest["last_updated_iteration"] = assigned_iteration
    _write_json_atomic(manifest, path)
    return manifest


def split_frame_by_manifest(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    source: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_lineage_columns(frame, source=source)
    raw_assignments = manifest.get("assignments")
    if not isinstance(raw_assignments, Mapping):
        raise ValueError("Physical split manifest has no assignments.")
    split_by_fingerprint = {
        str(fingerprint): str(entry.get("split", ""))
        for fingerprint, entry in raw_assignments.items()
        if isinstance(entry, Mapping)
    }
    fingerprints = (
        frame[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    missing = sorted(set(fingerprints) - set(split_by_fingerprint))
    if missing:
        raise ValueError(
            f"{source} contains unassigned physical lineages: {missing[:5]}."
        )
    labels = fingerprints.map(split_by_fingerprint)
    invalid = sorted(set(labels) - _VALID_SPLITS)
    if invalid:
        raise ValueError(
            f"Physical split manifest contains invalid labels: {invalid}."
        )
    train = frame.loc[labels == TRAIN_SPLIT].copy()
    validation = frame.loc[labels == VALIDATION_SPLIT].copy()
    if train.empty or validation.empty:
        raise ValueError(
            "Physical split produced an empty train or validation frame."
        )
    overlap = set(
        train[PHYSICAL_LINEAGE_FINGERPRINT_FIELD].astype(str)
    ) & set(
        validation[PHYSICAL_LINEAGE_FINGERPRINT_FIELD].astype(str)
    )
    if overlap:
        raise RuntimeError(
            f"Physical lineage leakage detected: {sorted(overlap)[:5]}."
        )
    return train, validation

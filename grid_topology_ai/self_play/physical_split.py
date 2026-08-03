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
    require_one_lineage_per_scenario,
    require_physical_lineage,
)

PHYSICAL_SPLIT_MANIFEST_SCHEMA_VERSION = 2
PHYSICAL_SPLIT_ASSIGNMENT_STRATEGY = "difficulty_outcome_stratified_v1"
PHYSICAL_SPLIT_STRATIFICATION_FIELDS = (
    "difficulty_class",
    "outcome_class_at_assignment",
)
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


def _normalize_stratum_label(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number):
            return "unknown"
    text = str(value).strip().casefold()
    return text or "unknown"


def _stratum_id(difficulty: object, outcome: object) -> str:
    payload = {
        "difficulty_class": _normalize_stratum_label(difficulty),
        "outcome_class_at_assignment": _normalize_stratum_label(outcome),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stratum_rank(seed: int, stratum_id: str) -> str:
    payload = f"{int(seed)}:stratum:{stratum_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _group_label(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return "unknown"
    values = {
        _normalize_stratum_label(value)
        for value in group[column].tolist()
        if _normalize_stratum_label(value) != "unknown"
    }
    if not values:
        return "unknown"
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


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
        difficulty = _group_label(group, "difficulty_class")
        outcome = _group_label(group, "outcome_class")
        summaries[lineage.fingerprint] = {
            **lineage.as_dict(),
            "scenario_ids": scenarios,
            "difficulty_class": difficulty,
            "outcome_class_at_assignment": outcome,
            "stratum_id": _stratum_id(difficulty, outcome),
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
        "assignment_strategy": PHYSICAL_SPLIT_ASSIGNMENT_STRATEGY,
        "stratification_fields": list(PHYSICAL_SPLIT_STRATIFICATION_FIELDS),
        **physics_provenance(physics_config),
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "min_validation_lineages": int(min_validation_lineages),
        "source_hashes": dict(sorted(hashes.items())),
        "assignments": {},
        "strata": {},
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
    difficulty = _normalize_stratum_label(
        entry.get("difficulty_class", "unknown")
    )
    outcome = _normalize_stratum_label(
        entry.get("outcome_class_at_assignment", "unknown")
    )
    expected_stratum = _stratum_id(difficulty, outcome)
    observed_stratum = _require_sha256(
        entry.get("stratum_id"),
        name="stratum_id",
    )
    if observed_stratum != expected_stratum:
        raise ValueError(
            f"Physical split stratum_id mismatch for {fingerprint} "
            f"in {source}."
        )
    return {
        **lineage.as_dict(),
        "split": split,
        "assigned_iteration": assigned_iteration,
        "scenario_ids": scenario_ids,
        "difficulty_class": difficulty,
        "outcome_class_at_assignment": outcome,
        "stratum_id": observed_stratum,
        "assignment_rank": _require_sha256(
            entry.get("assignment_rank"),
            name="assignment_rank",
        ),
    }


def _strata_summary(
    assignments: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    strata: dict[str, dict[str, Any]] = {}
    for entry in assignments.values():
        stratum_id = str(entry["stratum_id"])
        summary = strata.setdefault(
            stratum_id,
            {
                "difficulty_class": str(entry["difficulty_class"]),
                "outcome_class_at_assignment": str(
                    entry["outcome_class_at_assignment"]
                ),
                "lineage_count": 0,
                "train_lineage_count": 0,
                "validation_lineage_count": 0,
            },
        )
        if (
            summary["difficulty_class"] != entry["difficulty_class"]
            or summary["outcome_class_at_assignment"]
            != entry["outcome_class_at_assignment"]
        ):
            raise ValueError(
                f"Physical split stratum {stratum_id} has mixed labels."
            )
        summary["lineage_count"] += 1
        summary[f"{entry['split']}_lineage_count"] += 1
    return dict(sorted(strata.items()))


def _validate_strata_payload(
    raw: object,
    *,
    expected: Mapping[str, Mapping[str, Any]],
    source: Path,
) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Physical split strata must be an object: {source}."
        )
    normalized: dict[str, dict[str, Any]] = {}
    for stratum_id, entry in raw.items():
        key = _require_sha256(stratum_id, name="stratum id")
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Physical split stratum must be an object: {source}."
            )
        difficulty = _normalize_stratum_label(
            entry.get("difficulty_class", "unknown")
        )
        outcome = _normalize_stratum_label(
            entry.get("outcome_class_at_assignment", "unknown")
        )
        if key != _stratum_id(difficulty, outcome):
            raise ValueError(
                f"Physical split stratum key mismatch in {source}."
            )
        normalized[key] = {
            "difficulty_class": difficulty,
            "outcome_class_at_assignment": outcome,
            "lineage_count": _require_non_negative_integer(
                entry.get("lineage_count"),
                name="stratum lineage_count",
            ),
            "train_lineage_count": _require_non_negative_integer(
                entry.get("train_lineage_count"),
                name="stratum train_lineage_count",
            ),
            "validation_lineage_count": _require_non_negative_integer(
                entry.get("validation_lineage_count"),
                name="stratum validation_lineage_count",
            ),
        }
    if normalized != dict(expected):
        raise ValueError(
            f"Physical split strata summary mismatch for {source}."
        )


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
        regeneration_command="regenerate physical lineage artifacts",
    )
    if payload.get("assignment_strategy") != PHYSICAL_SPLIT_ASSIGNMENT_STRATEGY:
        raise ValueError(
            f"Physical split assignment strategy mismatch for {manifest_path}."
        )
    if payload.get("stratification_fields") != list(
        PHYSICAL_SPLIT_STRATIFICATION_FIELDS
    ):
        raise ValueError(
            f"Physical split stratification fields mismatch for {manifest_path}."
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

    expected_strata = _strata_summary(assignments)
    _validate_strata_payload(
        payload.get("strata"),
        expected=expected_strata,
        source=manifest_path,
    )
    payload["source_hashes"] = dict(sorted(normalized_recorded_hashes.items()))
    payload["assignments"] = assignments
    payload["strata"] = expected_strata
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


def _strata_for_assignment(
    assignments: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    strata: dict[str, dict[str, Any]] = {}
    for entry in assignments.values():
        stratum = strata.setdefault(
            str(entry["stratum_id"]),
            {
                "difficulty_class": entry["difficulty_class"],
                "outcome_class_at_assignment": entry[
                    "outcome_class_at_assignment"
                ],
                "existing_validation": 0,
                "existing_train": 0,
                "unseen": [],
            },
        )
        stratum[f"existing_{entry['split']}"] += 1
    for fingerprint, summary in summaries.items():
        if fingerprint in assignments:
            continue
        stratum = strata.setdefault(
            str(summary["stratum_id"]),
            {
                "difficulty_class": summary["difficulty_class"],
                "outcome_class_at_assignment": summary[
                    "outcome_class_at_assignment"
                ],
                "existing_validation": 0,
                "existing_train": 0,
                "unseen": [],
            },
        )
        stratum["unseen"].append(fingerprint)
    return strata


def _validation_capacity(stratum: Mapping[str, Any]) -> int:
    unseen_count = len(stratum["unseen"])
    if unseen_count == 0:
        return 0
    existing_train = int(stratum["existing_train"])
    existing_validation = int(stratum["existing_validation"])
    total = existing_train + existing_validation + unseen_count
    if total >= 2 and existing_train == 0:
        return max(0, unseen_count - 1)
    return unseen_count


def _select_validation_lineages(
    *,
    assignments: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    seed: int,
    validation_fraction: float,
    min_validation_lineages: int,
) -> set[str]:
    strata = _strata_for_assignment(assignments, summaries)
    total_after = len(assignments) + sum(
        len(stratum["unseen"])
        for stratum in strata.values()
    )
    base_target = _validation_target(
        total_after,
        validation_fraction=validation_fraction,
        min_validation_lineages=min_validation_lineages,
    )
    current_validation = sum(
        entry["split"] == VALIDATION_SPLIT
        for entry in assignments.values()
    )
    uncovered = [
        stratum_id
        for stratum_id, stratum in strata.items()
        if (
            int(stratum["existing_validation"]) == 0
            and int(stratum["existing_train"])
            + len(stratum["unseen"]) >= 2
            and _validation_capacity(stratum) > 0
        )
    ]
    total_capacity = sum(
        _validation_capacity(stratum)
        for stratum in strata.values()
    )
    desired_total = max(
        base_target,
        current_validation + len(uncovered),
    )
    desired_total = min(
        total_after - 1,
        current_validation + total_capacity,
        desired_total,
    )
    seats = max(0, desired_total - current_validation)
    if seats == 0:
        return set()

    ranked_unseen = {
        stratum_id: sorted(
            stratum["unseen"],
            key=lambda fingerprint: _assignment_rank(seed, fingerprint),
        )
        for stratum_id, stratum in strata.items()
    }
    selected: set[str] = set()
    projected_validation = {
        stratum_id: int(stratum["existing_validation"])
        for stratum_id, stratum in strata.items()
    }
    capacity = {
        stratum_id: _validation_capacity(stratum)
        for stratum_id, stratum in strata.items()
    }

    for stratum_id in sorted(
        uncovered,
        key=lambda value: _stratum_rank(seed, value),
    ):
        if seats == 0:
            break
        fingerprint = ranked_unseen[stratum_id].pop(0)
        selected.add(fingerprint)
        projected_validation[stratum_id] += 1
        capacity[stratum_id] -= 1
        seats -= 1

    while seats > 0:
        candidates = [
            stratum_id
            for stratum_id in strata
            if capacity[stratum_id] > 0 and ranked_unseen[stratum_id]
        ]
        if not candidates:
            break

        def priority(stratum_id: str) -> tuple[float, str]:
            stratum = strata[stratum_id]
            total = (
                int(stratum["existing_train"])
                + int(stratum["existing_validation"])
                + len(stratum["unseen"])
            )
            deficit = (
                total * validation_fraction
                - projected_validation[stratum_id]
            )
            return (-deficit, _stratum_rank(seed, stratum_id))

        chosen = min(candidates, key=priority)
        fingerprint = ranked_unseen[chosen].pop(0)
        selected.add(fingerprint)
        projected_validation[chosen] += 1
        capacity[chosen] -= 1
        seats -= 1

    return selected


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
    validation_unseen = _select_validation_lineages(
        assignments=assignments,
        summaries=summaries,
        seed=resolved_seed,
        validation_fraction=fraction,
        min_validation_lineages=minimum,
    )
    unseen = sorted(
        set(summaries) - set(assignments),
        key=lambda fingerprint: _assignment_rank(
            resolved_seed,
            fingerprint,
        ),
    )
    for fingerprint in unseen:
        summary = summaries[fingerprint]
        assignments[fingerprint] = {
            **summary,
            "split": (
                VALIDATION_SPLIT
                if fingerprint in validation_unseen
                else TRAIN_SPLIT
            ),
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
    manifest["strata"] = _strata_summary(assignments)
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

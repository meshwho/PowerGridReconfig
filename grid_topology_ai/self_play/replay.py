from __future__ import annotations

import gzip
import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral, Real
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from grid_topology_ai.config import ReplayBufferConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    REPLAY_BUFFER_SCHEMA_VERSION,
    physics_provenance,
    require_exact_contract_version,
    require_outcome_objective_version,
    require_physics_provenance,
    require_topology_action_provenance,
    topology_action_provenance,
)
from grid_topology_ai.models.graph_batch import collate_graph_samples
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.self_play.artifacts import sha256_file, sha256_json
from grid_topology_ai.self_play.example_validation import (
    load_and_validate_examples_csv,
    validate_example_outcome_contracts,
)
from grid_topology_ai.training.checkpoints import (
    extract_normalization_stats,
    load_checkpoint_payload,
)
from grid_topology_ai.topology_actions import (
    STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    ActionSlot,
    ActionSpaceConfig,
    action_layout_fingerprint,
    require_branch_status_policy_layout,
)

__all__ = (
    "RollingReplayBuffer",
    "load_and_validate_examples_csv",
)

_CHUNK_HEADER_RECORD_TYPE = "replay_chunk_header"
_REPLAY_MANIFEST_FORMAT_VERSION = 3
_REPLAY_CHUNK_FORMAT_VERSION = 2


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values to JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _row_to_json_safe_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in row.items()}


def _write_json_line(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            _row_to_json_safe_dict(dict(payload)),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    handle.write("\n")


def _write_jsonl_gz(
    *,
    header: dict[str, Any],
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
            _write_json_line(handle, header)
            for row in rows:
                _write_json_line(handle, row)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_jsonl_gz(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Replay buffer file not found: {path}")
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid replay JSON in {path} at line {line_number}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Replay record must be an object: {path} line {line_number}."
                )
            records.append(payload)
    if not records:
        raise ValueError(f"Replay chunk is empty: {path}")
    header = records[0]
    rows = records[1:]
    if header.get("record_type") != _CHUNK_HEADER_RECORD_TYPE:
        raise ValueError(
            f"Replay chunk has no {_CHUNK_HEADER_RECORD_TYPE!r} header: {path}"
        )
    if not rows:
        raise ValueError(f"Replay chunk contains no examples: {path}")
    return header, rows


def _load_examples_csv(path: str | Path) -> list[dict[str, Any]]:
    df = load_and_validate_examples_csv(path)
    return [_row_to_json_safe_dict(row) for row in df.to_dict(orient="records")]


def _validate_replay_batch_outcomes(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> None:
    if not rows:
        return
    validate_example_outcome_contracts(pd.DataFrame(rows), source_path=source)


def _require_replay_row_contracts(
    row: dict[str, Any],
    *,
    source: str,
    expected_physics_config: PhysicsConfig,
    expected_action_space_config: ActionSpaceConfig | None = None,
    expected_action_layout: tuple[ActionSlot, ...] | None = None,
) -> tuple[ActionSpaceConfig, tuple[ActionSlot, ...]]:
    require_exact_contract_version(
        row.get("physical_objective_schema_version"),
        expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        name="physical-objective contract",
        source=source,
        regeneration_command="python -m scripts.self_play.generate ...",
    )
    require_outcome_objective_version(row, source=source)
    require_exact_contract_version(
        row.get("outcome_value_target_contract_version"),
        expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
        name="outcome/value-target contract",
        source=source,
        regeneration_command="python -m scripts.self_play.generate ...",
    )
    require_physics_provenance(
        row,
        source=source,
        expected_physics_config=expected_physics_config,
    )
    action_space_config, action_layout = require_topology_action_provenance(
        row,
        source=source,
        expected_action_space_config=expected_action_space_config,
        expected_action_layout=expected_action_layout,
    )
    validate_example_outcome_contracts(pd.DataFrame([row]), source_path=source)
    return action_space_config, action_layout


def _load_manifest(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Replay manifest must be an object: {path}")
    return payload


def _objective_contract() -> dict[str, int]:
    return {
        "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        "outcome_objective_version": OUTCOME_OBJECTIVE_VERSION,
        "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    }


def _objective_contract_fingerprint() -> str:
    return sha256_json(_objective_contract())


def _episode_count(rows: list[dict[str, Any]]) -> int:
    return len({_episode_key(row) for row in rows})


def _scenario_count(rows: list[dict[str, Any]]) -> int:
    return len({str(row.get("scenario_id")) for row in rows})


def _layout_fingerprints(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> tuple[str, ...]:
    fingerprints = {
        _require_sha256(
            row.get("action_layout_fingerprint"),
            name="action layout fingerprint",
            source=source,
        )
        for row in rows
    }
    if not fingerprints:
        raise ValueError(f"No action layouts found for {source}.")
    return tuple(sorted(fingerprints))


def _require_layout_fingerprints(
    value: object,
    *,
    source: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(
            "action_layout_fingerprints must be "
            f"a list for {source}."
        )
    fingerprints = tuple(
        _require_sha256(
            item,
            name="action layout fingerprint",
            source=source,
        )
        for item in value
    )
    if not fingerprints:
        raise ValueError(
            "action_layout_fingerprints must not "
            f"be empty for {source}."
        )
    normalized = tuple(sorted(set(fingerprints)))
    if fingerprints != normalized:
        raise ValueError(
            "action_layout_fingerprints must be "
            f"sorted and unique for {source}."
        )
    return fingerprints


def _require_sha256(value: object, *, name: str, source: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(
            f"Invalid {name} for {source}: expected a SHA-256 digest."
        )
    return text


def _exact_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return int(number) if number.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("+", "-")):
            sign = text[0]
            digits = text[1:]
        else:
            sign = ""
            digits = text
        if digits.isdigit():
            return int(sign + digits)
    return None


def _require_positive_integer(
    value: object,
    *,
    name: str,
    source: str,
) -> int:
    parsed = _exact_integer(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"Invalid {name} for {source}: {value!r}")
    return parsed


def _require_non_negative_integer(
    value: object,
    *,
    name: str,
    source: str,
) -> int:
    parsed = _exact_integer(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"Invalid {name} for {source}: {value!r}")
    return parsed


SAMPLING_CONTRACT_VERSION = 2
AGE_DECAY = 0.95
ERROR_PRIORITY_SCALE = 0.10
_ERROR_FIELDS = (
    "value_error",
    "value_abs_error",
    "td_error",
    "policy_error",
    "policy_kl_error",
)
_DIFFICULTY_FIELDS = (
    "difficulty",
    "difficulty_class",
    "difficulty_label",
    "scenario_difficulty",
    "difficulty_bucket",
    "difficulty_level",
)


def _save_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
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
                manifest,
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _episode_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    episode_id = str(row.get("episode_id", "")).strip()
    if episode_id:
        return ("episode", episode_id)

    return (
        "scenario",
        str(row.get("run_id", "")),
        str(row.get("replay_iteration", row.get("iteration", ""))),
        str(row.get("scenario_id", "")),
    )


def _first_text(rows: list[dict[str, Any]], *fields: str) -> str:
    for field in fields:
        for row in rows:
            value = row.get(field)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _finite_error(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = abs(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _selected_action_policy_error(row: dict[str, Any]) -> float | None:
    """Return one-hot policy loss when explicit training errors are absent."""

    selected = row.get("selected_action_id")
    raw_policy = row.get("mcts_policy_json")
    if selected is None or raw_policy is None or isinstance(selected, bool):
        return None

    try:
        selected_action_id = int(selected)
        policy = json.loads(str(raw_policy))
    except (TypeError, ValueError, json.JSONDecodeError, OverflowError):
        return None

    if selected_action_id < 0 or not isinstance(policy, Mapping):
        return None

    probability = _finite_error(
        policy.get(str(selected_action_id), policy.get(selected_action_id))
    )
    if probability is None or probability <= 0.0 or probability > 1.0:
        return None

    return float(-math.log(max(probability, 1e-12)))


def _error_score(rows: list[dict[str, Any]]) -> float:
    explicit: list[float] = []
    for row in rows:
        for field in _ERROR_FIELDS:
            value = _finite_error(row.get(field))
            if value is not None:
                explicit.append(value)

    values = explicit
    if not values:
        values = [
            value
            for row in rows
            if (value := _selected_action_policy_error(row)) is not None
        ]
    if not values:
        return 0.0

    largest = max(values)
    return float(largest / (1.0 + largest))


class EpisodeSamplingMixin:
    """Sample replay by episode before selecting states."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.scenario_metadata: dict[str, dict[str, Any]] = {}
        super().__init__(*args, **kwargs)

    def set_scenario_metadata(self, pool_metadata: Mapping[str, Any]) -> None:
        scenarios = pool_metadata.get("scenarios", {})
        if not isinstance(scenarios, Mapping):
            raise ValueError("Pool metadata scenarios must be a mapping.")

        normalized: dict[str, dict[str, Any]] = {}
        for scenario_id, metadata in scenarios.items():
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    "Pool scenario metadata must be a mapping: "
                    f"scenario_id={scenario_id!r}."
                )
            try:
                key = str(int(scenario_id))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Invalid scenario_id in pool metadata: {scenario_id!r}."
                ) from exc
            normalized[key] = dict(metadata)
        self.scenario_metadata = normalized

    def _difficulty(self, rows: list[dict[str, Any]]) -> str:
        explicit = _first_text(rows, *_DIFFICULTY_FIELDS)
        if explicit:
            return explicit
        scenario_id = _first_text(rows, "scenario_id")
        metadata = self.scenario_metadata.get(scenario_id, {})
        return str(metadata.get("difficulty_class", "unknown")).strip() or "unknown"

    def _episode_groups(
        self,
        rows: list[dict[str, Any]],
        current_iteration: int,
        rng: np.random.Generator,
    ) -> tuple[
        list[dict[str, Any]],
        dict[tuple[str, str], list[dict[str, Any]]],
    ]:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_episode_key(row)].append(row)

        episodes: list[dict[str, Any]] = []
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for episode_rows in grouped.values():
            shuffled = list(episode_rows)
            rng.shuffle(shuffled)
            iteration = max(
                int(row.get("replay_iteration", -1))
                for row in episode_rows
            )
            age = max(0, int(current_iteration) - iteration)
            priority = AGE_DECAY ** age * (
                1.0 + ERROR_PRIORITY_SCALE * _error_score(episode_rows)
            )
            outcome = _first_text(
                episode_rows,
                "outcome_class",
                "termination_reason",
            ) or "unknown"
            stratum = (outcome, self._difficulty(episode_rows))
            episode = {
                "rows": shuffled,
                "priority": priority,
                "selected": 0,
            }
            episodes.append(episode)
            strata[stratum].append(episode)
        return episodes, strata

    def _sample_episode_rows(
        self,
        rows: list[dict[str, Any]],
        n_examples: int,
        current_iteration: int,
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if n_examples <= 0 or not rows:
            return [], {
                "source_examples": len(rows),
                "source_episodes": 0,
                "selected_examples": 0,
                "selected_episodes": 0,
                "source_strata": {},
                "selected_strata": {},
            }

        episodes, strata = self._episode_groups(
            rows,
            current_iteration,
            rng,
        )
        target = min(int(n_examples), len(rows))
        selected: list[dict[str, Any]] = []
        selected_strata: dict[str, int] = defaultdict(int)

        while len(selected) < target:
            queues: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for stratum, members in strata.items():
                active = [
                    episode
                    for episode in members
                    if episode["rows"]
                ]
                if not active:
                    continue
                weights = np.asarray(
                    [
                        max(float(episode["priority"]), 1e-12)
                        for episode in active
                    ],
                    dtype=np.float64,
                )
                keys = rng.exponential(scale=1.0 / weights)
                queues[stratum] = [
                    active[int(index)]
                    for index in np.argsort(keys)[::-1]
                ]

            if not queues:
                break

            order = list(queues)
            rng.shuffle(order)
            while order and len(selected) < target:
                next_order: list[tuple[str, str]] = []
                for stratum in order:
                    queue = queues[stratum]
                    if not queue:
                        continue
                    episode = queue.pop()
                    selected.append(episode["rows"].pop())
                    episode["selected"] += 1
                    label = (
                        f"outcome={stratum[0]}|difficulty={stratum[1]}"
                    )
                    selected_strata[label] += 1
                    if queue:
                        next_order.append(stratum)
                    if len(selected) >= target:
                        break
                order = next_order
                rng.shuffle(order)

        source_strata = {
            f"outcome={key[0]}|difficulty={key[1]}": len(value)
            for key, value in sorted(strata.items())
        }
        return selected, {
            "source_examples": len(rows),
            "source_episodes": len(episodes),
            "selected_examples": len(selected),
            "selected_episodes": sum(
                episode["selected"] > 0
                for episode in episodes
            ),
            "source_strata": source_strata,
            "selected_strata": dict(sorted(selected_strata.items())),
        }

    def export_mixed_batch(
        self,
        output_path: str | Path,
        *,
        current_iteration: int,
        n_examples: int | None = None,
        fresh_fraction: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if len(self.buffer) < int(self.config.min_size_to_train):
            raise ValueError(
                f"Replay buffer has only {len(self.buffer)} examples, "
                f"but min_size_to_train={self.config.min_size_to_train}."
            )

        total = (
            len(self.buffer)
            if n_examples is None
            else min(int(n_examples), len(self.buffer))
        )
        if total <= 0:
            raise ValueError("n_examples must be positive.")
        fraction = (
            self.config.fresh_fraction
            if fresh_fraction is None
            else fresh_fraction
        )
        fraction = float(np.clip(fraction, 0.0, 1.0))
        rng_seed = int(
            self.config.random_seed
            if seed is None
            else seed
        )
        rng = np.random.default_rng(rng_seed)
        fresh, old = self._split_fresh_old(
            current_iteration=current_iteration
        )

        n_fresh = min(int(round(total * fraction)), len(fresh))
        n_old = min(total - n_fresh, len(old))
        remaining = total - n_fresh - n_old
        take_fresh = min(remaining, len(fresh) - n_fresh)
        n_fresh += take_fresh
        remaining -= take_fresh
        n_old += min(remaining, len(old) - n_old)

        fresh_rows, fresh_meta = self._sample_episode_rows(
            fresh,
            n_fresh,
            current_iteration,
            rng,
        )
        old_rows, old_meta = self._sample_episode_rows(
            old,
            n_old,
            current_iteration,
            rng,
        )
        selected = fresh_rows + old_rows
        if not selected:
            raise ValueError(
                "Could not sample any examples from replay buffer."
            )
        rng.shuffle(selected)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selected).to_csv(output_path, index=False)

        metadata = {
            **physics_provenance(self.physics_config),
            "path": str(output_path),
            "n_examples": len(selected),
            "n_fresh": len(fresh_rows),
            "n_old": len(old_rows),
            "fresh_fraction_target": fraction,
            "fresh_fraction_actual": len(fresh_rows) / len(selected),
            "current_iteration": int(current_iteration),
            "seed": rng_seed,
            "sampling_contract_version": SAMPLING_CONTRACT_VERSION,
            "sampling_unit": "episode_then_state",
            "sampling_strata": ["outcome", "difficulty"],
            "scenario_metadata_count": len(self.scenario_metadata),
            "age_decay_per_iteration": AGE_DECAY,
            "error_priority_scale": ERROR_PRIORITY_SCALE,
            "error_priority_source": (
                "explicit_error_or_selected_action_policy_loss"
            ),
            "fresh_sampling": fresh_meta,
            "old_sampling": old_meta,
        }
        _save_manifest(
            metadata,
            output_path.with_suffix(".metadata.json"),
        )
        return metadata


PREDICTION_ERROR_SCHEMA_VERSION = 1
PREDICTION_ERROR_FILENAME = "replay_prediction_errors.json"


def _require_prediction_error_sha256(value: object, *, source: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef"
        for character in text
    ):
        raise ValueError(
            f"Invalid replay prediction checkpoint SHA-256 for {source}."
        )
    return text


def _non_negative_float(
    value: object,
    *,
    name: str,
    source: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {name} for {source}: {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid {name} for {source}: {value!r}."
        ) from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"Invalid {name} for {source}: {value!r}.")
    return number


class ReplayPredictionErrorMixin(EpisodeSamplingMixin):
    """Persist model errors and feed them into episode sampling priority."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prediction_error_path = self.save_dir / PREDICTION_ERROR_FILENAME
        (
            self.prediction_errors,
            self.prediction_error_last_iteration,
        ) = self._load_prediction_errors()
        self._prune_prediction_errors(persist=False)

    def _load_prediction_errors(
        self,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        if not self.prediction_error_path.exists():
            return {}, 0

        payload = json.loads(
            self.prediction_error_path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError(
                "Replay prediction error sidecar must be an object: "
                f"{self.prediction_error_path}"
            )
        require_exact_contract_version(
            payload.get("schema_version"),
            expected=PREDICTION_ERROR_SCHEMA_VERSION,
            name="replay prediction-error schema",
            source=str(self.prediction_error_path),
            regeneration_command=(
                f"remove {self.prediction_error_path.name} and rerun self-play"
            ),
        )
        require_physics_provenance(
            payload,
            source=str(self.prediction_error_path),
            expected_physics_config=self.physics_config,
        )
        raw_entries = payload.get("entries", {})
        if not isinstance(raw_entries, Mapping):
            raise ValueError(
                "Replay prediction error entries must be a mapping: "
                f"{self.prediction_error_path}"
            )

        entries: dict[str, dict[str, Any]] = {}
        for raw_state_id, raw_entry in raw_entries.items():
            state_id = str(raw_state_id).strip()
            if not state_id or not isinstance(raw_entry, Mapping):
                raise ValueError(
                    "Invalid replay prediction error entry in "
                    f"{self.prediction_error_path}."
                )
            source = f"{self.prediction_error_path} state_id={state_id!r}"
            scored_iteration = int(raw_entry.get("scored_iteration", 0))
            if scored_iteration <= 0:
                raise ValueError(
                    f"Invalid scored_iteration for {source}."
                )
            entries[state_id] = {
                "value_error": _non_negative_float(
                    raw_entry.get("value_error"),
                    name="value_error",
                    source=source,
                ),
                "policy_kl_error": _non_negative_float(
                    raw_entry.get("policy_kl_error"),
                    name="policy_kl_error",
                    source=source,
                ),
                "checkpoint_sha256": _require_prediction_error_sha256(
                    raw_entry.get("checkpoint_sha256"),
                    source=source,
                ),
                "scored_iteration": scored_iteration,
            }

        last_iteration = int(payload.get("last_updated_iteration", 0))
        if last_iteration < 0:
            raise ValueError(
                "Invalid last_updated_iteration in "
                f"{self.prediction_error_path}."
            )
        if int(payload.get("entry_count", len(entries))) != len(entries):
            raise ValueError(
                "Replay prediction error entry_count mismatch in "
                f"{self.prediction_error_path}."
            )
        return entries, last_iteration

    def _save_prediction_errors(self) -> None:
        payload = {
            "schema_version": PREDICTION_ERROR_SCHEMA_VERSION,
            "algorithm": "absolute_value_error_and_policy_kl_v1",
            **physics_provenance(self.physics_config),
            "last_updated_iteration": int(
                self.prediction_error_last_iteration
            ),
            "entry_count": len(self.prediction_errors),
            "entries": dict(sorted(self.prediction_errors.items())),
        }
        _save_manifest(payload, self.prediction_error_path)

    def _prune_prediction_errors(self, *, persist: bool = True) -> None:
        active_state_ids = {
            str(row.get("state_id", "")).strip()
            for row in self.buffer
            if str(row.get("state_id", "")).strip()
        }
        retained = {
            state_id: entry
            for state_id, entry in self.prediction_errors.items()
            if state_id in active_state_ids
        }
        if retained == self.prediction_errors:
            return
        self.prediction_errors = retained
        if persist and (self.prediction_error_path.exists() or retained):
            self._save_prediction_errors()

    def save_manifest(self) -> None:
        super().save_manifest()
        self._prune_prediction_errors()

    def _record_prediction_errors(
        self,
        report: Mapping[str, Any],
        *,
        iteration: int,
    ) -> None:
        require_exact_contract_version(
            report.get("schema_version"),
            expected=PREDICTION_ERROR_SCHEMA_VERSION,
            name="replay prediction-error report",
            source="replay priority scorer",
            regeneration_command="rerun replay prediction scoring",
        )
        checkpoint_sha = _require_prediction_error_sha256(
            report.get("checkpoint_sha256"),
            source="replay priority scorer",
        )
        raw_entries = report.get("entries", {})
        if not isinstance(raw_entries, Mapping):
            raise ValueError("Replay prediction error report entries are invalid.")
        if int(report.get("example_count", -1)) != len(raw_entries):
            raise ValueError(
                "Replay prediction error report example_count mismatch."
            )

        iteration = int(iteration)
        if iteration <= 0:
            raise ValueError("Replay prediction scoring iteration must be positive.")

        for raw_state_id, raw_entry in raw_entries.items():
            state_id = str(raw_state_id).strip()
            if not state_id or not isinstance(raw_entry, Mapping):
                raise ValueError("Invalid replay prediction error report entry.")
            source = f"replay prediction report state_id={state_id!r}"
            self.prediction_errors[state_id] = {
                "value_error": _non_negative_float(
                    raw_entry.get("value_error"),
                    name="value_error",
                    source=source,
                ),
                "policy_kl_error": _non_negative_float(
                    raw_entry.get("policy_kl_error"),
                    name="policy_kl_error",
                    source=source,
                ),
                "checkpoint_sha256": checkpoint_sha,
                "scored_iteration": iteration,
            }

        self.prediction_error_last_iteration = iteration
        self._prune_prediction_errors(persist=False)
        self._save_prediction_errors()

    def _refresh_prediction_errors(
        self,
        *,
        current_iteration: int,
    ) -> dict[str, Any]:
        if self.producer_checkpoint is None or not self.buffer:
            return {
                "checkpoint_sha256": None,
                "refreshed_examples": 0,
                "available_entries": len(self.prediction_errors),
            }

        checkpoint_sha = sha256_file(self.producer_checkpoint)
        stale_rows = [
            row
            for row in self.buffer
            if self.prediction_errors.get(
                str(row.get("state_id", "")).strip(),
                {},
            ).get("checkpoint_sha256")
            != checkpoint_sha
        ]
        if not stale_rows:
            return {
                "checkpoint_sha256": checkpoint_sha,
                "refreshed_examples": 0,
                "available_entries": len(self.prediction_errors),
            }

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.save_dir,
                prefix=".replay_priority_",
                suffix=".csv",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            pd.DataFrame(stale_rows).to_csv(temporary_path, index=False)

            report = score_replay_prediction_errors(
                examples_csv=temporary_path,
                checkpoint_path=self.producer_checkpoint,
                physics_config=self.physics_config,
            )
            self._record_prediction_errors(
                report,
                iteration=current_iteration,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return {
            "checkpoint_sha256": checkpoint_sha,
            "refreshed_examples": len(stale_rows),
            "available_entries": len(self.prediction_errors),
            "mean_value_error": report["mean_value_error"],
            "mean_policy_kl_error": report["mean_policy_kl_error"],
        }

    def _episode_groups(
        self,
        rows: list[dict[str, Any]],
        current_iteration: int,
        rng: np.random.Generator,
    ) -> tuple[
        list[dict[str, Any]],
        dict[tuple[str, str], list[dict[str, Any]]],
    ]:
        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            state_id = str(row.get("state_id", "")).strip()
            entry = self.prediction_errors.get(state_id)
            if entry is not None:
                item["value_error"] = entry["value_error"]
                item["policy_kl_error"] = entry["policy_kl_error"]
            enriched_rows.append(item)
        return super()._episode_groups(
            enriched_rows,
            current_iteration,
            rng,
        )

    def export_mixed_batch(
        self,
        output_path: str | Path,
        *,
        current_iteration: int,
        n_examples: int | None = None,
        fresh_fraction: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        error_refresh = self._refresh_prediction_errors(
            current_iteration=current_iteration
        )
        metadata = super().export_mixed_batch(
            output_path=output_path,
            current_iteration=current_iteration,
            n_examples=n_examples,
            fresh_fraction=fresh_fraction,
            seed=seed,
        )
        metadata.update(
            {
                "prediction_error_schema_version": (
                    PREDICTION_ERROR_SCHEMA_VERSION
                ),
                "prediction_error_entries": len(self.prediction_errors),
                "prediction_error_refresh": error_refresh,
            }
        )
        metadata_path = Path(output_path).with_suffix(".metadata.json")
        _save_manifest(metadata, metadata_path)
        return metadata


_MODEL_TYPE = "graph_policy_value_net_v2"


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> GraphPolicyValueNetV2:
    model_type = str(checkpoint.get("model_type", "")).strip()
    if model_type != _MODEL_TYPE:
        raise ValueError(
            "Replay priority scoring requires a Graph V2 checkpoint, "
            f"got model_type={model_type!r}."
        )

    model = GraphPolicyValueNetV2(
        num_bus_features=int(checkpoint["num_bus_features"]),
        num_branch_features=int(checkpoint["num_branch_features"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_layers=int(checkpoint.get("num_layers", 3)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for name, value in batch.items()
    }


def score_replay_prediction_errors(
    *,
    examples_csv: str | Path,
    checkpoint_path: str | Path,
    physics_config: PhysicsConfig,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Score replay examples against one Graph V2 checkpoint."""

    examples_csv = Path(examples_csv)
    checkpoint_path = Path(checkpoint_path)
    device = _resolve_device()
    checkpoint = dict(
        load_checkpoint_payload(
            checkpoint_path,
            map_location=device,
            expected_physics_config=physics_config,
        )
    )

    if str(checkpoint.get("model_type", "")).strip() != _MODEL_TYPE:
        raise ValueError(
            "Replay priority scoring requires a Graph V2 checkpoint."
        )

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
        normalization_stats=extract_normalization_stats(
            checkpoint,
            source=checkpoint_path,
        ),
        physics_config=physics_config,
    )

    mismatches = [
        name
        for name, expected in {
            "num_bus_features": dataset.num_bus_features,
            "num_branch_features": dataset.num_branch_features,
        }.items()
        if int(checkpoint.get(name, -1)) != int(expected)
    ]
    if mismatches:
        raise ValueError(
            "Replay examples are incompatible with the scoring checkpoint: "
            + ", ".join(mismatches)
            + "."
        )

    require_topology_action_provenance(
        checkpoint,
        source=str(checkpoint_path),
        expected_action_space_config=dataset.topology_action_config,
    )

    model = _load_model(checkpoint, device=device)
    loader = DataLoader(
        dataset,
        batch_size=min(max(1, int(batch_size)), len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_graph_samples,
    )

    entries: dict[str, dict[str, float]] = {}
    value_errors: list[float] = []
    policy_errors: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            policy_logits, predicted_value = model(
                bus_features=batch["bus_features"],
                branch_features=batch["branch_features"],
                edge_index=batch["edge_index"],
                edge_active_mask=batch["edge_active_mask"],
                action_mask=batch["action_mask"],
                node_batch=batch["node_batch"],
                edge_batch=batch["edge_batch"],
            )
            target_policy = batch["target_policy"].float()
            target_value = batch["target_value"].float().reshape(-1)
            predicted_value = predicted_value.float().reshape(-1)

            log_probs = torch.log_softmax(policy_logits.float(), dim=1)
            positive = target_policy > 0.0
            target_log = torch.zeros_like(target_policy)
            target_log[positive] = torch.log(target_policy[positive])
            policy_kl = torch.where(
                positive,
                target_policy * (target_log - log_probs),
                torch.zeros_like(target_policy),
            ).sum(dim=1).clamp_min(0.0)
            value_error = torch.abs(predicted_value - target_value)

            state_ids = [str(value) for value in batch["state_id"]]
            batch_value_errors = value_error.detach().cpu().numpy()
            batch_policy_errors = policy_kl.detach().cpu().numpy()

            for state_id, value_item, policy_item in zip(
                state_ids,
                batch_value_errors,
                batch_policy_errors,
            ):
                if state_id in entries:
                    raise ValueError(
                        f"Duplicate state_id while scoring replay: {state_id!r}."
                    )
                value_number = float(value_item)
                policy_number = float(policy_item)
                if not np.isfinite(value_number) or not np.isfinite(policy_number):
                    raise ValueError(
                        f"Non-finite replay prediction error for state {state_id!r}."
                    )
                entries[state_id] = {
                    "value_error": value_number,
                    "policy_kl_error": policy_number,
                }
                value_errors.append(value_number)
                policy_errors.append(policy_number)

    if len(entries) != len(dataset):
        raise RuntimeError(
            "Replay prediction scoring did not cover every example: "
            f"expected {len(dataset)}, observed {len(entries)}."
        )

    return {
        "schema_version": PREDICTION_ERROR_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_contract_version": int(
            checkpoint["checkpoint_contract_version"]
        ),
        "model_type": str(checkpoint["model_type"]),
        "examples_csv": str(examples_csv),
        "examples_csv_sha256": sha256_file(examples_csv),
        "example_count": len(entries),
        "mean_value_error": float(np.mean(value_errors)),
        "mean_policy_kl_error": float(np.mean(policy_errors)),
        "entries": entries,
    }


class RollingReplayBuffer(ReplayPredictionErrorMixin):
    """Persistent replay buffer with episode-balanced priority sampling."""

    def __init__(
        self,
        save_dir: str | Path,
        config: ReplayBufferConfig | None = None,
        physics_config: PhysicsConfig | None = None,
    ):
        self.scenario_metadata: dict[str, dict[str, Any]] = {}
        self.save_dir = Path(save_dir)
        self.config = config or ReplayBufferConfig()
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.save_dir / "buffer_manifest.json"
        self.buffer: list[dict[str, Any]] = []
        self.producer_checkpoint: Path | None = None
        self.load()
        self.prediction_error_path = self.save_dir / PREDICTION_ERROR_FILENAME
        (
            self.prediction_errors,
            self.prediction_error_last_iteration,
        ) = self._load_prediction_errors()
        self._prune_prediction_errors(persist=False)

    def __len__(self) -> int:
        return len(self.buffer)

    def _require_manifest_contracts(
        self,
        manifest: dict[str, Any],
    ) -> tuple[ActionSpaceConfig, tuple[ActionSlot, ...], tuple[str, ...]]:
        require_exact_contract_version(
            manifest.get("schema_version"),
            expected=REPLAY_BUFFER_SCHEMA_VERSION,
            name="replay-buffer schema",
            source=str(self.manifest_path),
            regeneration_command=(
                "remove or archive the legacy replay directory, then run "
                "python -m scripts.self_play.loop ..."
            ),
        )
        require_exact_contract_version(
            manifest.get("format_version"),
            expected=_REPLAY_MANIFEST_FORMAT_VERSION,
            name="replay-manifest format",
            source=str(self.manifest_path),
            regeneration_command=(
                "remove or archive the legacy replay directory, then run "
                "python -m scripts.self_play.loop ..."
            ),
        )
        require_exact_contract_version(
            manifest.get("physical_objective_schema_version"),
            expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
            name="physical-objective contract",
            source=str(self.manifest_path),
            regeneration_command="python -m scripts.self_play.generate ...",
        )
        require_outcome_objective_version(manifest, source=str(self.manifest_path))
        require_exact_contract_version(
            manifest.get("outcome_value_target_contract_version"),
            expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
            name="outcome/value-target contract",
            source=str(self.manifest_path),
            regeneration_command="python -m scripts.self_play.generate ...",
        )
        expected_objective_fingerprint = _objective_contract_fingerprint()
        if manifest.get("objective_contract_fingerprint") != expected_objective_fingerprint:
            raise ValueError(
                "Replay objective contract fingerprint mismatch for "
                f"{self.manifest_path}."
            )
        require_physics_provenance(
            manifest,
            source=str(self.manifest_path),
            expected_physics_config=self.physics_config,
        )
        action_space_config, representative_layout = require_topology_action_provenance(
            manifest,
            source=str(self.manifest_path),
        )
        policy_layout = require_branch_status_policy_layout(representative_layout)
        if manifest.get("policy_layout") != policy_layout:
            raise ValueError(
                "Replay manifest policy layout "
                f"mismatch: {self.manifest_path}."
            )
        layout_fingerprints = _require_layout_fingerprints(
            manifest.get("action_layout_fingerprints"),
            source=str(self.manifest_path),
        )
        representative_fingerprint = action_layout_fingerprint(representative_layout)
        if representative_fingerprint not in layout_fingerprints:
            raise ValueError(
                "Replay manifest representative layout "
                "is not included in action_layout_fingerprints."
            )
        return action_space_config, representative_layout, layout_fingerprints

    def _require_producer_checkpoint(
        self,
        producer: object,
        *,
        header: Mapping[str, Any],
        source: str,
    ) -> dict[str, Any]:
        if not isinstance(producer, Mapping):
            raise ValueError(
                f"Replay chunk is missing producer checkpoint provenance: {source}"
            )
        payload = dict(producer)
        _require_sha256(payload.get("sha256"), name="producer checkpoint hash", source=source)
        require_exact_contract_version(
            payload.get("checkpoint_contract_version"),
            expected=CHECKPOINT_CONTRACT_VERSION,
            name="checkpoint contract",
            source=f"{source} producer checkpoint",
            regeneration_command="regenerate self-play with a compatible checkpoint",
        )
        model_type = str(payload.get("model_type", "")).strip()
        if not model_type:
            raise ValueError(
                f"Replay producer checkpoint model_type is missing: {source}"
            )
        matching_fields = [
            "physical_objective_schema_version",
            "outcome_objective_version",
            "outcome_value_target_contract_version",
            "state_feature_schema_fingerprint",
            "physics_config_fingerprint",
            "topology_action_config_fingerprint",
            "policy_layout",
        ]
        if model_type not in {"graph_v2", "graph_policy_value_net_v2"}:
            matching_fields.append("action_layout_fingerprint")
        mismatches = [
            field for field in matching_fields if payload.get(field) != header.get(field)
        ]
        if mismatches:
            raise ValueError(
                "Replay producer checkpoint is incompatible with chunk "
                f"{source}: {', '.join(mismatches)}."
            )
        return payload

    def _require_chunk_contracts(
        self,
        header: dict[str, Any],
        *,
        source: str,
        expected_action_space_config: ActionSpaceConfig,
        expected_layout_fingerprints: tuple[str, ...],
    ) -> None:
        require_exact_contract_version(
            header.get("schema_version"),
            expected=REPLAY_BUFFER_SCHEMA_VERSION,
            name="replay-chunk schema",
            source=source,
            regeneration_command="regenerate self-play replay chunks",
        )
        require_exact_contract_version(
            header.get("format_version"),
            expected=_REPLAY_CHUNK_FORMAT_VERSION,
            name="replay-chunk format",
            source=source,
            regeneration_command="regenerate self-play replay chunks",
        )
        require_exact_contract_version(
            header.get("physical_objective_schema_version"),
            expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
            name="physical-objective contract",
            source=source,
            regeneration_command="python -m scripts.self_play.generate ...",
        )
        require_outcome_objective_version(header, source=source)
        require_exact_contract_version(
            header.get("outcome_value_target_contract_version"),
            expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
            name="outcome/value-target contract",
            source=source,
            regeneration_command="python -m scripts.self_play.generate ...",
        )
        if header.get("objective_contract_fingerprint") != _objective_contract_fingerprint():
            raise ValueError(
                f"Replay objective contract fingerprint mismatch for {source}."
            )
        require_physics_provenance(
            header,
            source=source,
            expected_physics_config=self.physics_config,
        )
        _, representative_layout = require_topology_action_provenance(
            header,
            source=source,
            expected_action_space_config=expected_action_space_config,
        )
        policy_layout = require_branch_status_policy_layout(representative_layout)
        if header.get("policy_layout") != policy_layout:
            raise ValueError(f"Replay chunk policy layout mismatch: {source}.")
        header_layout_fingerprints = _require_layout_fingerprints(
            header.get("action_layout_fingerprints"),
            source=source,
        )
        if not set(header_layout_fingerprints).issubset(expected_layout_fingerprints):
            raise ValueError(
                "Replay chunk contains action layouts "
                f"not declared by its manifest: {source}."
            )
        if action_layout_fingerprint(representative_layout) not in header_layout_fingerprints:
            raise ValueError(
                "Replay chunk representative layout is "
                f"not included in its layout list: {source}."
            )
        self._require_producer_checkpoint(
            header.get("producer_checkpoint"),
            header=header,
            source=source,
        )

    def _validate_chunk_contents(
        self,
        header: dict[str, Any],
        rows: list[dict[str, Any]],
        *,
        source: str,
        expected_action_space_config: ActionSpaceConfig,
        expected_layout_fingerprints: tuple[str, ...],
    ) -> None:
        iteration = _require_positive_integer(
            header.get("iteration"), name="chunk iteration", source=source
        )
        expected_examples = _require_non_negative_integer(
            header.get("example_count"), name="chunk example_count", source=source
        )
        expected_episodes = _require_non_negative_integer(
            header.get("episode_count"), name="chunk episode_count", source=source
        )
        expected_scenarios = _require_non_negative_integer(
            header.get("scenario_count"), name="chunk scenario_count", source=source
        )
        observed = {
            "example_count": len(rows),
            "episode_count": _episode_count(rows),
            "scenario_count": _scenario_count(rows),
        }
        expected = {
            "example_count": expected_examples,
            "episode_count": expected_episodes,
            "scenario_count": expected_scenarios,
        }
        if observed != expected:
            raise ValueError(
                f"Replay chunk counts do not match its header for {source}: "
                f"expected {expected}, observed {observed}."
            )
        _validate_replay_batch_outcomes(rows, source=source)
        for row_index, row in enumerate(rows):
            row_iteration = int(row.get("replay_iteration", -1))
            if row_iteration != iteration:
                raise ValueError(
                    "Replay row iteration does not match chunk header for "
                    f"{source} row {row_index}."
                )
            _require_replay_row_contracts(
                row,
                source=f"{source} row {row_index}",
                expected_physics_config=self.physics_config,
                expected_action_space_config=expected_action_space_config,
            )
        observed_layout_fingerprints = _layout_fingerprints(rows, source=source)
        header_layout_fingerprints = _require_layout_fingerprints(
            header.get("action_layout_fingerprints"), source=source
        )
        if observed_layout_fingerprints != header_layout_fingerprints:
            raise ValueError(
                "Replay chunk action layouts do not "
                f"match its header for {source}."
            )
        if not set(observed_layout_fingerprints).issubset(expected_layout_fingerprints):
            raise ValueError(
                "Replay chunk action layouts are not "
                f"declared by the manifest for {source}."
            )
        self._require_episode_sizes(rows, source=source)

    def _require_chunk_hash(
        self,
        item: Mapping[str, Any],
        *,
        file_path: Path,
    ) -> None:
        expected_hash = _require_sha256(
            item.get("sha256"),
            name="replay chunk hash",
            source=str(self.manifest_path),
        )
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Replay chunk hash mismatch for {file_path}: expected "
                f"{expected_hash}, observed {actual_hash}."
            )

    def _require_manifest_item(
        self,
        item: Mapping[str, Any],
        *,
        header: Mapping[str, Any],
        file_path: Path,
    ) -> None:
        producer = header.get("producer_checkpoint")
        producer_sha = producer.get("sha256") if isinstance(producer, Mapping) else None
        checks = {
            "iteration": header.get("iteration"),
            "n": header.get("example_count"),
            "episode_count": header.get("episode_count"),
            "scenario_count": header.get("scenario_count"),
            "objective_contract_fingerprint": header.get("objective_contract_fingerprint"),
            "physics_config_fingerprint": header.get("physics_config_fingerprint"),
            "topology_action_config_fingerprint": header.get("topology_action_config_fingerprint"),
            "policy_layout": header.get("policy_layout"),
            "action_layout_fingerprints": header.get("action_layout_fingerprints"),
            "action_layout_fingerprint": header.get("action_layout_fingerprint"),
            "producer_checkpoint_sha256": producer_sha,
        }
        mismatches = [
            name for name, expected_value in checks.items() if item.get(name) != expected_value
        ]
        if mismatches:
            raise ValueError(
                "Replay manifest metadata does not match chunk header for "
                f"{file_path}: {', '.join(mismatches)}."
            )

    def load(self) -> None:
        """Load compatible chunks, keeping complete episodes."""
        manifest = _load_manifest(self.manifest_path)
        if manifest is None:
            self.buffer = []
            return
        (
            manifest_action_space_config,
            _manifest_representative_layout,
            manifest_layout_fingerprints,
        ) = self._require_manifest_contracts(manifest)
        files = manifest.get("files", [])
        if not isinstance(files, list):
            raise ValueError(f"Invalid replay buffer manifest: {self.manifest_path}")
        selected_chunks: list[list[dict[str, Any]]] = []
        selected_count = 0
        for item in sorted(
            files,
            key=lambda value: int(
                value.get("iteration", 0) if isinstance(value, Mapping) else 0
            ),
            reverse=True,
        ):
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid replay buffer manifest: {self.manifest_path}")
            relative_path = item.get("path")
            if not relative_path:
                raise ValueError(
                    f"Replay manifest entry has no path: {self.manifest_path}"
                )
            file_path = self.save_dir / str(relative_path)
            self._require_chunk_hash(item, file_path=file_path)
            header, rows = _read_jsonl_gz(file_path)
            self._require_manifest_item(item, header=header, file_path=file_path)
            self._require_chunk_contracts(
                header,
                source=str(file_path),
                expected_action_space_config=manifest_action_space_config,
                expected_layout_fingerprints=manifest_layout_fingerprints,
            )
            self._validate_chunk_contents(
                header,
                rows,
                source=str(file_path),
                expected_action_space_config=manifest_action_space_config,
                expected_layout_fingerprints=manifest_layout_fingerprints,
            )
            selected_chunks.append(rows)
            selected_count += len(rows)
            if selected_count >= self.config.max_size:
                break
        self.buffer = [row for chunk in reversed(selected_chunks) for row in chunk]
        self._evict_if_needed()

    def _require_episode_sizes(
        self,
        rows: list[dict[str, Any]],
        *,
        source: str,
    ) -> None:
        sizes: dict[tuple[str, ...], int] = {}
        for row in rows:
            key = _episode_key(row)
            sizes[key] = sizes.get(key, 0) + 1
        oversized = [
            size for size in sizes.values() if size > int(self.config.max_size)
        ]
        if oversized:
            raise ValueError(
                f"Replay episode has {max(oversized)} examples in {source}, "
                f"which exceeds max_size={self.config.max_size}."
            )

    def _evict_if_needed(self) -> None:
        """Remove the oldest complete episodes until max_size is met."""
        max_size = int(self.config.max_size)
        if max_size <= 0:
            raise ValueError("ReplayBufferConfig.max_size must be positive.")
        while len(self.buffer) > max_size:
            oldest_episode = _episode_key(self.buffer[0])
            self.buffer = [
                row for row in self.buffer if _episode_key(row) != oldest_episode
            ]

    def _buffer_action_contract(
        self,
    ) -> tuple[ActionSpaceConfig | None, tuple[ActionSlot, ...] | None]:
        if not self.buffer:
            return None, None
        return require_topology_action_provenance(
            self.buffer[0], source="replay buffer"
        )

    def _prepare_examples(
        self,
        examples: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> tuple[
        list[dict[str, Any]],
        ActionSpaceConfig,
        tuple[ActionSlot, ...],
        tuple[str, ...],
    ]:
        iteration = _require_positive_integer(
            iteration, name="replay iteration", source="replay buffer"
        )
        _validate_replay_batch_outcomes(
            examples, source=f"replay iteration {iteration}"
        )
        expected_action_space_config, _ = self._buffer_action_contract()
        representative_action_layout: tuple[ActionSlot, ...] | None = None
        normalized: list[dict[str, Any]] = []
        for row in examples:
            item = _row_to_json_safe_dict(dict(row))
            row_action_space_config, row_action_layout = _require_replay_row_contracts(
                item,
                source=f"replay iteration {iteration}",
                expected_physics_config=self.physics_config,
                expected_action_space_config=expected_action_space_config,
            )
            if expected_action_space_config is None:
                expected_action_space_config = row_action_space_config
            if representative_action_layout is None:
                representative_action_layout = row_action_layout
            item["replay_iteration"] = iteration
            normalized.append(item)
        if not normalized:
            raise ValueError(f"Replay iteration {iteration} contains no examples.")
        self._require_episode_sizes(
            normalized, source=f"replay iteration {iteration}"
        )
        if expected_action_space_config is None or representative_action_layout is None:
            raise RuntimeError("Replay examples have no topology action contract.")
        layout_fingerprints = _layout_fingerprints(
            normalized, source=f"replay iteration {iteration}"
        )
        return (
            normalized,
            expected_action_space_config,
            representative_action_layout,
            layout_fingerprints,
        )

    def add_examples(
        self,
        examples: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> None:
        """Add validated examples and evict only complete episodes."""
        normalized, _, _, _ = self._prepare_examples(examples, iteration=iteration)
        self.buffer.extend(normalized)
        self._evict_if_needed()

    def add_examples_from_csv(
        self,
        examples_csv: str | Path,
        *,
        iteration: int,
    ) -> list[dict[str, Any]]:
        examples = _load_examples_csv(examples_csv)
        self.add_examples(examples=examples, iteration=iteration)
        return examples

    def set_producer_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Replay producer checkpoint not found: {checkpoint_path}"
            )
        self.producer_checkpoint = checkpoint_path

    def _producer_checkpoint_provenance(
        self,
        checkpoint_path: str | Path,
        *,
        expected_action_space_config: ActionSpaceConfig,
        representative_action_layout: tuple[ActionSlot, ...],
        layout_fingerprints: tuple[str, ...],
    ) -> dict[str, Any]:
        from grid_topology_ai.training.checkpoints import load_checkpoint_payload
        checkpoint_path = Path(checkpoint_path)
        checkpoint = load_checkpoint_payload(
            checkpoint_path,
            map_location="cpu",
            expected_physics_config=self.physics_config,
        )
        model_type = str(checkpoint.get("model_type", "")).strip()
        is_graph_v2 = model_type in {"graph_v2", "graph_policy_value_net_v2"}
        if is_graph_v2:
            require_topology_action_provenance(
                checkpoint,
                source=str(checkpoint_path),
                expected_action_space_config=expected_action_space_config,
            )
        else:
            require_topology_action_provenance(
                checkpoint,
                source=str(checkpoint_path),
                expected_action_space_config=expected_action_space_config,
                expected_action_layout=representative_action_layout,
            )
            checkpoint_layout_fingerprint = str(checkpoint["action_layout_fingerprint"])
            if layout_fingerprints != (checkpoint_layout_fingerprint,):
                raise ValueError(
                    "Graph V1 cannot produce replay containing multiple action layouts."
                )
        return {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "checkpoint_contract_version": int(checkpoint["checkpoint_contract_version"]),
            "model_type": str(checkpoint["model_type"]),
            "policy_layout": str(checkpoint["policy_layout"]),
            "physical_objective_schema_version": int(checkpoint["physical_objective_schema_version"]),
            "outcome_objective_version": int(checkpoint["outcome_objective_version"]),
            "outcome_value_target_contract_version": int(checkpoint["outcome_value_target_contract_version"]),
            "state_feature_schema_fingerprint": str(checkpoint["state_feature_schema_fingerprint"]),
            "physics_config_fingerprint": str(checkpoint["physics_config_fingerprint"]),
            "topology_action_config_fingerprint": str(checkpoint["topology_action_config_fingerprint"]),
            "action_layout_fingerprint": str(checkpoint["action_layout_fingerprint"]),
        }

    def _chunk_header(
        self,
        *,
        rows: list[dict[str, Any]],
        iteration: int,
        action_space_config: ActionSpaceConfig,
        action_layout: tuple[ActionSlot, ...],
        layout_fingerprints: tuple[str, ...],
        producer_checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "record_type": _CHUNK_HEADER_RECORD_TYPE,
            "schema_version": REPLAY_BUFFER_SCHEMA_VERSION,
            "format_version": _REPLAY_CHUNK_FORMAT_VERSION,
            "iteration": int(iteration),
            "example_count": int(len(rows)),
            "episode_count": int(_episode_count(rows)),
            "scenario_count": int(_scenario_count(rows)),
            **_objective_contract(),
            "objective_contract_fingerprint": _objective_contract_fingerprint(),
            **physics_provenance(self.physics_config),
            **topology_action_provenance(action_space_config, action_layout),
            "policy_layout": require_branch_status_policy_layout(action_layout),
            "action_layout_fingerprints": list(layout_fingerprints),
            "producer_checkpoint": producer_checkpoint,
        }

    def save_iteration_file(
        self,
        examples: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> Path:
        """Save one self-contained replay chunk atomically."""
        rows, action_space_config, action_layout, layout_fingerprints = self._prepare_examples(
            examples, iteration=iteration
        )
        if self.producer_checkpoint is None:
            raise RuntimeError("Replay producer checkpoint is not configured.")
        producer = self._producer_checkpoint_provenance(
            self.producer_checkpoint,
            expected_action_space_config=action_space_config,
            representative_action_layout=action_layout,
            layout_fingerprints=layout_fingerprints,
        )
        header = self._chunk_header(
            rows=rows,
            iteration=iteration,
            action_space_config=action_space_config,
            action_layout=action_layout,
            producer_checkpoint=producer,
            layout_fingerprints=layout_fingerprints,
        )
        output_path = self.save_dir / f"buffer_iter_{int(iteration):03d}.jsonl.gz"
        _write_jsonl_gz(header=header, rows=rows, path=output_path)
        return output_path

    def _chunk_manifest_item(
        self,
        file_path: Path,
        *,
        expected_action_space_config: ActionSpaceConfig,
        expected_layout_fingerprints: tuple[str, ...],
    ) -> dict[str, Any]:
        header, rows = _read_jsonl_gz(file_path)
        self._require_chunk_contracts(
            header,
            source=str(file_path),
            expected_action_space_config=expected_action_space_config,
            expected_layout_fingerprints=expected_layout_fingerprints,
        )
        self._validate_chunk_contents(
            header,
            rows,
            source=str(file_path),
            expected_action_space_config=expected_action_space_config,
            expected_layout_fingerprints=expected_layout_fingerprints,
        )
        producer = header["producer_checkpoint"]
        return {
            "path": file_path.name,
            "sha256": sha256_file(file_path),
            "n": int(header["example_count"]),
            "episode_count": int(header["episode_count"]),
            "scenario_count": int(header["scenario_count"]),
            "iteration": int(header["iteration"]),
            "objective_contract_fingerprint": header["objective_contract_fingerprint"],
            "physics_config_fingerprint": header["physics_config_fingerprint"],
            "topology_action_config_fingerprint": header["topology_action_config_fingerprint"],
            "action_layout_fingerprint": header["action_layout_fingerprint"],
            "producer_checkpoint_sha256": producer["sha256"],
            "policy_layout": header["policy_layout"],
            "action_layout_fingerprints": header["action_layout_fingerprints"],
        }

    def save_manifest(self) -> None:
        """Write retained chunk metadata without assuming one topology."""
        if not self.buffer:
            raise ValueError("Cannot save replay manifest for an empty buffer.")
        topology_action_config, _ = require_topology_action_provenance(
            self.buffer[0], source="replay buffer"
        )
        chunk_paths = sorted(self.save_dir.glob("buffer_iter_*.jsonl.gz"))
        if not chunk_paths:
            raise ValueError(f"Replay directory contains no chunks: {self.save_dir}")
        all_layout_fingerprint_set: set[str] = set()
        for file_path in chunk_paths:
            header, _ = _read_jsonl_gz(file_path)
            all_layout_fingerprint_set.update(
                _require_layout_fingerprints(
                    header.get("action_layout_fingerprints"), source=str(file_path)
                )
            )
        all_layout_fingerprints = tuple(sorted(all_layout_fingerprint_set))
        items = [
            self._chunk_manifest_item(
                file_path,
                expected_action_space_config=topology_action_config,
                expected_layout_fingerprints=all_layout_fingerprints,
            )
            for file_path in chunk_paths
        ]
        newest_iteration = max(int(item["iteration"]) for item in items)
        oldest_retained_iteration = max(
            1,
            newest_iteration - int(self.config.retention_iterations) + 1,
        )
        retained_items = [
            item for item in items if int(item["iteration"]) >= oldest_retained_iteration
        ]
        retained_paths = {str(item["path"]) for item in retained_items}
        stale_paths = [
            file_path for file_path in chunk_paths if file_path.name not in retained_paths
        ]
        retained_buffer = [
            row
            for row in self.buffer
            if int(row.get("replay_iteration", -1)) >= oldest_retained_iteration
        ]
        if not retained_buffer:
            raise RuntimeError("Replay retention produced an empty buffer.")
        retained_action_space_config, retained_action_layout = require_topology_action_provenance(
            retained_buffer[0], source="retained replay buffer"
        )
        retained_layout_fingerprints = tuple(
            sorted(
                {
                    str(fingerprint)
                    for item in retained_items
                    for fingerprint in item["action_layout_fingerprints"]
                }
            )
        )
        manifest = {
            "schema_version": REPLAY_BUFFER_SCHEMA_VERSION,
            "format_version": _REPLAY_MANIFEST_FORMAT_VERSION,
            **_objective_contract(),
            "objective_contract_fingerprint": _objective_contract_fingerprint(),
            **physics_provenance(self.physics_config),
            **topology_action_provenance(
                retained_action_space_config,
                retained_action_layout,
            ),
            "policy_layout": require_branch_status_policy_layout(retained_action_layout),
            "action_layout_fingerprints": list(retained_layout_fingerprints),
            "config": asdict(self.config),
            "latest_iteration": int(newest_iteration),
            "oldest_retained_iteration": int(oldest_retained_iteration),
            "total_examples_on_disk": int(sum(int(item["n"]) for item in retained_items)),
            "total_examples_loaded": int(len(retained_buffer)),
            "files": retained_items,
        }
        _save_manifest(manifest=manifest, path=self.manifest_path)
        self.buffer = retained_buffer
        for stale_path in stale_paths:
            stale_path.unlink()

    def add_and_save_from_csv(
        self,
        examples_csv: str | Path,
        *,
        iteration: int,
    ) -> list[dict[str, Any]]:
        examples = _load_examples_csv(examples_csv)
        self.save_iteration_file(examples=examples, iteration=iteration)
        self.add_examples(examples=examples, iteration=iteration)
        self.save_manifest()
        return examples

    def _split_fresh_old(
        self,
        *,
        current_iteration: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        fresh: list[dict[str, Any]] = []
        old: list[dict[str, Any]] = []
        for row in self.buffer:
            row_iteration = int(row.get("replay_iteration", -1))
            if row_iteration == int(current_iteration):
                fresh.append(row)
            else:
                old.append(row)
        return fresh, old

    def export_all(self, output_path: str | Path) -> Path:
        """Export the whole currently loaded buffer to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.buffer)
        df.to_csv(output_path, index=False)
        return output_path

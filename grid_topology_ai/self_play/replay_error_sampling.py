from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.contracts import (
    physics_provenance,
    require_exact_contract_version,
    require_physics_provenance,
)
from grid_topology_ai.self_play.artifacts import sha256_file
from grid_topology_ai.self_play.replay_sampling import (
    EpisodeSamplingMixin,
    _save_manifest,
)

PREDICTION_ERROR_SCHEMA_VERSION = 1
PREDICTION_ERROR_FILENAME = "replay_prediction_errors.json"


def _require_sha256(value: object, *, source: str) -> str:
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
                "checkpoint_sha256": _require_sha256(
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
        checkpoint_sha = _require_sha256(
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

            from grid_topology_ai.self_play.replay_priority import (
                score_replay_prediction_errors,
            )

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

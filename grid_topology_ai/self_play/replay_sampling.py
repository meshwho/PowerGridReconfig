from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.contracts import physics_provenance
from grid_topology_ai.self_play._replay_core import _episode_key, _save_manifest

SAMPLING_CONTRACT_VERSION = 1
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


def _first_text(rows: list[dict[str, Any]], *fields: str) -> str:
    for field in fields:
        for row in rows:
            value = row.get(field)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _error_score(rows: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for row in rows:
        for field in _ERROR_FIELDS:
            value = row.get(field)
            if value is None or isinstance(value, bool):
                continue
            try:
                number = abs(float(value))
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(number):
                values.append(number)
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
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_episode_key(row)].append(row)

        episodes: list[dict[str, Any]] = []
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for episode_rows in grouped.values():
            shuffled = list(episode_rows)
            rng.shuffle(shuffled)
            iteration = max(int(row.get("replay_iteration", -1)) for row in episode_rows)
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
            episode = {"rows": shuffled, "priority": priority, "selected": 0}
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

        episodes, strata = self._episode_groups(rows, current_iteration, rng)
        target = min(int(n_examples), len(rows))
        selected: list[dict[str, Any]] = []
        selected_strata: dict[str, int] = defaultdict(int)

        while len(selected) < target:
            queues: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for stratum, members in strata.items():
                active = [episode for episode in members if episode["rows"]]
                if not active:
                    continue
                weights = np.asarray(
                    [max(float(episode["priority"]), 1e-12) for episode in active],
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
                    label = f"outcome={stratum[0]}|difficulty={stratum[1]}"
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
            "selected_episodes": sum(episode["selected"] > 0 for episode in episodes),
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

        total = len(self.buffer) if n_examples is None else min(int(n_examples), len(self.buffer))
        if total <= 0:
            raise ValueError("n_examples must be positive.")
        fraction = self.config.fresh_fraction if fresh_fraction is None else fresh_fraction
        fraction = float(np.clip(fraction, 0.0, 1.0))
        rng_seed = int(self.config.random_seed if seed is None else seed)
        rng = np.random.default_rng(rng_seed)
        fresh, old = self._split_fresh_old(current_iteration=current_iteration)

        n_fresh = min(int(round(total * fraction)), len(fresh))
        n_old = min(total - n_fresh, len(old))
        remaining = total - n_fresh - n_old
        take_fresh = min(remaining, len(fresh) - n_fresh)
        n_fresh += take_fresh
        remaining -= take_fresh
        n_old += min(remaining, len(old) - n_old)

        fresh_rows, fresh_meta = self._sample_episode_rows(
            fresh, n_fresh, current_iteration, rng
        )
        old_rows, old_meta = self._sample_episode_rows(
            old, n_old, current_iteration, rng
        )
        selected = fresh_rows + old_rows
        if not selected:
            raise ValueError("Could not sample any examples from replay buffer.")
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
            "fresh_sampling": fresh_meta,
            "old_sampling": old_meta,
        }
        _save_manifest(metadata, output_path.with_suffix(".metadata.json"))
        return metadata

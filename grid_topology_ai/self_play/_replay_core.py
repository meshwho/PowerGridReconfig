from __future__ import annotations

import gzip
import json
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.self_play.artifacts import (
    sha256_file,
    sha256_json,
)
from grid_topology_ai.self_play.example_validation import (
    load_and_validate_examples_csv,
    validate_example_outcome_contracts,
)
from grid_topology_ai.topology_actions import (
    ActionSlot,
    ActionSpaceConfig,
)

_CHUNK_HEADER_RECORD_TYPE = "replay_chunk_header"
_REPLAY_MANIFEST_FORMAT_VERSION = 2
_REPLAY_CHUNK_FORMAT_VERSION = 1


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
    return {
        str(key): _json_safe(value)
        for key, value in row.items()
    }


def _write_json_line(
    handle: Any,
    payload: Mapping[str, Any],
) -> None:
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

        with gzip.open(
            temporary_path,
            "wt",
            encoding="utf-8",
        ) as handle:
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

    return [
        _row_to_json_safe_dict(row)
        for row in df.to_dict(orient="records")
    ]


def _validate_replay_batch_outcomes(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> None:
    if not rows:
        return

    validate_example_outcome_contracts(
        pd.DataFrame(rows),
        source_path=source,
    )


def _require_replay_row_contracts(
    row: dict[str, Any],
    *,
    source: str,
    expected_physics_config: PhysicsConfig,
    expected_action_space_config: ActionSpaceConfig | None = None,
    expected_action_layout: tuple[ActionSlot, ...] | None = None,
) -> tuple[
    ActionSpaceConfig,
    tuple[ActionSlot, ...],
]:
    require_exact_contract_version(
        row.get("physical_objective_schema_version"),
        expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        name="physical-objective contract",
        source=source,
        regeneration_command=(
            "python -m scripts.self_play.generate ..."
        ),
    )
    require_outcome_objective_version(
        row,
        source=source,
    )
    require_exact_contract_version(
        row.get("outcome_value_target_contract_version"),
        expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
        name="outcome/value-target contract",
        source=source,
        regeneration_command=(
            "python -m scripts.self_play.generate ..."
        ),
    )

    require_physics_provenance(
        row,
        source=source,
        expected_physics_config=expected_physics_config,
    )

    action_space_config, action_layout = (
        require_topology_action_provenance(
            row,
            source=source,
            expected_action_space_config=expected_action_space_config,
            expected_action_layout=expected_action_layout,
        )
    )

    validate_example_outcome_contracts(
        pd.DataFrame([row]),
        source_path=source,
    )

    return action_space_config, action_layout


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
        "physical_objective_schema_version": (
            PHYSICAL_OBJECTIVE_SCHEMA_VERSION
        ),
        "outcome_objective_version": OUTCOME_OBJECTIVE_VERSION,
        "outcome_value_target_contract_version": (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        ),
    }


def _objective_contract_fingerprint() -> str:
    return sha256_json(_objective_contract())


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


def _episode_count(rows: list[dict[str, Any]]) -> int:
    return len({_episode_key(row) for row in rows})


def _scenario_count(rows: list[dict[str, Any]]) -> int:
    return len({str(row.get("scenario_id")) for row in rows})


def _require_sha256(value: object, *, name: str, source: str) -> str:
    text = str(value or "").strip().lower()

    if len(text) != 64 or any(
        character not in "0123456789abcdef"
        for character in text
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


class RollingReplayBuffer:
    """Rolling persistent replay buffer for pool-guided self-play."""

    def __init__(
        self,
        save_dir: str | Path,
        config: ReplayBufferConfig | None = None,
        physics_config: PhysicsConfig | None = None,
    ):
        self.save_dir = Path(save_dir)
        self.config = config or ReplayBufferConfig()
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.save_dir / "buffer_manifest.json"
        self.buffer: list[dict[str, Any]] = []
        self.producer_checkpoint: Path | None = None

        self.load()

    def __len__(self) -> int:
        return len(self.buffer)

    def _require_manifest_contracts(
        self,
        manifest: dict[str, Any],
    ) -> tuple[ActionSpaceConfig, tuple[ActionSlot, ...]]:
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
        require_outcome_objective_version(
            manifest,
            source=str(self.manifest_path),
        )
        require_exact_contract_version(
            manifest.get("outcome_value_target_contract_version"),
            expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
            name="outcome/value-target contract",
            source=str(self.manifest_path),
            regeneration_command="python -m scripts.self_play.generate ...",
        )

        expected_objective_fingerprint = (
            _objective_contract_fingerprint()
        )
        if (
            manifest.get("objective_contract_fingerprint")
            != expected_objective_fingerprint
        ):
            raise ValueError(
                "Replay objective contract fingerprint mismatch for "
                f"{self.manifest_path}."
            )

        require_physics_provenance(
            manifest,
            source=str(self.manifest_path),
            expected_physics_config=self.physics_config,
        )

        return require_topology_action_provenance(
            manifest,
            source=str(self.manifest_path),
        )

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
        _require_sha256(
            payload.get("sha256"),
            name="producer checkpoint hash",
            source=source,
        )
        require_exact_contract_version(
            payload.get("checkpoint_contract_version"),
            expected=CHECKPOINT_CONTRACT_VERSION,
            name="checkpoint contract",
            source=f"{source} producer checkpoint",
            regeneration_command=(
                "regenerate self-play with a compatible checkpoint"
            ),
        )

        model_type = str(payload.get("model_type", "")).strip()
        if not model_type:
            raise ValueError(
                f"Replay producer checkpoint model_type is missing: {source}"
            )

        matching_fields = (
            "physical_objective_schema_version",
            "outcome_objective_version",
            "outcome_value_target_contract_version",
            "state_feature_schema_fingerprint",
            "physics_config_fingerprint",
            "topology_action_config_fingerprint",
            "action_layout_fingerprint",
        )

        mismatches = [
            field
            for field in matching_fields
            if payload.get(field) != header.get(field)
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
        expected_action_layout: tuple[ActionSlot, ...],
    ) -> None:
        require_exact_contract_version(
            header.get("schema_version"),
            expected=REPLAY_BUFFER_SCHEMA_VERSION,
            name="replay-chunk schema",
            source=source,
            regeneration_command=(
                "regenerate self-play replay chunks"
            ),
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

        if (
            header.get("objective_contract_fingerprint")
            != _objective_contract_fingerprint()
        ):
            raise ValueError(
                f"Replay objective contract fingerprint mismatch for {source}."
            )

        require_physics_provenance(
            header,
            source=source,
            expected_physics_config=self.physics_config,
        )
        require_topology_action_provenance(
            header,
            source=source,
            expected_action_space_config=expected_action_space_config,
            expected_action_layout=expected_action_layout,
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
        expected_action_layout: tuple[ActionSlot, ...],
    ) -> None:
        iteration = _require_positive_integer(
            header.get("iteration"),
            name="chunk iteration",
            source=source,
        )
        expected_examples = _require_non_negative_integer(
            header.get("example_count"),
            name="chunk example_count",
            source=source,
        )
        expected_episodes = _require_non_negative_integer(
            header.get("episode_count"),
            name="chunk episode_count",
            source=source,
        )
        expected_scenarios = _require_non_negative_integer(
            header.get("scenario_count"),
            name="chunk scenario_count",
            source=source,
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
                expected_action_layout=expected_action_layout,
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
        producer_sha = (
            producer.get("sha256")
            if isinstance(producer, Mapping)
            else None
        )
        checks = {
            "iteration": header.get("iteration"),
            "n": header.get("example_count"),
            "episode_count": header.get("episode_count"),
            "scenario_count": header.get("scenario_count"),
            "objective_contract_fingerprint": header.get(
                "objective_contract_fingerprint"
            ),
            "physics_config_fingerprint": header.get(
                "physics_config_fingerprint"
            ),
            "topology_action_config_fingerprint": header.get(
                "topology_action_config_fingerprint"
            ),
            "action_layout_fingerprint": header.get(
                "action_layout_fingerprint"
            ),
            "producer_checkpoint_sha256": producer_sha,
        }
        mismatches = [
            name
            for name, expected_value in checks.items()
            if item.get(name) != expected_value
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
            manifest_action_layout,
        ) = self._require_manifest_contracts(manifest)

        files = manifest.get("files", [])

        if not isinstance(files, list):
            raise ValueError(
                f"Invalid replay buffer manifest: {self.manifest_path}"
            )

        selected_chunks: list[list[dict[str, Any]]] = []
        selected_count = 0

        for item in sorted(
            files,
            key=lambda value: int(
                value.get("iteration", 0)
                if isinstance(value, Mapping)
                else 0
            ),
            reverse=True,
        ):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"Invalid replay buffer manifest: {self.manifest_path}"
                )

            relative_path = item.get("path")
            if not relative_path:
                raise ValueError(
                    f"Replay manifest entry has no path: {self.manifest_path}"
                )

            file_path = self.save_dir / str(relative_path)
            self._require_chunk_hash(
                item,
                file_path=file_path,
            )
            header, rows = _read_jsonl_gz(file_path)
            self._require_manifest_item(
                item,
                header=header,
                file_path=file_path,
            )
            self._require_chunk_contracts(
                header,
                source=str(file_path),
                expected_action_space_config=(
                    manifest_action_space_config
                ),
                expected_action_layout=manifest_action_layout,
            )
            self._validate_chunk_contents(
                header,
                rows,
                source=str(file_path),
                expected_action_space_config=(
                    manifest_action_space_config
                ),
                expected_action_layout=manifest_action_layout,
            )

            selected_chunks.append(rows)
            selected_count += len(rows)

            if selected_count >= self.config.max_size:
                break

        self.buffer = [
            row
            for chunk in reversed(selected_chunks)
            for row in chunk
        ]
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
            size
            for size in sizes.values()
            if size > int(self.config.max_size)
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
                row
                for row in self.buffer
                if _episode_key(row) != oldest_episode
            ]

    def _buffer_action_contract(
        self,
    ) -> tuple[
        ActionSpaceConfig | None,
        tuple[ActionSlot, ...] | None,
    ]:
        if not self.buffer:
            return None, None

        return require_topology_action_provenance(
            self.buffer[0],
            source="replay buffer",
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
    ]:
        iteration = _require_positive_integer(
            iteration,
            name="replay iteration",
            source="replay buffer",
        )
        _validate_replay_batch_outcomes(
            examples,
            source=f"replay iteration {iteration}",
        )

        expected_action_space_config, expected_action_layout = (
            self._buffer_action_contract()
        )
        normalized: list[dict[str, Any]] = []

        for row in examples:
            item = _row_to_json_safe_dict(dict(row))
            row_action_space_config, row_action_layout = (
                _require_replay_row_contracts(
                    item,
                    source=f"replay iteration {iteration}",
                    expected_physics_config=self.physics_config,
                    expected_action_space_config=(
                        expected_action_space_config
                    ),
                    expected_action_layout=expected_action_layout,
                )
            )

            if expected_action_space_config is None:
                expected_action_space_config = row_action_space_config
                expected_action_layout = row_action_layout

            item["replay_iteration"] = iteration
            normalized.append(item)

        if not normalized:
            raise ValueError(
                f"Replay iteration {iteration} contains no examples."
            )

        self._require_episode_sizes(
            normalized,
            source=f"replay iteration {iteration}",
        )

        if (
            expected_action_space_config is None
            or expected_action_layout is None
        ):
            raise RuntimeError(
                "Replay examples have no topology action contract."
            )

        return (
            normalized,
            expected_action_space_config,
            expected_action_layout,
        )

    def add_examples(
        self,
        examples: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> None:
        """Add validated examples and evict only complete episodes."""

        normalized, _, _ = self._prepare_examples(
            examples,
            iteration=iteration,
        )
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
        expected_action_layout: tuple[ActionSlot, ...],
    ) -> dict[str, Any]:
        from grid_topology_ai.training.checkpoints import (
            load_checkpoint_payload,
        )

        checkpoint_path = Path(checkpoint_path)
        checkpoint = load_checkpoint_payload(
            checkpoint_path,
            map_location="cpu",
            expected_physics_config=self.physics_config,
        )
        require_topology_action_provenance(
            checkpoint,
            source=str(checkpoint_path),
            expected_action_space_config=expected_action_space_config,
            expected_action_layout=expected_action_layout,
        )

        return {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "checkpoint_contract_version": int(
                checkpoint["checkpoint_contract_version"]
            ),
            "model_type": str(checkpoint["model_type"]),
            "physical_objective_schema_version": int(
                checkpoint["physical_objective_schema_version"]
            ),
            "outcome_objective_version": int(
                checkpoint["outcome_objective_version"]
            ),
            "outcome_value_target_contract_version": int(
                checkpoint["outcome_value_target_contract_version"]
            ),
            "state_feature_schema_fingerprint": str(
                checkpoint["state_feature_schema_fingerprint"]
            ),
            "physics_config_fingerprint": str(
                checkpoint["physics_config_fingerprint"]
            ),
            "topology_action_config_fingerprint": str(
                checkpoint["topology_action_config_fingerprint"]
            ),
            "action_layout_fingerprint": str(
                checkpoint["action_layout_fingerprint"]
            ),
        }

    def _chunk_header(
        self,
        *,
        rows: list[dict[str, Any]],
        iteration: int,
        action_space_config: ActionSpaceConfig,
        action_layout: tuple[ActionSlot, ...],
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
            "objective_contract_fingerprint": (
                _objective_contract_fingerprint()
            ),
            **physics_provenance(self.physics_config),
            **topology_action_provenance(
                action_space_config,
                action_layout,
            ),
            "producer_checkpoint": producer_checkpoint,
        }

    def save_iteration_file(
        self,
        examples: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> Path:
        """Save one self-contained replay chunk atomically."""

        rows, action_space_config, action_layout = (
            self._prepare_examples(
                examples,
                iteration=iteration,
            )
        )
        if self.producer_checkpoint is None:
            raise RuntimeError(
                "Replay producer checkpoint is not configured."
            )

        producer = self._producer_checkpoint_provenance(
            self.producer_checkpoint,
            expected_action_space_config=action_space_config,
            expected_action_layout=action_layout,
        )
        header = self._chunk_header(
            rows=rows,
            iteration=iteration,
            action_space_config=action_space_config,
            action_layout=action_layout,
            producer_checkpoint=producer,
        )
        output_path = (
            self.save_dir
            / f"buffer_iter_{int(iteration):03d}.jsonl.gz"
        )
        _write_jsonl_gz(
            header=header,
            rows=rows,
            path=output_path,
        )
        return output_path

    def _chunk_manifest_item(
        self,
        file_path: Path,
        *,
        expected_action_space_config: ActionSpaceConfig,
        expected_action_layout: tuple[ActionSlot, ...],
    ) -> dict[str, Any]:
        header, rows = _read_jsonl_gz(file_path)
        self._require_chunk_contracts(
            header,
            source=str(file_path),
            expected_action_space_config=expected_action_space_config,
            expected_action_layout=expected_action_layout,
        )
        self._validate_chunk_contents(
            header,
            rows,
            source=str(file_path),
            expected_action_space_config=expected_action_space_config,
            expected_action_layout=expected_action_layout,
        )
        producer = header["producer_checkpoint"]

        return {
            "path": file_path.name,
            "sha256": sha256_file(file_path),
            "n": int(header["example_count"]),
            "episode_count": int(header["episode_count"]),
            "scenario_count": int(header["scenario_count"]),
            "iteration": int(header["iteration"]),
            "objective_contract_fingerprint": header[
                "objective_contract_fingerprint"
            ],
            "physics_config_fingerprint": header[
                "physics_config_fingerprint"
            ],
            "topology_action_config_fingerprint": header[
                "topology_action_config_fingerprint"
            ],
            "action_layout_fingerprint": header[
                "action_layout_fingerprint"
            ],
            "producer_checkpoint_sha256": producer["sha256"],
        }

    def save_manifest(self) -> None:
        """Write the retained chunk index, then remove stale files."""

        if not self.buffer:
            raise ValueError(
                "Cannot save replay manifest for an empty buffer."
            )

        topology_action_config, action_layout = (
            require_topology_action_provenance(
                self.buffer[0],
                source="replay buffer",
            )
        )
        chunk_paths = sorted(
            self.save_dir.glob("buffer_iter_*.jsonl.gz")
        )

        if not chunk_paths:
            raise ValueError(
                f"Replay directory contains no chunks: {self.save_dir}"
            )

        items = [
            self._chunk_manifest_item(
                file_path,
                expected_action_space_config=topology_action_config,
                expected_action_layout=action_layout,
            )
            for file_path in chunk_paths
        ]
        newest_iteration = max(
            int(item["iteration"])
            for item in items
        )
        oldest_retained_iteration = max(
            1,
            newest_iteration
            - int(self.config.retention_iterations)
            + 1,
        )
        retained_items = [
            item
            for item in items
            if int(item["iteration"])
            >= oldest_retained_iteration
        ]
        retained_paths = {
            str(item["path"])
            for item in retained_items
        }
        stale_paths = [
            file_path
            for file_path in chunk_paths
            if file_path.name not in retained_paths
        ]

        retained_buffer = [
            row
            for row in self.buffer
            if int(row.get("replay_iteration", -1))
            >= oldest_retained_iteration
        ]

        manifest = {
            "schema_version": REPLAY_BUFFER_SCHEMA_VERSION,
            "format_version": _REPLAY_MANIFEST_FORMAT_VERSION,
            **_objective_contract(),
            "objective_contract_fingerprint": (
                _objective_contract_fingerprint()
            ),
            **physics_provenance(self.physics_config),
            **topology_action_provenance(
                topology_action_config,
                action_layout,
            ),
            "config": asdict(self.config),
            "latest_iteration": int(newest_iteration),
            "oldest_retained_iteration": int(
                oldest_retained_iteration
            ),
            "total_examples_on_disk": int(
                sum(int(item["n"]) for item in retained_items)
            ),
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

        self.save_iteration_file(
            examples=examples,
            iteration=iteration,
        )
        self.add_examples(
            examples=examples,
            iteration=iteration,
        )
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

    def export_mixed_batch(
        self,
        output_path: str | Path,
        *,
        current_iteration: int,
        n_examples: int | None = None,
        fresh_fraction: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Export a mixed train batch as CSV."""

        if len(self.buffer) < int(self.config.min_size_to_train):
            raise ValueError(
                f"Replay buffer has only {len(self.buffer)} examples, "
                f"but min_size_to_train={self.config.min_size_to_train}."
            )

        output_path = Path(output_path)

        if n_examples is None:
            n_total = len(self.buffer)
        else:
            n_total = min(int(n_examples), len(self.buffer))

        if n_total <= 0:
            raise ValueError("n_examples must be positive.")

        if fresh_fraction is None:
            fresh_fraction = float(self.config.fresh_fraction)

        fresh_fraction = float(np.clip(fresh_fraction, 0.0, 1.0))

        rng_seed = int(
            self.config.random_seed
            if seed is None
            else seed
        )
        rng = np.random.default_rng(rng_seed)

        fresh, old = self._split_fresh_old(
            current_iteration=current_iteration,
        )

        target_fresh = int(round(n_total * fresh_fraction))
        target_old = n_total - target_fresh

        n_fresh = min(target_fresh, len(fresh))
        n_old = min(target_old, len(old))

        remaining = n_total - n_fresh - n_old

        if remaining > 0:
            fresh_left = len(fresh) - n_fresh
            take_more_fresh = min(remaining, fresh_left)
            n_fresh += take_more_fresh
            remaining -= take_more_fresh

        if remaining > 0:
            old_left = len(old) - n_old
            take_more_old = min(remaining, old_left)
            n_old += take_more_old

        if n_fresh + n_old <= 0:
            raise ValueError(
                "Could not sample any examples from replay buffer."
            )

        selected: list[dict[str, Any]] = []

        if n_fresh > 0:
            fresh_indices = rng.choice(
                len(fresh),
                size=n_fresh,
                replace=False,
            )
            selected.extend(
                fresh[int(index)]
                for index in fresh_indices
            )

        if n_old > 0:
            old_indices = rng.choice(
                len(old),
                size=n_old,
                replace=False,
            )
            selected.extend(
                old[int(index)]
                for index in old_indices
            )

        rng.shuffle(selected)

        df = pd.DataFrame(selected)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        metadata = {
            **physics_provenance(self.physics_config),
            "path": str(output_path),
            "n_examples": int(len(df)),
            "n_fresh": int(n_fresh),
            "n_old": int(n_old),
            "fresh_fraction_target": float(fresh_fraction),
            "fresh_fraction_actual": (
                float(n_fresh / len(df))
                if len(df) > 0
                else 0.0
            ),
            "current_iteration": int(current_iteration),
            "seed": int(rng_seed),
        }

        metadata_path = output_path.with_suffix(".metadata.json")
        _save_manifest(
            manifest=metadata,
            path=metadata_path,
        )

        return metadata

    def export_all(
        self,
        output_path: str | Path,
    ) -> Path:
        """Export the whole currently loaded buffer to CSV."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.buffer)
        df.to_csv(output_path, index=False)

        return output_path

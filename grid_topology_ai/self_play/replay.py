from __future__ import annotations

from dataclasses import asdict
from typing import Any

from grid_topology_ai.self_play import _replay_core as _core
from grid_topology_ai.self_play._replay_core import (
    RollingReplayBuffer as _CoreRollingReplayBuffer,
)
from grid_topology_ai.self_play.example_validation import (
    load_and_validate_examples_csv,
)
from grid_topology_ai.self_play.replay_error_sampling import (
    ReplayPredictionErrorMixin,
)


__all__ = (
    "RollingReplayBuffer",
    "load_and_validate_examples_csv",
)


class RollingReplayBuffer(
    ReplayPredictionErrorMixin,
    _CoreRollingReplayBuffer,
):
    """Persistent replay buffer with episode-balanced priority sampling."""

    def save_manifest(self) -> None:
        """Write retained chunk metadata without assuming one topology."""

        if not self.buffer:
            raise ValueError(
                "Cannot save replay manifest for an empty buffer."
            )

        topology_action_config, _ = (
            _core.require_topology_action_provenance(
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

        all_layout_fingerprint_set: set[str] = set()

        for file_path in chunk_paths:
            header, _ = _core._read_jsonl_gz(file_path)
            all_layout_fingerprint_set.update(
                _core._require_layout_fingerprints(
                    header.get("action_layout_fingerprints"),
                    source=str(file_path),
                )
            )

        all_layout_fingerprints = tuple(
            sorted(all_layout_fingerprint_set)
        )

        items = [
            self._chunk_manifest_item(
                file_path,
                expected_action_space_config=(
                    topology_action_config
                ),
                expected_layout_fingerprints=(
                    all_layout_fingerprints
                ),
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

        if not retained_buffer:
            raise RuntimeError(
                "Replay retention produced an empty buffer."
            )

        (
            retained_action_space_config,
            retained_action_layout,
        ) = _core.require_topology_action_provenance(
            retained_buffer[0],
            source="retained replay buffer",
        )

        retained_layout_fingerprints = tuple(
            sorted(
                {
                    str(fingerprint)
                    for item in retained_items
                    for fingerprint in item[
                        "action_layout_fingerprints"
                    ]
                }
            )
        )

        manifest = {
            "schema_version": _core.REPLAY_BUFFER_SCHEMA_VERSION,
            "format_version": _core._REPLAY_MANIFEST_FORMAT_VERSION,
            **_core._objective_contract(),
            "objective_contract_fingerprint": (
                _core._objective_contract_fingerprint()
            ),
            **_core.physics_provenance(self.physics_config),
            **_core.topology_action_provenance(
                retained_action_space_config,
                retained_action_layout,
            ),
            "policy_layout": (
                _core.require_branch_status_policy_layout(
                    retained_action_layout
                )
            ),
            "action_layout_fingerprints": list(
                retained_layout_fingerprints
            ),
            "config": asdict(self.config),
            "latest_iteration": int(newest_iteration),
            "oldest_retained_iteration": int(
                oldest_retained_iteration
            ),
            "total_examples_on_disk": int(
                sum(
                    int(item["n"])
                    for item in retained_items
                )
            ),
            "total_examples_loaded": int(
                len(retained_buffer)
            ),
            "files": retained_items,
        }

        _core._save_manifest(
            manifest=manifest,
            path=self.manifest_path,
        )
        self.buffer = retained_buffer

        for stale_path in stale_paths:
            stale_path.unlink()


def __getattr__(name: str) -> Any:
    """Keep private replay helpers available to validation tests."""

    return getattr(_core, name)

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from grid_topology_ai.config import ReplayBufferConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play import replay_priority
from grid_topology_ai.self_play.artifacts import sha256_file
from grid_topology_ai.self_play.replay import RollingReplayBuffer
from grid_topology_ai.self_play.replay_error_sampling import (
    PREDICTION_ERROR_SCHEMA_VERSION,
)


def _row(state_id: str, scenario_id: int = 1) -> dict[str, object]:
    return {
        "state_id": state_id,
        "episode_id": state_id,
        "scenario_id": scenario_id,
        "replay_iteration": 2,
        "outcome_class": "solved",
    }


def _buffer(tmp_path: Path) -> RollingReplayBuffer:
    return RollingReplayBuffer(
        save_dir=tmp_path / "replay",
        config=ReplayBufferConfig(
            max_size=20,
            min_size_to_train=1,
            fresh_fraction=0.5,
        ),
    )


def _report(checkpoint_sha: str, state_ids: list[str]) -> dict[str, Any]:
    entries = {
        state_id: {
            "value_error": float(index),
            "policy_kl_error": float(index) / 10.0,
        }
        for index, state_id in enumerate(state_ids)
    }
    return {
        "schema_version": PREDICTION_ERROR_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "example_count": len(entries),
        "mean_value_error": float(
            np.mean([item["value_error"] for item in entries.values()])
        ),
        "mean_policy_kl_error": float(
            np.mean([item["policy_kl_error"] for item in entries.values()])
        ),
        "entries": entries,
    }


def test_persisted_errors_drive_episode_priority(tmp_path: Path) -> None:
    buffer = _buffer(tmp_path)
    rows = [_row("neutral"), _row("difficult", 2)]
    buffer.buffer = rows
    report = _report("a" * 64, ["neutral", "difficult"])
    report["entries"]["difficult"]["value_error"] = 9.0
    buffer._record_prediction_errors(report, iteration=2)

    episodes, _ = buffer._episode_groups(
        rows,
        current_iteration=2,
        rng=np.random.default_rng(1),
    )
    priorities = {
        str(item["rows"][0]["state_id"]): float(item["priority"])
        for item in episodes
    }

    assert priorities["neutral"] == 1.0
    assert priorities["difficult"] == pytest.approx(1.09)


def test_export_refreshes_errors_once_per_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = _buffer(tmp_path)
    buffer.buffer = [_row("a"), _row("b", 2)]
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"one")
    buffer.set_producer_checkpoint(checkpoint)
    calls: list[str] = []

    def fake_score(**kwargs: Any) -> dict[str, Any]:
        state_ids = pd.read_csv(kwargs["examples_csv"])["state_id"].tolist()
        checkpoint_sha = sha256_file(kwargs["checkpoint_path"])
        calls.append(checkpoint_sha)
        return _report(checkpoint_sha, [str(value) for value in state_ids])

    monkeypatch.setattr(
        replay_priority,
        "score_replay_prediction_errors",
        fake_score,
    )

    first = buffer.export_mixed_batch(
        tmp_path / "first.csv",
        current_iteration=2,
        n_examples=2,
    )
    second = buffer.export_mixed_batch(
        tmp_path / "second.csv",
        current_iteration=2,
        n_examples=2,
    )
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"two")
    buffer.set_producer_checkpoint(replacement)
    third = buffer.export_mixed_batch(
        tmp_path / "third.csv",
        current_iteration=3,
        n_examples=2,
    )

    assert len(calls) == 2
    assert first["prediction_error_refresh"]["refreshed_examples"] == 2
    assert second["prediction_error_refresh"]["refreshed_examples"] == 0
    assert third["prediction_error_refresh"]["refreshed_examples"] == 2


class _Dataset:
    num_bus_features = 1
    num_branch_features = 1
    topology_action_config = object()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.samples = [
            {"state_id": "first", "policy": [1.0, 0.0], "value": 1.0},
            {"state_id": "second", "policy": [0.25, 0.75], "value": -1.0},
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class _Model:
    def __call__(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(
                [
                    [0.0, 0.0],
                    [math.log(0.25), math.log(0.75)],
                ]
            ),
            torch.tensor([0.5, -0.5]),
        )


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bus_features": torch.zeros(2, 1),
        "branch_features": torch.zeros(2, 1),
        "edge_index": torch.tensor([[0, 1], [0, 1]]),
        "edge_active_mask": torch.ones(2, dtype=torch.bool),
        "action_mask": torch.ones(2, 2, dtype=torch.bool),
        "node_batch": torch.tensor([0, 1]),
        "edge_batch": torch.tensor([0, 1]),
        "target_policy": torch.tensor([item["policy"] for item in samples]),
        "target_value": torch.tensor([item["value"] for item in samples]),
        "state_id": [item["state_id"] for item in samples],
    }


def test_graph_v2_scorer_computes_value_and_policy_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = tmp_path / "examples.csv"
    examples.write_text("placeholder\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = {
        "checkpoint_contract_version": 1,
        "model_type": "graph_policy_value_net_v2",
        "num_bus_features": 1,
        "num_branch_features": 1,
    }

    monkeypatch.setattr(
        replay_priority,
        "_resolve_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        replay_priority,
        "load_checkpoint_payload",
        lambda *args, **kwargs: checkpoint,
    )
    monkeypatch.setattr(
        replay_priority,
        "extract_normalization_stats",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(replay_priority, "GraphSelfPlayDataset", _Dataset)
    monkeypatch.setattr(
        replay_priority,
        "require_topology_action_provenance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        replay_priority,
        "_load_model",
        lambda *args, **kwargs: _Model(),
    )
    monkeypatch.setattr(replay_priority, "collate_graph_samples", _collate)

    report = replay_priority.score_replay_prediction_errors(
        examples_csv=examples,
        checkpoint_path=checkpoint_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )

    assert report["entries"]["first"]["value_error"] == pytest.approx(0.5)
    assert report["entries"]["second"]["value_error"] == pytest.approx(0.5)
    assert report["entries"]["first"]["policy_kl_error"] == pytest.approx(
        math.log(2.0)
    )
    assert report["entries"]["second"]["policy_kl_error"] == pytest.approx(
        0.0,
        abs=1e-7,
    )

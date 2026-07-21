from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    EVALUATION_METRICS_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.physical_objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
    physical_objective_contract,
)
from grid_topology_ai.self_play import iteration as iteration_module
from grid_topology_ai.self_play.checkpoint_state import BestState
from grid_topology_ai.self_play.iteration import (
    IterationRequest,
    _count_examples_csv,
    run_self_play_iteration,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths


class _FakeReplayBuffer:
    def __init__(self) -> None:
        self.rows = [1, 2, 3, 4]
        self.added: list[tuple[Path, int]] = []
        self.export_calls: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.rows)

    def add_and_save_from_csv(self, *, examples_csv: Path, iteration: int):
        self.added.append((examples_csv, iteration))
        return [{"fresh": True}, {"fresh": True}]

    def export_mixed_batch(
        self,
        *,
        output_path: Path,
        current_iteration: int,
        n_examples: int,
        fresh_fraction: float,
        seed: int,
    ) -> dict[str, int]:
        self.export_calls.append(
            {
                "output_path": output_path,
                "current_iteration": current_iteration,
                "n_examples": n_examples,
                "fresh_fraction": fresh_fraction,
                "seed": seed,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        provenance = physics_provenance(DEFAULT_PHYSICS_CONFIG)
        pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "physical_objective_schema_version": (
                        PHYSICAL_OBJECTIVE_SCHEMA_VERSION
                    ),
                    "outcome_value_target_contract_version": (
                        OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                    ),
                    "physics_config_contract_version": provenance[
                        "physics_config_contract_version"
                    ],
                    "physics_config": json.dumps(
                        provenance["physics_config"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "physics_config_fingerprint": provenance[
                        "physics_config_fingerprint"
                    ],
                }
                for scenario_id in (1, 2, 3)
            ]
        ).to_csv(output_path, index=False)
        return {"n_examples": int(n_examples), "n_fresh": 2, "n_old": 1}


def _metrics(
    solve_rate: float,
    *,
    pf_alg: int = 3,
    failed_scenarios: int = 0,
) -> dict[str, object]:
    physics_config = replace(DEFAULT_PHYSICS_CONFIG, pf_alg=pf_alg)
    return {
        "solve_rate": solve_rate,
        "failed_scenarios": failed_scenarios,
        "pf_alg": pf_alg,
        "task_config": {"pf_alg": pf_alg},
        "evaluation_metrics_contract_version": EVALUATION_METRICS_CONTRACT_VERSION,
        **physics_provenance(physics_config),
        "physical_objective_contract": physical_objective_contract(
            physics_config
        ),
    }


def _config() -> SelfPlayConfig:
    return SelfPlayConfig.from_mapping(
        {
            "run_name": "iteration_test",
            "seed": 10,
            "n_iterations": 3,
            "n_scenarios_per_iteration": 2,
            "epochs_per_iteration": 1,
            "pool": {
                "transitions_csv": "pool/transitions.csv",
                "raw_dir": "pool/raw",
                "metadata_path": "runs/iteration_test/pool_metadata.json",
            },
            "eval_csv": "eval/transitions.csv",
            "eval_raw_dir": "eval/raw",
            "bootstrap_checkpoint": "bootstrap.pt",
            "bootstrap_eval_metrics": "bootstrap_metrics.json",
            "checkpoint_dir": "runs/iteration_test",
            "best_checkpoint_path": "runs/iteration_test/checkpoints/best.pt",
            "best_metrics_path": "runs/iteration_test/checkpoints/best_metrics.json",
            "replay_buffer": {
                "max_size": 100,
                "min_size_to_train": 1,
                "fresh_fraction": 0.5,
                "random_seed": 10,
            },
            "generation": {"simulations": 1, "depth": 1, "max_steps": 1, "top_k": 1},
            "training": {
                "examples_per_iteration": 3,
                "batch_size": 2,
                "learning_rate": 0.001,
                "device": "cpu",
            },
            "evaluation": {"simulations": 1, "depth": 1, "max_steps": 1, "top_k": 1},
            "acceptance": {"metric": "solve_rate", "min_improvement": 0.0},
        }
    )


def _paths(tmp_path: Path) -> SelfPlayPaths:
    return SelfPlayPaths.from_config(_config(), tmp_path)


def _request(tmp_path: Path, *, iteration: int = 2) -> IterationRequest:
    paths = _paths(tmp_path)
    parent_checkpoint = tmp_path / "parent.pt"
    parent_checkpoint.write_bytes(b"parent")
    return IterationRequest(
        iteration=iteration,
        config=_config(),
        raw_config={"raw": "config"},
        paths=paths,
        parent_checkpoint=parent_checkpoint,
        parent_metrics=_metrics(0.5),
        pool_metadata={"scenarios": {"1": {}, "2": {}, "3": {}}},
        replay_buffer=_FakeReplayBuffer(),  # type: ignore[arg-type]
    )


def _install_stage_fakes(monkeypatch: pytest.MonkeyPatch, calls: list[str] | None = None) -> None:
    def fake_generate(**kwargs: Any) -> Path:
        if calls is not None:
            calls.append("generate")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        examples = output_dir / "examples.csv"
        examples.write_text("scenario_id,solved\n1,true\n2,false\n", encoding="utf-8")
        return examples

    def fake_train(**kwargs: Any) -> Path:
        if calls is not None:
            calls.append("train")
        checkpoint = Path(kwargs["output_dir"]) / "candidate_checkpoint.pt"
        torch.save(
            {
                "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
                "physical_objective_schema_version": (
                    PHYSICAL_OBJECTIVE_SCHEMA_VERSION
                ),
                "outcome_value_target_contract_version": (
                    OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                ),
                **physics_provenance(DEFAULT_PHYSICS_CONFIG),
            },
            checkpoint,
        )
        return checkpoint

    def fake_evaluate(**kwargs: Any) -> dict[str, object]:
        if calls is not None:
            calls.append("evaluate")
        return _metrics(0.6)

    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_generate", fake_generate)
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_train", fake_train)
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_evaluate", fake_evaluate)
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.sample_from_pool",
        lambda *, pool_metadata, n, seed: [2, 1],
    )
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.update_and_save_pool_metadata",
        lambda *, pool_metadata, episode_results, current_iter, path, **kwargs: {
            **pool_metadata,
            "updated": current_iter,
        },
    )


def test_iteration_request_is_frozen_and_slotted(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(FrozenInstanceError):
        request.iteration = 3  # type: ignore[misc]

    assert not hasattr(request, "__dict__")


def test_iteration_rejects_non_positive_number(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iteration"):
        _request(tmp_path, iteration=0)


def test_iteration_uses_seed_and_samples_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    captured: dict[str, object] = {}

    def fake_sample(*, pool_metadata, n, seed):
        captured["n"] = n
        captured["seed"] = seed
        return [1, 2]

    monkeypatch.setattr("grid_topology_ai.self_play.iteration.sample_from_pool", fake_sample)

    run_self_play_iteration(_request(tmp_path, iteration=2))

    assert captured == {"n": 2, "seed": 12}


def test_iteration_writes_selected_scenario_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    run_self_play_iteration(_request(tmp_path, iteration=2))

    selected = _paths(tmp_path).iteration_dir(2) / "selected_scenario_ids.txt"
    assert selected.read_text(encoding="utf-8") == "2\n1\n"


def test_iteration_runs_generation_training_evaluation_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_stage_fakes(monkeypatch, calls)

    run_self_play_iteration(_request(tmp_path))

    assert calls == ["generate", "train", "evaluate"]


def test_accepted_iteration_promotes_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    promoted_checkpoint = tmp_path / "best.pt"
    promoted_metrics = _metrics(0.8)
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.accept_candidate", lambda **kwargs: True)
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.promote_candidate",
        lambda **kwargs: BestState(checkpoint=promoted_checkpoint, metrics=promoted_metrics),
    )

    result = run_self_play_iteration(_request(tmp_path))

    assert result.status == "ACCEPTED"
    assert result.best_checkpoint == promoted_checkpoint
    assert result.best_metrics == promoted_metrics


def test_rejected_iteration_keeps_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.accept_candidate", lambda **kwargs: False)
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.promote_candidate",
        lambda **kwargs: pytest.fail("promote_candidate should not be called"),
    )
    request = _request(tmp_path)

    result = run_self_play_iteration(request)

    assert result.status == "REJECTED"
    assert result.best_checkpoint == request.parent_checkpoint
    assert result.best_metrics == dict(request.parent_metrics)


def test_metadata_is_saved_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.accept_candidate", lambda **kwargs: True)

    def fake_save_metadata(**kwargs: Any) -> Path:
        calls.append("metadata")
        return Path(kwargs["path"])

    def fake_promote(**kwargs: Any) -> BestState:
        calls.append("promote")
        return BestState(checkpoint=tmp_path / "best.pt", metrics=_metrics(0.7))

    monkeypatch.setattr("grid_topology_ai.self_play.iteration._save_iteration_metadata", fake_save_metadata)
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.promote_candidate", fake_promote)

    run_self_play_iteration(_request(tmp_path))

    assert calls == ["metadata", "promote"]


def test_pool_is_updated_for_rejected_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    called = {"updated": False}
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.accept_candidate", lambda **kwargs: False)

    def fake_update(**kwargs: Any) -> dict[str, Any]:
        called["updated"] = True
        return {"updated": True}

    monkeypatch.setattr("grid_topology_ai.self_play.iteration.update_and_save_pool_metadata", fake_update)

    result = run_self_play_iteration(_request(tmp_path))

    assert called["updated"] is True
    assert result.pool_metadata == {"updated": True}


def test_iteration_returns_learning_curve_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    result = run_self_play_iteration(_request(tmp_path))
    row = result.learning_curve_row

    assert row["iteration"] == 2
    assert row["accepted"] is True
    assert row["status"] == "ACCEPTED"
    assert row["candidate_metric"] == 0.6
    assert row["best_metric_after"] == 0.6
    assert row["n_sampled_scenarios"] == 2
    assert row["n_raw_examples"] == 2
    assert row["n_train_examples"] == 3
    assert row["n_fresh"] == 2
    assert row["n_old"] == 1
    assert row["candidate_checkpoint"] == str(result.candidate_checkpoint)
    assert row["best_checkpoint_after"] == str(result.best_checkpoint)
    assert row["candidate_solve_rate"] == 0.6
    assert row["best_solve_rate"] == 0.6


def test_parent_metrics_are_not_mutated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    parent_metrics = _metrics(0.5)
    request = _request(tmp_path)
    request = IterationRequest(
        iteration=request.iteration,
        config=request.config,
        raw_config=request.raw_config,
        paths=request.paths,
        parent_checkpoint=request.parent_checkpoint,
        parent_metrics=parent_metrics,
        pool_metadata=request.pool_metadata,
        replay_buffer=request.replay_buffer,
    )

    run_self_play_iteration(request)

    assert parent_metrics == _metrics(0.5)


def test_iteration_stops_before_training_when_replay_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_generate(**kwargs: Any) -> Path:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        examples = output_dir / "examples.csv"
        examples.write_text("scenario_id\n1\n", encoding="utf-8")
        return examples

    class FailingReplay(_FakeReplayBuffer):
        def add_and_save_from_csv(self, *, examples_csv: Path, iteration: int):
            raise ValueError("invalid examples")

    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_generate", fake_generate)
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_train", lambda **kwargs: pytest.fail("run_train called"))
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_evaluate", lambda **kwargs: pytest.fail("run_evaluate called"))
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.update_and_save_pool_metadata",
        lambda **kwargs: pytest.fail("pool update called"),
    )
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.sample_from_pool",
        lambda *, pool_metadata, n, seed: [1, 2],
    )
    base = _request(tmp_path)
    request = IterationRequest(
        iteration=base.iteration,
        config=base.config,
        raw_config=base.raw_config,
        paths=base.paths,
        parent_checkpoint=base.parent_checkpoint,
        parent_metrics=base.parent_metrics,
        pool_metadata=base.pool_metadata,
        replay_buffer=FailingReplay(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="invalid examples"):
        run_self_play_iteration(request)

    assert not (_paths(tmp_path).iteration_dir(2) / "metadata.json").exists()


def test_count_examples_csv_returns_row_count(tmp_path: Path) -> None:
    path = tmp_path / "examples.csv"
    path.write_text("scenario_id,solved\n1,true\n2,false\n", encoding="utf-8")

    assert _count_examples_csv(path) == 2


def test_count_examples_csv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _count_examples_csv(tmp_path / "missing.csv")


def test_count_examples_csv_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "examples.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no readable columns"):
        _count_examples_csv(path)


def test_count_examples_csv_rejects_header_only_file(tmp_path: Path) -> None:
    path = tmp_path / "examples.csv"
    path.write_text("scenario_id,solved\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no rows"):
        _count_examples_csv(path)


def test_count_examples_csv_rejects_malformed_csv(tmp_path: Path) -> None:
    path = tmp_path / "examples.csv"
    path.write_text('scenario_id,solved\n1,"unterminated\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse examples CSV"):
        _count_examples_csv(path)


def test_count_examples_csv_does_not_hide_unexpected_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "examples.csv"
    path.write_text("scenario_id,solved\n1,true\n", encoding="utf-8")

    def fail(path: object) -> object:
        raise RuntimeError("unexpected pandas failure")

    monkeypatch.setattr(iteration_module.pd, "read_csv", fail)

    with pytest.raises(RuntimeError, match="unexpected pandas failure"):
        _count_examples_csv(path)


def test_iteration_rejects_candidate_metrics_pf_alg_before_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.run_evaluate",
        lambda **kwargs: _metrics(0.9, pf_alg=1),
    )
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.accept_candidate",
        lambda **kwargs: pytest.fail("accept_candidate should not run"),
    )
    request = _request(tmp_path)

    with pytest.raises(ValueError, match="PF_ALG"):
        run_self_play_iteration(request)

    assert request.parent_checkpoint.read_bytes() == b"parent"
    assert not (_paths(tmp_path).iteration_completion_marker(request.iteration)).exists()

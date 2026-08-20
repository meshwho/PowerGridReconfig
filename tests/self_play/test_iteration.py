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
from grid_topology_ai.config.pool import CurriculumSamplingConfig
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    EVALUATION_METRICS_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.physics.objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
    physical_objective_contract,
)
from grid_topology_ai.self_play import iteration as iteration_module
from grid_topology_ai.self_play.checkpoints import BestState
from grid_topology_ai.self_play.iteration import (
    IterationRequest,
    _count_examples_csv,
    run_self_play_iteration,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.pool_sampling import (
    _priority_weights,
    sample_curriculum_from_pool,
)
from grid_topology_ai.evaluation.paired_results import (
    PAIRED_OUTCOME_FIELDS,
)


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
    pf_alg: int = 1,
    failed_scenarios: int = 0,
    constrained_solve_rate: float | None = None,
    primary_policy_mode: str = "ungated",
) -> dict[str, object]:
    physics_config = replace(
        DEFAULT_PHYSICS_CONFIG,
        pf_alg=pf_alg,
    )
    provenance = physics_provenance(physics_config)
    constrained_rate = (
        solve_rate
        if constrained_solve_rate is None
        else constrained_solve_rate
    )
    ungated_metrics = {
        "solve_rate": solve_rate,
        "failed_scenarios": failed_scenarios,
        "physically_secure_rate_requested": solve_rate,
    }
    constrained_metrics = {
        "solve_rate": constrained_rate,
        "failed_scenarios": failed_scenarios,
        "physically_secure_rate_requested": constrained_rate,
    }
    return {
        **ungated_metrics,
        "pf_alg": pf_alg,
        "task_config": {
            "pf_alg": pf_alg,
            "primary_policy_mode": primary_policy_mode,
        },
        "primary_policy_mode": primary_policy_mode,
        "mode_metrics": {
            "ungated": ungated_metrics,
            "constrained": constrained_metrics,
        },
        "evaluation_metrics_contract_version": EVALUATION_METRICS_CONTRACT_VERSION,
        **provenance,
        "physical_objective_contract": physical_objective_contract(
            physics_config
        ),
        "run_info": {
            "checkpoint_sha256": f"checkpoint-{solve_rate}",
            "transitions_sha256": "eval-transitions",
            "raw_data_sha256": "eval-raw-data",
            "scenario_ids_sha256": "eval-scenarios",
            "task_config_sha256": "eval-task-config",
            "physics_config_fingerprint": provenance[
                "physics_config_fingerprint"
            ],
            "evaluation_metrics_contract_version": (
                EVALUATION_METRICS_CONTRACT_VERSION
            ),
            "git_revision": "test-revision",
            "git_dirty": False,
        },
    }


def _write_evaluation_results(
    *,
    output_dir: Path,
    output_csv_name: str,
    scenario_ids: tuple[int, ...],
    secure: bool,
) -> Path:
    rows: list[dict[str, object]] = []

    for scenario_id in scenario_ids:
        row: dict[str, object] = {
            "scenario_id": int(scenario_id),
            "policy_mode": "ungated",
            "evaluation_failed": False,
        }

        for field in PAIRED_OUTCOME_FIELDS:
            if field != "evaluation_success":
                row[field] = secure

        rows.append(row)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = output_dir / output_csv_name
    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    return output_path


def test_iteration_rejects_mismatched_evaluation_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    def fake_evaluate(**kwargs: Any) -> dict[str, object]:
        checkpoint = Path(kwargs["checkpoint"])

        metrics = _metrics(
            0.5
            if checkpoint.name == "parent.pt"
            else 0.6
        )

        if checkpoint.name != "parent.pt":
            run_info = dict(metrics["run_info"])
            run_info["transitions_sha256"] = "changed-eval-csv"
            metrics["run_info"] = run_info

        return metrics

    monkeypatch.setattr(
        iteration_module,
        "run_evaluate",
        fake_evaluate,
    )
    monkeypatch.setattr(
        iteration_module,
        "accept_candidate",
        lambda **kwargs: pytest.fail(
            "accept_candidate should not run"
        ),
    )

    with pytest.raises(
        ValueError,
        match="transitions_sha256",
    ):
        run_self_play_iteration(_request(tmp_path))


def test_iteration_rejects_missing_evaluation_run_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    def fake_evaluate(**kwargs: Any) -> dict[str, object]:
        metrics = _metrics(0.5)
        metrics.pop("run_info")
        return metrics

    monkeypatch.setattr(
        iteration_module,
        "run_evaluate",
        fake_evaluate,
    )

    with pytest.raises(
        ValueError,
        match="missing run_info",
    ):
        run_self_play_iteration(_request(tmp_path))


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
                "curriculum": {
                    "stale_after_iterations": 7,
                    "priority_floor": 0.01,
                },
            },
            "eval_csv": "eval/transitions.csv",
            "eval_raw_dir": "eval/raw",
            "final_test_csv": "final_test/transitions.csv",
            "final_test_raw_dir": "final_test/raw",
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
            "acceptance": {
                "metric": "solve_rate",
                "min_improvement": 0.0,
                "confidence_level": 0.95,
                "bootstrap_samples": 200,
            },
        }
    )


def _paths(tmp_path: Path) -> SelfPlayPaths:
    return SelfPlayPaths.from_config(_config(), tmp_path)


def _request(tmp_path: Path, *, iteration: int = 2) -> IterationRequest:
    paths = _paths(tmp_path)
    paths.eval_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pd.DataFrame(
        {
            "scenario_id": [3, 1, 2],
        }
    ).to_csv(
        paths.eval_csv,
        index=False,
    )

    paths.eval_raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    parent_checkpoint = tmp_path / "parent.pt"
    parent_checkpoint.write_bytes(b"parent")
    return IterationRequest(
        iteration=iteration,
        config=_config(),
        raw_config={"raw": "config"},
        paths=paths,
        parent_checkpoint=parent_checkpoint,
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

    def fake_evaluate(
        **kwargs: Any,
    ) -> dict[str, object]:
        checkpoint = Path(
            kwargs["checkpoint"]
        )
        is_candidate = (
            checkpoint.name != "parent.pt"
        )

        _write_evaluation_results(
            output_dir=Path(
                kwargs["output_dir"]
            ),
            output_csv_name=(
                kwargs["config"].output_csv_name
            ),
            scenario_ids=tuple(
                kwargs["scenario_ids"]
            ),
            secure=is_candidate,
        )

        if calls is not None:
            calls.append(
                f"evaluate:{checkpoint.name}"
            )

        if checkpoint.name == "parent.pt":
            return _metrics(0.5)

        return _metrics(0.6)

    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_generate", fake_generate)
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_train", fake_train)
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.run_evaluate", fake_evaluate)
    monkeypatch.setattr(
        "grid_topology_ai.self_play.iteration.sample_from_pool",
        lambda *, pool_metadata, n, seed, current_iter, config: [2, 1],
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

    def fake_sample(
        *,
        pool_metadata,
        n,
        seed,
        current_iter,
        config,
    ):
        captured["pool_metadata"] = pool_metadata
        captured["n"] = n
        captured["seed"] = seed
        captured["current_iter"] = current_iter
        captured["config"] = config
        return [1, 2]

    monkeypatch.setattr("grid_topology_ai.self_play.iteration.sample_from_pool", fake_sample)

    request = _request(tmp_path, iteration=2)
    run_self_play_iteration(request)

    expected_seed = iteration_module._self_play_seeds(
        base_seed=10,
        iteration=2,
    ).scenario_sampling
    assert captured["pool_metadata"] is request.pool_metadata
    assert captured["n"] == 2
    assert captured["seed"] == expected_seed
    assert captured["current_iter"] == 2
    assert captured["config"] is request.config.pool.curriculum


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

    assert calls == [
        "generate",
        "train",
        "evaluate:parent.pt",
        "evaluate:candidate_checkpoint.pt",
    ]


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
    assert result.parent_metrics["solve_rate"] == 0.5
    assert result.best_metrics["solve_rate"] == 0.5


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
    captured: dict[str, object] = {}
    monkeypatch.setattr("grid_topology_ai.self_play.iteration.accept_candidate", lambda **kwargs: False)

    def fake_update(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"updated": True}

    monkeypatch.setattr("grid_topology_ai.self_play.iteration.update_and_save_pool_metadata", fake_update)

    request = _request(tmp_path)
    result = run_self_play_iteration(request)

    assert captured["current_iter"] == request.iteration
    assert captured["selected_scenario_ids"] == [2, 1]
    assert captured["stale_after_iterations"] == 7
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
    assert row["aggregate_gates_passed"] is True
    assert row["confidence_gates_passed"] is True
    assert row["paired_scenario_count"] == 3
    assert row["paired_confidence_level"] == 0.95
    assert row["paired_bootstrap_samples"] == 200
    assert row["physically_secure_rate_difference"] == 1.0
    assert row["physically_secure_ci_lower"] == 1.0
    assert row["physically_secure_ci_upper"] == 1.0


def test_parent_is_reevaluated_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    captured: dict[str, object] = {}

    def fake_accept_candidate(
        *,
        new_metrics,
        best_metrics,
        config,
    ) -> bool:
        captured["candidate"] = dict(new_metrics)
        captured["parent"] = dict(best_metrics)
        return True

    monkeypatch.setattr(
        iteration_module,
        "accept_candidate",
        fake_accept_candidate,
    )

    result = run_self_play_iteration(_request(tmp_path))

    assert captured["candidate"]["solve_rate"] == 0.6
    assert captured["parent"]["solve_rate"] == 0.5
    assert result.parent_metrics["solve_rate"] == 0.5


def test_constrained_gain_does_not_override_ungated_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    captured: dict[str, dict[str, object]] = {}

    def fake_evaluate(**kwargs: Any) -> dict[str, object]:
        checkpoint = Path(kwargs["checkpoint"])
        is_candidate = checkpoint.name != "parent.pt"
        _write_evaluation_results(
            output_dir=Path(kwargs["output_dir"]),
            output_csv_name=kwargs["config"].output_csv_name,
            scenario_ids=tuple(kwargs["scenario_ids"]),
            secure=not is_candidate,
        )
        if is_candidate:
            return _metrics(0.60, constrained_solve_rate=0.90)
        return _metrics(0.70, constrained_solve_rate=0.72)

    def fake_accept_candidate(
        *,
        new_metrics,
        best_metrics,
        config,
    ) -> bool:
        captured["candidate"] = dict(new_metrics)
        captured["parent"] = dict(best_metrics)
        return False

    monkeypatch.setattr(iteration_module, "run_evaluate", fake_evaluate)
    monkeypatch.setattr(
        iteration_module,
        "accept_candidate",
        fake_accept_candidate,
    )
    monkeypatch.setattr(
        iteration_module,
        "passes_confidence_gates",
        lambda **kwargs: True,
    )

    result = run_self_play_iteration(_request(tmp_path))

    assert result.accepted is False
    assert captured["candidate"]["solve_rate"] == pytest.approx(0.60)
    assert captured["parent"]["solve_rate"] == pytest.approx(0.70)
    assert captured["candidate"][
        "physically_secure_rate_requested"
    ] == pytest.approx(0.60)
    assert result.candidate_metrics["mode_metrics"]["constrained"][
        "solve_rate"
    ] == pytest.approx(0.90)


def test_parent_and_candidate_use_the_same_evaluation_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    evaluation_calls: list[tuple[int, ...]] = []

    def fake_evaluate(**kwargs: Any) -> dict[str, object]:
        checkpoint = Path(kwargs["checkpoint"])
        is_candidate = (
            checkpoint.name != "parent.pt"
        )
        evaluation_calls.append(
            tuple(kwargs["scenario_ids"])
        )

        _write_evaluation_results(
            output_dir=Path(
                kwargs["output_dir"]
            ),
            output_csv_name=(
                kwargs["config"].output_csv_name
            ),
            scenario_ids=tuple(
                kwargs["scenario_ids"]
            ),
            secure=is_candidate,
        )

        checkpoint = Path(kwargs["checkpoint"])
        return _metrics(
            0.5
            if checkpoint.name == "parent.pt"
            else 0.6
        )

    monkeypatch.setattr(
        iteration_module,
        "run_evaluate",
        fake_evaluate,
    )

    run_self_play_iteration(_request(tmp_path))

    assert evaluation_calls == [
        (1, 2, 3),
        (1, 2, 3),
    ]


def test_iteration_saves_paired_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    result = run_self_play_iteration(
        _request(tmp_path)
    )

    comparison_path = (
        _paths(tmp_path)
        .iteration_dir(result.iteration)
        / "evaluation"
        / "comparison.json"
    )

    comparison = json.loads(
        comparison_path.read_text(
            encoding="utf-8"
        )
    )

    assert comparison["policy_mode"] == "ungated"
    assert comparison["scenario_count"] == 3
    assert (
        "physically_secure"
        in comparison["metrics"]
    )
    assert (
        comparison["metrics"]
        ["physically_secure"]
        ["rate_difference"] == 1.0
    )
    assert (
        comparison["metrics"]
        ["physically_secure"]
        ["ci_lower"]
        == 1.0
    )


def test_confidence_gate_can_reject_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    monkeypatch.setattr(
        iteration_module,
        "accept_candidate",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        iteration_module,
        "passes_confidence_gates",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        iteration_module,
        "promote_candidate",
        lambda **kwargs: pytest.fail(
            "candidate should not be promoted"
        ),
    )

    result = run_self_play_iteration(
        _request(tmp_path)
    )

    assert result.accepted is False
    assert result.status == "REJECTED"


def test_rejected_candidate_refreshes_best_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)

    monkeypatch.setattr(
        iteration_module,
        "accept_candidate",
        lambda **kwargs: False,
    )

    result = run_self_play_iteration(_request(tmp_path))

    best_metrics_path = _paths(tmp_path).best_metrics
    saved_metrics = json.loads(
        best_metrics_path.read_text(encoding="utf-8")
    )

    assert result.best_metrics["solve_rate"] == 0.5
    assert saved_metrics["solve_rate"] == 0.5


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
        lambda **kwargs: [1, 2],
    )
    base = _request(tmp_path)
    request = IterationRequest(
        iteration=base.iteration,
        config=base.config,
        raw_config=base.raw_config,
        paths=base.paths,
        parent_checkpoint=base.parent_checkpoint,
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
        lambda **kwargs: (
            _metrics(0.5)
            if Path(kwargs["checkpoint"]).name == "parent.pt"
            else _metrics(0.9, pf_alg=3)
        ),
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


def _curriculum_scenario(difficulty: str) -> dict[str, object]:
    return {
        "difficulty_class": difficulty,
        "times_attempted": 1,
        "times_solved": 0,
        "solve_rate": 0.0,
        "last_attempted_iter": 0,
        "learning_progress": 0.0,
        "uncertainty": 1.0,
        "staleness": 0.0,
        "priority": 0.05,
    }


@pytest.mark.parametrize("seed", [0, 1, 11, 91])
def test_curriculum_sampling_reserves_required_quota_overlap(
    seed: int,
) -> None:
    metadata = {
        "schema_version": 3,
        "last_updated_iteration": 0,
        "scenarios": {
            "1": _curriculum_scenario("medium"),
            "2": _curriculum_scenario("medium"),
            "3": _curriculum_scenario("hard"),
            "4": _curriculum_scenario("hard"),
        },
    }
    config = CurriculumSamplingConfig(
        never_solved_min_fraction=2 / 3,
        hard_min_fraction=2 / 3,
        simple_max_fraction=1.0,
        frontier_max_fraction=1.0,
    )

    sample = sample_curriculum_from_pool(
        metadata,
        n=3,
        seed=seed,
        current_iter=1,
        config=config,
    )

    assert sample.report["never_solved"]["shortfall"] == 0
    assert sample.report["hard"]["shortfall"] == 0
    assert sample.report["never_solved"]["selected"] >= 2
    assert sample.report["hard"]["selected"] >= 2


def test_priority_weights_use_the_configured_floor() -> None:
    scenarios = {
        "1": {"priority": 0.01},
        "2": {"priority": 0.11},
        "3": {"priority": float("nan")},
    }

    weights = _priority_weights(
        scenarios,
        ["1", "2", "3"],
        priority_floor=0.01,
    )

    assert weights.tolist() == pytest.approx([0.01, 0.11, 0.01])

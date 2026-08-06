from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from grid_topology_ai.config import AcceptanceConfig
from grid_topology_ai.config.checkpoint_selection import (
    CheckpointSelectionConfig,
)
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.evaluation.paired_results import PAIRED_OUTCOME_FIELDS
from grid_topology_ai.evaluation.policy_comparison import (
    PolicyMode,
    require_primary_policy_mode,
)
from grid_topology_ai.self_play import checkpoint_arena
from grid_topology_ai.self_play import iteration as iteration_module
from grid_topology_ai.self_play.acceptance import accept_candidate
from grid_topology_ai.self_play.iteration import (
    _metrics_for_policy_mode,
    run_self_play_iteration,
)


def _load_test_module(name: str) -> ModuleType:
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(
        f"_contract_helpers_{path.stem}",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_acceptance_helpers = _load_test_module("test_acceptance.py")
_iteration_helpers = _load_test_module("test_iteration.py")


def _evaluation_metrics(
    ungated_rate: float,
    constrained_rate: float,
) -> dict[str, object]:
    def mode_metrics(rate: float) -> dict[str, object]:
        return _acceptance_helpers._metrics(
            requested_scenarios=100,
            physically_secure_count=int(round(rate * 100)),
            task_config={
                "pf_alg": 3,
                "primary_policy_mode": "ungated",
            },
        )

    ungated = mode_metrics(ungated_rate)
    constrained = mode_metrics(constrained_rate)
    metrics = {
        **ungated,
        "primary_policy_mode": "ungated",
        "mode_metrics": {
            "ungated": ungated,
            "constrained": constrained,
        },
        "ungated_physically_secure_rate_requested": ungated_rate,
        "constrained_physically_secure_rate_requested": constrained_rate,
        "continuation_gate_gain": constrained_rate - ungated_rate,
        "run_info": {
            "checkpoint_sha256": f"checkpoint-{ungated_rate}",
            "transitions_sha256": "eval-transitions",
            "raw_data_sha256": "eval-raw-data",
            "scenario_ids_sha256": "eval-scenarios",
            "task_config_sha256": "eval-task-config",
            "physics_config_fingerprint": ungated[
                "physics_config_fingerprint"
            ],
            "evaluation_metrics_contract_version": ungated[
                "evaluation_metrics_contract_version"
            ],
            "git_revision": "test-revision",
            "git_dirty": False,
        },
    }
    return metrics


def _ungated_view(
    metrics: dict[str, object],
    *,
    source: str,
) -> dict[str, object]:
    return _metrics_for_policy_mode(
        metrics,
        PolicyMode.UNGATED,
        source=source,
    )


def test_primary_headline_must_match_nested_ungated_metrics() -> None:
    metrics = {
        "primary_policy_mode": "ungated",
        "solve_rate": 0.90,
        "mode_metrics": {
            "ungated": {
                "solve_rate": 0.70,
            }
        },
    }

    with pytest.raises(
        ValueError,
        match="headline metrics do not match",
    ):
        require_primary_policy_mode(
            metrics,
            PolicyMode.UNGATED,
            source="test metrics",
        )


def test_real_acceptance_rejects_constrained_only_improvement() -> None:
    parent = _evaluation_metrics(0.70, 0.72)
    candidate = _evaluation_metrics(0.60, 0.90)

    assert accept_candidate(
        new_metrics=_ungated_view(candidate, source="candidate"),
        best_metrics=_ungated_view(parent, source="parent"),
        config=AcceptanceConfig(),
    ) is False


def test_real_acceptance_uses_ungated_improvement_when_constrained_worsens() -> None:
    parent = _evaluation_metrics(0.70, 0.90)
    candidate = _evaluation_metrics(0.80, 0.81)

    assert accept_candidate(
        new_metrics=_ungated_view(candidate, source="candidate"),
        best_metrics=_ungated_view(parent, source="parent"),
        config=AcceptanceConfig(),
    ) is True


def _arena_candidate(path: Path, *, loss: float) -> dict[str, object]:
    return {
        "path": path,
        "sha256": checkpoint_arena.sha256_file(path),
        "payload": {},
        "val_metrics": {
            "loss": loss,
            "policy_loss": loss,
            "value_loss": loss,
            "value_calibration_error": loss,
        },
        "training_selector": "validation_loss",
        "saved_epoch": 1,
        "ranking_sources": [],
    }


def _arena_metrics(
    *,
    ungated: float,
    constrained: float,
    primary_policy_mode: str = "ungated",
) -> dict[str, object]:
    primary_rate = (
        constrained
        if primary_policy_mode == "constrained"
        else ungated
    )
    return {
        "primary_policy_mode": primary_policy_mode,
        "physically_secure_rate_requested": primary_rate,
        "failed_scenarios": 0,
        "mode_metrics": {
            "ungated": {
                "physically_secure_rate_requested": ungated,
                "failed_scenarios": 0,
            },
            "constrained": {
                "physically_secure_rate_requested": constrained,
                "failed_scenarios": 0,
            },
        },
    }


def test_arena_rejects_constrained_primary_runtime_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate_checkpoint.pt"
    candidate.write_bytes(b"candidate")
    tuning_csv = tmp_path / "tuning.csv"
    pd.DataFrame({"scenario_id": [1]}).to_csv(tuning_csv, index=False)
    tuning_raw_dir = tmp_path / "raw"
    tuning_raw_dir.mkdir()

    monkeypatch.setattr(
        checkpoint_arena,
        "_validate_tuning_independence",
        lambda **kwargs: (1,),
    )
    monkeypatch.setattr(
        checkpoint_arena,
        "_load_candidates",
        lambda **kwargs: [_arena_candidate(candidate, loss=0.1)],
    )

    config = CheckpointSelectionConfig(
        enabled=True,
        tuning_csv=tuning_csv,
        tuning_raw_dir=tuning_raw_dir,
        max_candidates=2,
    )

    with pytest.raises(
        ValueError,
        match="primary policy mode mismatch",
    ):
        checkpoint_arena.select_checkpoint_in_tuning_arena(
            canonical_checkpoint=candidate,
            project_root=tmp_path,
            output_dir=tmp_path / "selection",
            config=config,
            physics_config=DEFAULT_PHYSICS_CONFIG,
            tuning_csv=tuning_csv,
            tuning_raw_dir=tuning_raw_dir,
            excluded_csvs={},
            evaluate=lambda **kwargs: _arena_metrics(
                ungated=0.70,
                constrained=0.95,
                primary_policy_mode="constrained",
            ),
        )


def _install_iteration_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        iteration_module,
        "_self_play_exploration_metrics",
        lambda examples: {
            "steps": len(examples),
            "sampled_steps": 0,
            "sample_fraction": 0.0,
            "mean_selection_temperature": 0.0,
            "mean_policy_target_entropy": 0.0,
            "mean_policy_target_normalized_entropy": 0.0,
            "mean_mcts_legal_action_count": 0.0,
            "mean_mcts_considered_action_count": 0.0,
            "mean_mcts_visited_action_count": 0.0,
            "mean_mcts_action_coverage": 0.0,
            "min_mcts_action_coverage": 0.0,
            "mean_mcts_visited_action_coverage": 0.0,
            "min_mcts_visited_action_coverage": 0.0,
        },
    )

    def split(**kwargs: Any):
        replay_buffer = kwargs["replay_buffer"]
        batch_path = Path(kwargs["train_batch_path"])
        batch_metadata = replay_buffer.export_mixed_batch(
            output_path=batch_path,
            current_iteration=kwargs["iteration"],
            n_examples=kwargs["n_examples"],
            fresh_fraction=kwargs["fresh_fraction"],
            seed=kwargs["sampling_seed"],
        )
        batch = pd.read_csv(batch_path)
        train = batch.iloc[:-1].copy()
        validation = batch.iloc[-1:].copy()
        train.to_csv(kwargs["train_examples_path"], index=False)
        validation.to_csv(kwargs["validation_examples_path"], index=False)
        Path(kwargs["metadata_path"]).write_text("{}\n", encoding="utf-8")
        return batch_metadata, {
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_scenarios": train["scenario_id"].nunique(),
            "validation_scenarios": validation["scenario_id"].nunique(),
        }

    monkeypatch.setattr(
        iteration_module,
        "prepare_physical_iteration_split",
        split,
    )


def _write_paired_rows(
    *,
    output_dir: Path,
    output_csv_name: str,
    scenario_ids: tuple[int, ...],
    secure_count: int,
) -> None:
    rows: list[dict[str, object]] = []
    for index, scenario_id in enumerate(scenario_ids):
        secure = index < secure_count
        row: dict[str, object] = {
            "scenario_id": scenario_id,
            "policy_mode": "ungated",
            "evaluation_failed": False,
        }
        for field in PAIRED_OUTCOME_FIELDS:
            if field != "evaluation_success":
                row[field] = secure
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output_dir / output_csv_name,
        index=False,
    )


def test_iteration_rejects_candidate_after_ungated_arena_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _iteration_helpers._request(tmp_path)
    _iteration_helpers._install_stage_fakes(monkeypatch)
    _install_iteration_adapters(monkeypatch)
    calls: list[str] = []

    monkeypatch.setattr(
        iteration_module,
        "passes_confidence_gates",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        iteration_module,
        "promote_candidate",
        lambda **kwargs: pytest.fail(
            "constrained-only improvement must not be promoted"
        ),
    )
    monkeypatch.setattr(
        checkpoint_arena,
        "_validate_tuning_independence",
        lambda **kwargs: (101,),
    )

    def load_candidates(**kwargs: Any) -> list[dict[str, object]]:
        canonical = Path(kwargs["canonical_checkpoint"])
        alternate = canonical.with_name(
            f"{canonical.stem}_best_policy{canonical.suffix}"
        )
        return [
            _arena_candidate(canonical, loss=0.1),
            _arena_candidate(alternate, loss=0.2),
        ]

    monkeypatch.setattr(
        checkpoint_arena,
        "_load_candidates",
        load_candidates,
    )
    monkeypatch.setattr(
        checkpoint_arena,
        "_annotate_selected_checkpoint",
        lambda **kwargs: shutil.copy2(
            kwargs["source"],
            kwargs["destination"],
        ),
    )

    def train_with_arena(**kwargs: Any) -> Path:
        calls.append("training")
        output_dir = Path(kwargs["output_dir"])
        canonical = output_dir / "candidate_checkpoint.pt"
        alternate = output_dir / "candidate_checkpoint_best_policy.pt"
        canonical.write_bytes(b"a")
        alternate.write_bytes(b"b")
        tuning_csv = tmp_path / "tuning.csv"
        pd.DataFrame({"scenario_id": [101]}).to_csv(
            tuning_csv,
            index=False,
        )
        tuning_raw_dir = tmp_path / "tuning_raw"
        tuning_raw_dir.mkdir()
        config = CheckpointSelectionConfig(
            enabled=True,
            tuning_csv=tuning_csv,
            tuning_raw_dir=tuning_raw_dir,
            max_candidates=2,
        )
        result = checkpoint_arena.select_checkpoint_in_tuning_arena(
            canonical_checkpoint=canonical,
            project_root=tmp_path,
            output_dir=output_dir / "checkpoint_selection",
            config=config,
            physics_config=DEFAULT_PHYSICS_CONFIG,
            tuning_csv=tuning_csv,
            tuning_raw_dir=tuning_raw_dir,
            excluded_csvs={},
            evaluate=lambda **arena_kwargs: (
                _arena_metrics(ungated=0.80, constrained=0.82)
                if Path(arena_kwargs["checkpoint"]).read_bytes() == b"a"
                else _arena_metrics(ungated=0.70, constrained=0.95)
            ),
        )
        calls.append("arena")
        assert result.selected_source == canonical
        return result.checkpoint

    monkeypatch.setattr(
        iteration_module,
        "run_train",
        train_with_arena,
    )

    def evaluate(**kwargs: Any) -> dict[str, object]:
        checkpoint = Path(kwargs["checkpoint"])
        is_candidate = checkpoint.name != "parent.pt"
        calls.append(
            "candidate_evaluation"
            if is_candidate
            else "parent_evaluation"
        )
        _write_paired_rows(
            output_dir=Path(kwargs["output_dir"]),
            output_csv_name=kwargs["config"].output_csv_name,
            scenario_ids=tuple(kwargs["scenario_ids"]),
            secure_count=1 if is_candidate else 2,
        )
        return (
            _evaluation_metrics(0.60, 0.90)
            if is_candidate
            else _evaluation_metrics(0.70, 0.72)
        )

    monkeypatch.setattr(
        iteration_module,
        "run_evaluate",
        evaluate,
    )

    result = run_self_play_iteration(request)

    assert calls == [
        "training",
        "arena",
        "parent_evaluation",
        "candidate_evaluation",
    ]
    assert result.accepted is False
    assert result.status == "REJECTED"
    assert result.learning_curve_row["accepted"] is False
    assert result.learning_curve_row["primary_policy_mode"] == "ungated"
    assert result.learning_curve_row["candidate_metric"] == pytest.approx(0.60)
    assert result.learning_curve_row["best_metric_after"] == pytest.approx(0.70)
    assert result.candidate_metrics["mode_metrics"]["constrained"][
        "physically_secure_rate_requested"
    ] == pytest.approx(0.90)

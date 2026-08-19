from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.config.checkpoint_selection import (
    CheckpointSelectionConfig,
)
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play import checkpoint_arena, stages
from grid_topology_ai.self_play.checkpoint_arena import (
    CheckpointArenaResult,
)


def _candidate(
    path: Path,
    digest: str,
    *,
    loss: float,
    policy_loss: float,
    value_loss: float,
    calibration: float,
) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": digest,
        "payload": {},
        "val_metrics": {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "value_calibration_error": calibration,
        },
        "training_selector": "validation_loss",
        "saved_epoch": 1,
        "ranking_sources": [],
    }


def _arena_metrics(
    *,
    ungated: float,
    constrained: float,
    failed_scenarios: int = 0,
) -> dict[str, Any]:
    return {
        "primary_policy_mode": "ungated",
        "physically_secure_rate_requested": ungated,
        "failed_scenarios": failed_scenarios,
        "mode_metrics": {
            "ungated": {
                "physically_secure_rate_requested": ungated,
                "failed_scenarios": failed_scenarios,
            },
            "constrained": {
                "physically_secure_rate_requested": constrained,
                "failed_scenarios": failed_scenarios,
            },
        },
    }


def test_candidate_pool_covers_each_validation_objective(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(
            tmp_path / "loss.pt",
            "loss",
            loss=0.10,
            policy_loss=0.80,
            value_loss=0.70,
            calibration=0.60,
        ),
        _candidate(
            tmp_path / "policy.pt",
            "policy",
            loss=0.60,
            policy_loss=0.10,
            value_loss=0.80,
            calibration=0.70,
        ),
        _candidate(
            tmp_path / "value.pt",
            "value",
            loss=0.70,
            policy_loss=0.60,
            value_loss=0.10,
            calibration=0.80,
        ),
        _candidate(
            tmp_path / "calibration.pt",
            "calibration",
            loss=0.80,
            policy_loss=0.70,
            value_loss=0.60,
            calibration=0.10,
        ),
    ]
    config = CheckpointSelectionConfig(
        candidates_per_metric=1,
        max_candidates=4,
    )

    selected = checkpoint_arena._select_candidate_pool(
        candidates,
        config,
    )

    assert [item["path"].name for item in selected] == [
        "loss.pt",
        "policy.pt",
        "value.pt",
        "calibration.pt",
    ]
    assert selected[0]["ranking_sources"] == ["validation_loss"]
    assert selected[1]["ranking_sources"] == [
        "validation_policy_loss"
    ]
    assert selected[2]["ranking_sources"] == [
        "validation_value_loss"
    ]
    assert selected[3]["ranking_sources"] == [
        "validation_value_calibration_error"
    ]


def test_tuning_set_rejects_physical_lineage_overlap(
    tmp_path: Path,
) -> None:
    tuning = tmp_path / "tuning.csv"
    pool = tmp_path / "pool.csv"
    evaluation = tmp_path / "evaluation.csv"
    final_test = tmp_path / "final_test.csv"

    pd.DataFrame(
        {
            "scenario_id": [1],
            "physical_lineage_fingerprint": ["sha256:shared"],
        }
    ).to_csv(tuning, index=False)
    pd.DataFrame(
        {
            "scenario_id": [101],
            "physical_lineage_fingerprint": ["sha256:shared"],
        }
    ).to_csv(pool, index=False)
    for path, scenario_id, fingerprint in (
        (evaluation, 201, "sha256:eval"),
        (final_test, 301, "sha256:final"),
    ):
        pd.DataFrame(
            {
                "scenario_id": [scenario_id],
                "physical_lineage_fingerprint": [fingerprint],
            }
        ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="physical-lineage leakage",
    ):
        checkpoint_arena._validate_tuning_independence(
            tuning_csv=tuning,
            excluded_csvs={
                "self-play pool": pool,
                "evaluation set": evaluation,
                "final test set": final_test,
            },
        )


def test_tuning_arena_promotes_best_closed_loop_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "candidate_checkpoint.pt"
    best_loss = tmp_path / "candidate_checkpoint_best_loss.pt"
    best_policy = tmp_path / "candidate_checkpoint_best_policy.pt"
    for path in (canonical, best_loss, best_policy):
        path.write_bytes(path.name.encode("utf-8"))

    tuning_csv = tmp_path / "tuning.csv"
    pd.DataFrame({"scenario_id": [1, 2]}).to_csv(
        tuning_csv,
        index=False,
    )
    tuning_raw_dir = tmp_path / "raw"
    tuning_raw_dir.mkdir()

    loaded = [
        _candidate(
            canonical,
            checkpoint_arena.sha256_file(canonical),
            loss=0.10,
            policy_loss=0.40,
            value_loss=0.40,
            calibration=0.40,
        ),
        _candidate(
            best_loss,
            checkpoint_arena.sha256_file(best_loss),
            loss=0.20,
            policy_loss=0.10,
            value_loss=0.30,
            calibration=0.30,
        ),
        _candidate(
            best_policy,
            checkpoint_arena.sha256_file(best_policy),
            loss=0.30,
            policy_loss=0.30,
            value_loss=0.10,
            calibration=0.10,
        ),
    ]
    monkeypatch.setattr(
        checkpoint_arena,
        "_validate_tuning_independence",
        lambda **kwargs: (1, 2),
    )
    monkeypatch.setattr(
        checkpoint_arena,
        "_load_candidates",
        lambda **kwargs: loaded,
    )

    annotated: dict[str, Any] = {}

    def fake_annotate(**kwargs: Any) -> None:
        annotated.update(kwargs)
        Path(kwargs["destination"]).write_bytes(
            Path(kwargs["source"]).read_bytes()
        )

    monkeypatch.setattr(
        checkpoint_arena,
        "_annotate_selected_checkpoint",
        fake_annotate,
    )

    scores = {
        canonical.name: (0.40, 0.90),
        best_loss.name: (0.55, 0.80),
        best_policy.name: (0.70, 0.71),
    }
    evaluated: list[Path] = []

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        checkpoint = Path(kwargs["checkpoint"])
        evaluated.append(checkpoint)
        source_name = checkpoint.read_bytes().decode("utf-8")
        ungated, constrained = scores[source_name]
        return _arena_metrics(
            ungated=ungated,
            constrained=constrained,
        )

    config = CheckpointSelectionConfig(
        enabled=True,
        tuning_csv=tuning_csv,
        tuning_raw_dir=tuning_raw_dir,
        candidates_per_metric=1,
        max_candidates=3,
    )
    result = checkpoint_arena.select_checkpoint_in_tuning_arena(
        canonical_checkpoint=canonical,
        project_root=tmp_path,
        output_dir=tmp_path / "selection",
        config=config,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        tuning_csv=tuning_csv,
        tuning_raw_dir=tuning_raw_dir,
        excluded_csvs={},
        evaluate=fake_evaluate,
    )

    assert result.selected_source == best_policy
    assert result.metric_value == pytest.approx(0.70)
    assert result.candidate_count == 3
    assert Path(annotated["source"]).parent.name == "candidates"
    assert Path(annotated["source"]).read_bytes() == best_policy.read_bytes()
    assert canonical.read_bytes() == best_policy.read_bytes()
    assert len(evaluated) == 3
    assert all(path.parent.name == "candidates" for path in evaluated)
    assert {path.read_bytes() for path in evaluated} == {
        canonical.name.encode("utf-8"),
        best_loss.name.encode("utf-8"),
        best_policy.name.encode("utf-8"),
    }

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["selection_method"] == "closed_loop_tuning_arena"
    assert report["policy_mode"] == "ungated"
    assert report["selected_source_checkpoint"] == str(best_policy)
    assert Path(report["selected_archived_checkpoint"]).is_file()
    assert report["tuning_scenario_ids"] == [1, 2]
    assert all(
        Path(item["archived_checkpoint"]).is_file()
        for item in report["candidates"]
    )


def test_tuning_arena_ignores_better_constrained_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_a = tmp_path / "candidate_checkpoint.pt"
    candidate_b = tmp_path / "candidate_checkpoint_best_policy.pt"
    candidate_a.write_bytes(b"a")
    candidate_b.write_bytes(b"b")

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
        lambda **kwargs: [
            _candidate(
                candidate_a,
                checkpoint_arena.sha256_file(candidate_a),
                loss=0.1,
                policy_loss=0.2,
                value_loss=0.2,
                calibration=0.2,
            ),
            _candidate(
                candidate_b,
                checkpoint_arena.sha256_file(candidate_b),
                loss=0.2,
                policy_loss=0.1,
                value_loss=0.1,
                calibration=0.1,
            ),
        ],
    )
    monkeypatch.setattr(
        checkpoint_arena,
        "_annotate_selected_checkpoint",
        lambda **kwargs: Path(kwargs["destination"]).write_bytes(
            Path(kwargs["source"]).read_bytes()
        ),
    )

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        content = Path(kwargs["checkpoint"]).read_bytes()
        if content == b"a":
            return _arena_metrics(ungated=0.80, constrained=0.82)
        return _arena_metrics(ungated=0.70, constrained=0.95)

    config = CheckpointSelectionConfig(
        enabled=True,
        tuning_csv=tuning_csv,
        tuning_raw_dir=tuning_raw_dir,
        max_candidates=2,
    )
    result = checkpoint_arena.select_checkpoint_in_tuning_arena(
        canonical_checkpoint=candidate_a,
        project_root=tmp_path,
        output_dir=tmp_path / "selection",
        config=config,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        tuning_csv=tuning_csv,
        tuning_raw_dir=tuning_raw_dir,
        excluded_csvs={},
        evaluate=fake_evaluate,
    )

    assert result.selected_source == candidate_a
    assert result.metric_value == pytest.approx(0.80)


def test_run_train_routes_candidate_through_tuning_arena(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run" / "iter_001"
    output_dir.mkdir(parents=True)
    canonical = output_dir / "candidate_checkpoint.pt"
    torch.save(
        {"checkpoint_selection_metric": "validation_loss"},
        canonical,
    )
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    train_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    validation_csv.write_text("scenario_id\n2\n", encoding="utf-8")

    selection_config = CheckpointSelectionConfig(
        enabled=True,
        tuning_csv=Path("tuning.csv"),
        tuning_raw_dir=Path("tuning_raw"),
        calibration_bins=7,
    )
    self_play_config = SimpleNamespace(
        checkpoint_selection=selection_config,
    )
    resolved_paths = SimpleNamespace(
        tuning_csv=tmp_path / "tuning.csv",
        tuning_raw_dir=tmp_path / "tuning_raw",
        pool_transitions_csv=tmp_path / "pool.csv",
        eval_csv=tmp_path / "eval.csv",
        final_test_csv=tmp_path / "final.csv",
    )

    monkeypatch.setattr(
        stages,
        "_resolved_self_play_config",
        lambda output_dir: self_play_config,
    )
    monkeypatch.setattr(
        stages,
        "validation_diagnostic_options",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        stages.SelfPlayPaths,
        "from_config",
        lambda config, project_root: resolved_paths,
    )
    monkeypatch.setattr(
        stages,
        "train_graph_policy_value_model",
        lambda request: canonical,
    )
    monkeypatch.setattr(
        stages,
        "require_checkpoint_contracts",
        lambda *args, **kwargs: None,
    )

    captured: dict[str, Any] = {}

    def fake_select(**kwargs: Any) -> CheckpointArenaResult:
        captured.update(kwargs)
        return CheckpointArenaResult(
            checkpoint=canonical,
            report_path=output_dir / "checkpoint_selection.json",
            selected_source=canonical,
            metric_name=selection_config.metric,
            metric_value=0.75,
            candidate_count=2,
        )

    monkeypatch.setattr(
        stages,
        "select_checkpoint_in_tuning_arena",
        fake_select,
    )

    result = stages.run_train(
        project_root=tmp_path,
        examples_csv=train_csv,
        validation_examples_csv=validation_csv,
        init_checkpoint=tmp_path / "parent.pt",
        output_dir=output_dir,
        config=TrainingConfig(save_multiple_best=True),
        physics_config=DEFAULT_PHYSICS_CONFIG,
        iteration=1,
        seed=43,
    )

    assert result == canonical
    assert captured["canonical_checkpoint"] == canonical
    assert captured["tuning_csv"] == resolved_paths.tuning_csv
    assert captured["excluded_csvs"] == {
        "self-play pool": resolved_paths.pool_transitions_csv,
        "evaluation set": resolved_paths.eval_csv,
        "final test set": resolved_paths.final_test_csv,
    }
    assert captured["evaluate"] is stages.run_evaluate

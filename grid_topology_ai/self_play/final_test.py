from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.evaluation.checkpoint import load_scenario_ids
from grid_topology_ai.self_play.acceptance import (
    require_metrics_pf_alg,
    require_metrics_physics_config,
)
from grid_topology_ai.self_play.artifacts import (
    save_json,
    sha256_file,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.stages import run_evaluate

FINAL_TEST_REPORT_SCHEMA_VERSION = 1
FINAL_TEST_EVALUATION_ROLE = "reporting_only"


@dataclass(frozen=True, slots=True)
class FinalTestEvaluation:
    metrics: dict[str, object]
    metrics_path: Path
    results_path: Path
    report_path: Path
    checkpoint: Path


def _require_selected_best_checkpoint(
    checkpoint: Path,
    paths: SelfPlayPaths,
) -> None:
    if checkpoint.resolve() != paths.best_checkpoint.resolve():
        raise ValueError(
            "Final-test evaluation must use the selected best checkpoint: "
            f"expected {paths.best_checkpoint}, observed {checkpoint}."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Final-test best checkpoint not found: {checkpoint}"
        )
    if not paths.best_metrics.is_file():
        raise FileNotFoundError(
            f"Final-test best metrics not found: {paths.best_metrics}"
        )


def run_final_test_evaluation(
    *,
    paths: SelfPlayPaths,
    checkpoint: str | Path,
    config: EvaluationConfig,
    physics_config: PhysicsConfig,
) -> FinalTestEvaluation:
    selected_checkpoint = Path(checkpoint)
    _require_selected_best_checkpoint(selected_checkpoint, paths)

    checkpoint_sha_before = sha256_file(selected_checkpoint)
    best_metrics_sha_before = sha256_file(paths.best_metrics)
    final_test_csv_sha = sha256_file(paths.final_test_csv)
    scenario_ids = tuple(
        load_scenario_ids(
            paths.final_test_csv,
            limit=None,
        )
    )
    if not scenario_ids:
        raise ValueError("Final-test transitions contain no scenario IDs.")

    output_dir = paths.run_dir / "final_test"
    metrics = run_evaluate(
        project_root=paths.project_root,
        checkpoint=selected_checkpoint,
        eval_csv=paths.final_test_csv,
        eval_raw_dir=paths.final_test_raw_dir,
        output_dir=output_dir,
        config=config,
        physics_config=physics_config,
        scenario_ids=scenario_ids,
    )

    metrics_path = output_dir / config.output_json_name
    results_path = output_dir / config.output_csv_name
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Final-test metrics were not created: {metrics_path}"
        )
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Final-test results were not created: {results_path}"
        )

    require_metrics_pf_alg(
        metrics,
        expected_pf_alg=config.pf_alg,
        source=str(metrics_path),
    )
    require_metrics_physics_config(
        metrics,
        expected_physics_config=physics_config,
        source=str(metrics_path),
    )

    checkpoint_sha_after = sha256_file(selected_checkpoint)
    best_metrics_sha_after = sha256_file(paths.best_metrics)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError(
            "Final-test evaluation modified the selected best checkpoint."
        )
    if best_metrics_sha_after != best_metrics_sha_before:
        raise RuntimeError(
            "Final-test evaluation modified checkpoint-selection metrics."
        )

    report_path = output_dir / "final_test_report.json"
    report: dict[str, Any] = {
        "schema_version": FINAL_TEST_REPORT_SCHEMA_VERSION,
        "evaluation_role": FINAL_TEST_EVALUATION_ROLE,
        "checkpoint_selection_allowed": False,
        "checkpoint_promotion_allowed": False,
        "checkpoint_selected_before_evaluation": True,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": checkpoint_sha_before,
        "best_metrics_path": str(paths.best_metrics),
        "best_metrics_sha256": best_metrics_sha_before,
        "final_test_transitions_csv": str(paths.final_test_csv),
        "final_test_transitions_sha256": final_test_csv_sha,
        "final_test_raw_dir": str(paths.final_test_raw_dir),
        "scenario_count": len(scenario_ids),
        "scenario_ids": [int(value) for value in scenario_ids],
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
    }
    save_json(report, report_path)

    return FinalTestEvaluation(
        metrics=dict(metrics),
        metrics_path=metrics_path,
        results_path=results_path,
        report_path=report_path,
        checkpoint=selected_checkpoint,
    )

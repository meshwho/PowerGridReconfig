from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.evaluation.checkpoint import load_scenario_ids
from grid_topology_ai.evaluation.policy_comparison import (
    PolicyMode,
    require_policy_mode_metrics,
    require_primary_policy_mode,
)
from grid_topology_ai.self_play.acceptance import (
    require_metrics_pf_alg,
    require_metrics_physics_config,
)
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
    sha256_files,
    sha256_json,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.stages import run_evaluate

FINAL_TEST_REPORT_SCHEMA_VERSION = 3
FINAL_TEST_EVALUATION_ROLE = "reporting_only"
_SECURE_RATE_METRIC = "physically_secure_rate_requested"
_FINAL_TEST_RAW_FILES = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
)


@dataclass(frozen=True, slots=True)
class FinalTestEvaluation:
    metrics: dict[str, object]
    metrics_path: Path
    results_path: Path
    report_path: Path
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class _PolicyReportValues:
    ungated_rate: float
    constrained_rate: float | None
    continuation_gate_gain: float | None


def _require_ungated_config(config: EvaluationConfig) -> None:
    if config.primary_policy_mode != PolicyMode.UNGATED.value:
        raise ValueError(
            "Final-test primary_policy_mode must be 'ungated'."
        )


def _require_rate(
    metrics: Mapping[str, object],
    *,
    source: str,
) -> float:
    value = metrics.get(_SECURE_RATE_METRIC)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"Final-test metric {_SECURE_RATE_METRIC!r} is invalid: {source}."
        )

    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(
            f"Final-test metric {_SECURE_RATE_METRIC!r} is invalid: {source}."
        )
    return rate


def _policy_report_values(
    metrics: Mapping[str, object],
    *,
    source: str,
    expect_constrained: bool,
) -> _PolicyReportValues:
    require_primary_policy_mode(
        metrics,
        PolicyMode.UNGATED,
        source=source,
    )
    ungated_metrics = require_policy_mode_metrics(
        metrics,
        PolicyMode.UNGATED,
        source=source,
    )
    ungated_rate = _require_rate(
        ungated_metrics,
        source=f"{source} ungated metrics",
    )

    raw_mode_metrics = metrics.get("mode_metrics")
    assert isinstance(raw_mode_metrics, Mapping)
    raw_constrained = raw_mode_metrics.get(
        PolicyMode.CONSTRAINED.value
    )

    if raw_constrained is None:
        if expect_constrained:
            raise ValueError(
                f"Final-test constrained metrics are missing: {source}."
            )
        constrained_rate = None
    elif not isinstance(raw_constrained, Mapping):
        raise ValueError(
            f"Final-test constrained metrics are invalid: {source}."
        )
    else:
        constrained_rate = _require_rate(
            raw_constrained,
            source=f"{source} constrained metrics",
        )

    gain = (
        None
        if constrained_rate is None
        else constrained_rate - ungated_rate
    )
    return _PolicyReportValues(
        ungated_rate=ungated_rate,
        constrained_rate=constrained_rate,
        continuation_gate_gain=gain,
    )


def _require_report_value(
    report: Mapping[str, object],
    *,
    name: str,
    expected: float | None,
    source: Path,
) -> None:
    observed = report.get(name)
    if expected is None:
        if observed is not None:
            raise ValueError(
                f"Invalid final-test report field {name}: {source}."
            )
        return

    if isinstance(observed, bool) or not isinstance(observed, Real):
        raise ValueError(
            f"Invalid final-test report field {name}: {source}."
        )
    if not math.isclose(
        float(observed),
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Invalid final-test report field {name}: {source}."
        )


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


def _raw_files(paths: SelfPlayPaths) -> list[Path]:
    return [
        paths.final_test_raw_dir / file_name
        for file_name in _FINAL_TEST_RAW_FILES
    ]


def _raw_data_hash(paths: SelfPlayPaths) -> str:
    files = _raw_files(paths)
    existing = [path for path in files if path.is_file()]
    if not existing:
        return sha256_json([])
    missing_required = [path for path in files[:2] if not path.is_file()]
    if missing_required:
        raise FileNotFoundError(
            "Final-test raw files are incomplete: "
            + ", ".join(str(path) for path in missing_required)
        )
    return sha256_files(existing, root=paths.final_test_raw_dir)


def _scenario_ids(paths: SelfPlayPaths) -> tuple[int, ...]:
    scenario_ids = tuple(
        load_scenario_ids(
            paths.final_test_csv,
            limit=None,
        )
    )
    if not scenario_ids:
        raise ValueError("Final-test transitions contain no scenario IDs.")
    return scenario_ids


def load_final_test_evaluation(
    *,
    paths: SelfPlayPaths,
    config: EvaluationConfig,
    physics_config: PhysicsConfig,
) -> FinalTestEvaluation | None:
    _require_ungated_config(config)
    report_path = paths.final_test_report
    if not report_path.exists():
        partial = [
            path
            for path in (
                paths.final_test_dir / config.output_json_name,
                paths.final_test_dir / config.output_csv_name,
            )
            if path.exists()
        ]
        if partial:
            raise RuntimeError(
                "Incomplete final-test artifacts exist without a sealed report: "
                + ", ".join(str(path) for path in partial)
            )
        return None

    report = load_json(report_path)
    if report.get("schema_version") != FINAL_TEST_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported final-test report schema: {report_path}."
        )
    if report.get("evaluation_role") != FINAL_TEST_EVALUATION_ROLE:
        raise ValueError(f"Invalid final-test evaluation role: {report_path}.")
    if report.get("primary_policy_mode") != PolicyMode.UNGATED.value:
        raise ValueError(
            f"Invalid final-test primary policy mode: {report_path}."
        )
    required_flags = {
        "checkpoint_selection_allowed": False,
        "checkpoint_promotion_allowed": False,
        "checkpoint_selected_before_evaluation": True,
        "run_sealed": True,
    }
    for name, expected in required_flags.items():
        if report.get(name) is not expected:
            raise ValueError(
                f"Invalid final-test report flag {name}: {report_path}."
            )

    checkpoint = Path(str(report.get("checkpoint", "")))
    _require_selected_best_checkpoint(checkpoint, paths)
    if report.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError(
            f"Final-test checkpoint hash mismatch: {report_path}."
        )
    if report.get("best_metrics_sha256") != sha256_file(paths.best_metrics):
        raise ValueError(
            f"Final-test best metrics hash mismatch: {report_path}."
        )
    if report.get("final_test_transitions_sha256") != sha256_file(
        paths.final_test_csv
    ):
        raise ValueError(
            f"Final-test transitions hash mismatch: {report_path}."
        )
    raw_hash = _raw_data_hash(paths)
    if report.get("final_test_raw_sha256") != raw_hash:
        raise ValueError(f"Final-test raw data hash mismatch: {report_path}.")

    scenario_ids = _scenario_ids(paths)
    recorded_ids = report.get("scenario_ids")
    if recorded_ids != [int(value) for value in scenario_ids]:
        raise ValueError(f"Final-test scenario IDs changed: {report_path}.")
    if int(report.get("scenario_count", -1)) != len(scenario_ids):
        raise ValueError(f"Final-test scenario count mismatch: {report_path}.")

    metrics_path = paths.final_test_dir / config.output_json_name
    results_path = paths.final_test_dir / config.output_csv_name
    for path, hash_field in (
        (metrics_path, "metrics_sha256"),
        (results_path, "results_sha256"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Final-test artifact not found: {path}")
        if report.get(hash_field) != sha256_file(path):
            raise ValueError(f"Final-test artifact hash mismatch: {path}.")

    metrics = load_json(metrics_path)
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
    policy_values = _policy_report_values(
        metrics,
        source=str(metrics_path),
        expect_constrained=config.use_continuation_gate,
    )
    _require_report_value(
        report,
        name="ungated_physically_secure_rate_requested",
        expected=policy_values.ungated_rate,
        source=report_path,
    )
    _require_report_value(
        report,
        name="constrained_physically_secure_rate_requested",
        expected=policy_values.constrained_rate,
        source=report_path,
    )
    _require_report_value(
        report,
        name="continuation_gate_gain",
        expected=policy_values.continuation_gate_gain,
        source=report_path,
    )
    return FinalTestEvaluation(
        metrics=metrics,
        metrics_path=metrics_path,
        results_path=results_path,
        report_path=report_path,
        checkpoint=checkpoint,
    )


def run_final_test_evaluation(
    *,
    paths: SelfPlayPaths,
    checkpoint: str | Path,
    config: EvaluationConfig,
    physics_config: PhysicsConfig,
) -> FinalTestEvaluation:
    _require_ungated_config(config)
    selected_checkpoint = Path(checkpoint)
    _require_selected_best_checkpoint(selected_checkpoint, paths)
    if paths.final_test_report.exists():
        raise FileExistsError(
            "Final-test report already exists and the run is sealed: "
            f"{paths.final_test_report}."
        )

    output_dir = paths.final_test_dir
    metrics_path = output_dir / config.output_json_name
    results_path = output_dir / config.output_csv_name
    partial = [path for path in (metrics_path, results_path) if path.exists()]
    if partial:
        raise RuntimeError(
            "Incomplete final-test artifacts must be inspected before retrying: "
            + ", ".join(str(path) for path in partial)
        )

    checkpoint_sha_before = sha256_file(selected_checkpoint)
    best_metrics_sha_before = sha256_file(paths.best_metrics)
    final_test_csv_sha = sha256_file(paths.final_test_csv)
    final_test_raw_sha = _raw_data_hash(paths)
    scenario_ids = _scenario_ids(paths)

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
    policy_values = _policy_report_values(
        metrics,
        source=str(metrics_path),
        expect_constrained=config.use_continuation_gate,
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

    report: dict[str, Any] = {
        "schema_version": FINAL_TEST_REPORT_SCHEMA_VERSION,
        "evaluation_role": FINAL_TEST_EVALUATION_ROLE,
        "primary_policy_mode": PolicyMode.UNGATED.value,
        "ungated_physically_secure_rate_requested": (
            policy_values.ungated_rate
        ),
        "constrained_physically_secure_rate_requested": (
            policy_values.constrained_rate
        ),
        "continuation_gate_gain": (
            policy_values.continuation_gate_gain
        ),
        "checkpoint_selection_allowed": False,
        "checkpoint_promotion_allowed": False,
        "checkpoint_selected_before_evaluation": True,
        "run_sealed": True,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": checkpoint_sha_before,
        "best_metrics_path": str(paths.best_metrics),
        "best_metrics_sha256": best_metrics_sha_before,
        "final_test_transitions_csv": str(paths.final_test_csv),
        "final_test_transitions_sha256": final_test_csv_sha,
        "final_test_raw_dir": str(paths.final_test_raw_dir),
        "final_test_raw_sha256": final_test_raw_sha,
        "scenario_count": len(scenario_ids),
        "scenario_ids": [int(value) for value in scenario_ids],
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
    }
    save_json(report, paths.final_test_report)

    return FinalTestEvaluation(
        metrics=dict(metrics),
        metrics_path=metrics_path,
        results_path=results_path,
        report_path=paths.final_test_report,
        checkpoint=selected_checkpoint,
    )

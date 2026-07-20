from __future__ import annotations

import pandas as pd

from scripts.evaluation.evaluate_policy_supervised import (
    Metrics,
    build_parser,
    get_target_value,
)


def test_get_target_value_uses_strict_outcome_target() -> None:
    row = pd.Series({"outcome_value_target": 0.0, "discounted_return_from_step": 10000.0})
    assert get_target_value(row) == 0.0


def test_get_target_value_ignores_legacy_return() -> None:
    row = pd.Series({"outcome_value_target": -0.95, "discounted_return_from_step": 10000.0})
    assert get_target_value(row) == -0.95


def test_get_target_value_ignores_negative_legacy_return() -> None:
    row = pd.Series({"outcome_value_target": 0.81, "discounted_return_from_step": -999999.0})
    assert get_target_value(row) == 0.81


def test_evaluation_parser_has_no_legacy_value_scale() -> None:
    help_text = build_parser().format_help()
    assert "--value-scale" not in help_text
    assert "examples_csv" in help_text
    assert "--checkpoint" in help_text
    assert "--device" in help_text


def test_metrics_uses_given_outcome_target() -> None:
    metrics = Metrics()
    metrics.update(0, 0, [0], target_value=-0.75, predicted_value=0.25)
    assert metrics.value_abs_error_sum == 1.0
    assert metrics.as_dict()["mae_value"] == 1.0

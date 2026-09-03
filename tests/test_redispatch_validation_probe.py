from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from scripts import diagnose_redispatch_validation as probe


def _assessment(**overrides):
    values = {field: True for field in probe.FEASIBILITY_FIELDS}
    values.update(
        {
            "max_loading_percent": 99.0,
            "num_overloaded_branches": 0,
            "num_hard_overloaded_branches": 0,
            "total_voltage_violation": 0.0,
            "num_generator_p_violations": 0,
            "total_generator_p_violation_mw": 0.0,
            "num_generator_q_violations": 0,
            "total_generator_q_violation_mvar": 0.0,
            "num_angle_difference_violations": 0,
            "total_angle_difference_violation_degrees": 0.0,
        }
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_failure_key_reports_exact_failed_constraints() -> None:
    assessment = _assessment(
        thermal_feasible=False,
        generator_p_feasible=False,
    )

    assert probe._assessment_failure_key(assessment) == "thermal+generator_p"


def test_row_from_result_preserves_post_opf_assessment() -> None:
    assessment = _assessment(
        generator_p_feasible=False,
        num_generator_p_violations=4,
        total_generator_p_violation_mw=7e-6,
    )
    result = SimpleNamespace(
        opf_success=True,
        validated=False,
        message="strict contract failed",
        redispatch_l1_mw=2.5,
        redispatch_up_mw=1.5,
        redispatch_down_mw=1.0,
        redispatch_max_generator_delta_mw=0.75,
        assessment=assessment,
    )

    row = probe._row_from_result(17, result)

    assert row["scenario_id"] == 17
    assert row["opf_success"] is True
    assert row["validated"] is False
    assert row["failure_key"] == "generator_p"
    assert row["num_generator_p_violations"] == 4
    assert row["total_generator_p_violation_mw"] == 7e-6


def test_summary_separates_solver_success_from_postcheck_failure(capsys) -> None:
    rows = pd.DataFrame(
        [
            probe._row_from_result(
                1,
                SimpleNamespace(
                    opf_success=True,
                    validated=False,
                    message="strict contract failed",
                    redispatch_l1_mw=1.0,
                    redispatch_up_mw=1.0,
                    redispatch_down_mw=0.0,
                    redispatch_max_generator_delta_mw=1.0,
                    assessment=_assessment(
                        generator_p_feasible=False,
                        num_generator_p_violations=1,
                        total_generator_p_violation_mw=5e-6,
                    ),
                ),
            ),
            probe._row_from_result(
                2,
                SimpleNamespace(
                    opf_success=True,
                    validated=True,
                    message="validated",
                    redispatch_l1_mw=1.0,
                    redispatch_up_mw=1.0,
                    redispatch_down_mw=0.0,
                    redispatch_max_generator_delta_mw=1.0,
                    assessment=_assessment(),
                ),
            ),
        ]
    )

    probe._print_summary(rows)
    output = capsys.readouterr().out

    assert "OPF success:               2 (100.00%)" in output
    assert "Validated:                 1 (50.00%)" in output
    assert "Validated / OPF success: 1/2 (50.00%)" in output
    assert "generator_p" in output

import pandas as pd

from grid_topology_ai.evaluation import build_evaluation_metrics


_BASE = {
    "steps": 1,
    "discounted_return": 0.0,
    "final_num_outaged_branches": 0,
    "total_generator_p_violation_mw": 0.0,
    "total_generator_q_violation_mvar": 0.0,
    "total_angle_difference_violation_degrees": 0.0,
    "total_voltage_violation": 0.0,
    "num_high_voltage_buses": 0,
}


def _row(
    *,
    solved: bool,
    reason: str,
    physically_secure: bool,
    pf: bool = True,
    hard_free: bool = True,
    voltage: bool = True,
    thermal: bool = True,
    safe_handoff: bool = False,
    unsafe_terminal: bool = False,
) -> dict[str, object]:
    return {
        **_BASE,
        "solved": solved,
        "termination_reason": reason,
        "final_max_loading_percent": 90.0 if thermal else 130.0,
        "final_num_overloaded_branches": 0 if thermal else 1,
        "final_num_hard_overloaded_branches": 0 if hard_free else 1,
        "safety_score": 100.0 if physically_secure else -50.0,
        "hard_overload_free": hard_free,
        "voltage_feasible": voltage,
        "physically_secure": physically_secure,
        "safe_handoff": safe_handoff,
        "unsafe_terminal_state": unsafe_terminal,
        "power_flow_converged": pf,
        "all_values_finite": pf,
        "topology_connected": pf,
        "thermal_solved": thermal,
        "thermal_feasible": thermal,
        "generator_p_feasible": pf,
        "generator_q_feasible": pf,
        "angle_difference_feasible": pf,
        "num_generator_p_violations": 0 if pf else 1,
        "num_generator_q_violations": 0 if pf else 1,
        "num_angle_difference_violations": 0 if pf else 1,
        "num_low_voltage_buses": 0 if voltage else 1,
        "total_thermal_overload_mva": 0.0 if thermal else 30.0,
    }


def _evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                solved=True,
                reason="solved",
                physically_secure=True,
            ),
            _row(
                solved=False,
                reason="handoff_to_redispatch",
                physically_secure=False,
                thermal=False,
                safe_handoff=True,
            ),
            _row(
                solved=False,
                reason="unsafe_stop_with_hard_overload",
                physically_secure=False,
                thermal=False,
                hard_free=False,
                unsafe_terminal=True,
            ),
            _row(
                solved=False,
                reason="power_flow_failed",
                physically_secure=False,
                pf=False,
                thermal=False,
                hard_free=False,
                voltage=False,
            ),
        ]
    )


def test_evaluation_physical_counts_and_rates() -> None:
    metrics = build_evaluation_metrics(
        _evaluation_frame(),
        [{"scenario_id": 99, "error": "worker"}],
        5,
    )

    assert metrics["requested_scenarios"] == 5
    assert metrics["evaluated_scenarios"] == 4
    assert metrics["failed_scenarios"] == 1
    assert metrics["evaluation_coverage_rate"] == 0.8
    assert metrics["solve_count"] == 1
    assert metrics["solve_rate"] == 0.25
    assert metrics["solve_rate_requested"] == 0.2
    assert metrics["solve_rate"] == metrics["physically_secure_rate"]
    assert metrics["physically_secure_rate_requested"] == 0.2
    assert metrics["failed_scenario_rate_requested"] == 0.2
    assert metrics["safe_handoff_count"] == 1
    assert metrics["unsafe_terminal_state_count"] == 1
    assert metrics["power_flow_failure_count"] == 1


def test_component_requested_rates_use_requested_denominator() -> None:
    metrics = build_evaluation_metrics(
        _evaluation_frame(),
        [{"scenario_id": 99, "error": "worker"}],
        5,
    )

    for field in (
        "power_flow_converged",
        "all_values_finite",
        "topology_connected",
        "thermal_solved",
        "thermal_feasible",
        "hard_overload_free",
        "voltage_feasible",
        "generator_p_feasible",
        "generator_q_feasible",
        "angle_difference_feasible",
        "physically_secure",
    ):
        assert metrics[f"{field}_rate_requested"] == metrics[f"{field}_count"] / 5


def test_evaluation_rates_are_zero_for_empty_request() -> None:
    frame = _evaluation_frame().iloc[0:0]
    metrics = build_evaluation_metrics(frame, [], 0)

    for key, value in metrics.items():
        if key.endswith("_rate") or key.endswith("_rate_requested"):
            assert value == 0.0

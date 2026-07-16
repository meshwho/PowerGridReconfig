import pandas as pd

from grid_topology_ai.evaluation.metrics import build_evaluation_metrics

COLS = [
    "solved", "termination_reason", "steps", "discounted_return",
    "final_max_loading_percent", "final_num_overloaded_branches",
    "final_num_hard_overloaded_branches", "safety_score",
    "hard_overload_free", "voltage_feasible", "physically_secure",
    "safe_handoff", "unsafe_terminal_state",
]


def test_evaluation_physical_contract_counts_and_rates():
    df = pd.DataFrame([
        dict(zip(COLS, [True, "solved", 1, 10.0, 90.0, 0, 0, 100.0, True, True, True, False, False])),
        dict(zip(COLS, [False, "handoff_to_redispatch", 1, 5.0, 110.0, 1, 0, 50.0, True, True, False, True, False])),
        dict(zip(COLS, [False, "unsafe_stop_with_hard_overload", 1, 0.0, 130.0, 1, 1, -50.0, False, True, False, False, True])),
        dict(zip(COLS, [False, "power_flow_failed", 0, 0.0, float("nan"), -1, -1, -100.0, False, False, False, False, False])),
    ])
    metrics = build_evaluation_metrics(
        df, [{"scenario_id": 99, "error": "worker"}], 5, {}
    )
    assert metrics["requested_scenarios"] == 5
    assert metrics["evaluated_scenarios"] == 4
    assert metrics["failed_scenarios"] == 1
    assert metrics["evaluation_coverage_rate"] == 0.8
    assert metrics["solve_count"] == 1
    assert metrics["solve_rate"] == 0.25
    assert metrics["solve_rate_requested"] == 0.2
    assert metrics["failed_scenario_rate_requested"] == 0.2
    assert metrics["hard_overload_free_count"] == 2
    assert metrics["physically_secure_count"] == 1
    assert metrics["safe_handoff_count"] == 1
    assert metrics["unsafe_terminal_state_count"] == 1
    assert metrics["power_flow_failure_count"] == 1
    assert metrics["physical_objective_contract"]["schema_version"] == 1


def test_evaluation_physical_contract_rates_are_zero_for_empty_request():
    df = pd.DataFrame(columns=COLS)
    metrics = build_evaluation_metrics(df, [], 0, {})
    for key, value in metrics.items():
        if key.endswith("_rate") or key in {"evaluation_coverage_rate", "solve_rate_requested", "failed_scenario_rate_requested"}:
            assert value == 0.0

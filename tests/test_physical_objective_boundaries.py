from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_physical_objective_has_no_heavy_or_cyclic_imports():
    text = (ROOT / "grid_topology_ai/physical_objective.py").read_text()
    for token in ("numpy", "pandas", "torch", "grid_topology_ai.environment", "grid_topology_ai.search", "grid_topology_ai.self_play"):
        assert token not in text


def test_objective_consumers_import_physical_objective_contract():
    files = [
        "reward.py",
        "environment.py",
        "search/mcts.py",
        "evaluation/metrics.py",
        "evaluation/checkpoint.py",
    ]

    for rel in files:
        text = (ROOT / "grid_topology_ai" / rel).read_text()
        assert "physical_objective" in text


def test_runtime_threshold_consumers_import_physics_config():
    files = [
        "data_adapter.py",
        "search/dc_action_screener.py",
        "search/impact_beam_search.py",
        "search/mcts.py",
        "search/continuation_gate.py",
        "self_play/generation.py",
        "evaluation/metrics.py",
    ]

    for rel in files:
        text = (ROOT / "grid_topology_ai" / rel).read_text()
        assert "grid_topology_ai.config.physics" in text


def test_selected_literal_threshold_comparisons_were_removed():
    targets = [
        "search/continuation_gate.py",
        "search/dc_action_screener.py",
        "search/impact_beam_search.py",
        "search/mcts.py",
        "self_play/generation.py",
        "evaluation/metrics.py",
        "data_adapter.py",
    ]
    forbidden = ("loading - 100.0", "loading - 120.0", "final_loading > 100.0")
    for rel in targets:
        text = (ROOT / "grid_topology_ai" / rel).read_text()
        for token in forbidden:
            assert token not in text


def test_custom_runtime_thresholds_are_used_consistently():
    import numpy as np

    from grid_topology_ai.config.physics import PhysicsConfig
    from grid_topology_ai.action_space import GridFMAction
    from grid_topology_ai.data_adapter import (
        BRANCH_FEATURE_COLUMNS,
        GridFMState,
    )
    from grid_topology_ai.evaluation.metrics import compute_safety_score
    from grid_topology_ai.search.continuation_gate import topology_penalty
    from grid_topology_ai.search.dc_action_screener import DCActionScreener
    from grid_topology_ai.search.impact_beam_search import safety_score
    from pypower.idx_brch import BR_STATUS, PF, PT, RATE_A

    physics_config = PhysicsConfig(
        overload_limit_percent=115.0,
        hard_overload_limit_percent=135.0,
        thermal_tolerance_percent=0.0,
    )
    branch_features = np.zeros(
        (3, len(BRANCH_FEATURE_COLUMNS)),
        dtype=float,
    )
    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
    status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")
    branch_features[:, loading_idx] = [130.0, 105.0, 200.0]
    branch_features[:, status_idx] = [1.0, 1.0, 0.0]

    state = GridFMState(
        scenario_id=1,
        load_scenario_idx=1.0,
        bus_features=np.zeros((1, 1), dtype=float),
        branch_features=branch_features,
        edge_index=np.zeros((2, 3), dtype=int),
        branch_ids=np.array([1, 2, 3], dtype=int),
        branch_status=np.array([1, 1, 0], dtype=int),
        metrics={
            "max_loading_percent": 130.0,
            "num_overloaded_branches": 1,
            "num_hard_overloaded_branches": 0,
            "total_voltage_violation": 0.2,
        },
        outaged_branch_ids=[],
    )
    evaluation_row = {
        "solved": False,
        "physically_secure": False,
        "termination_reason": "max_steps_reached",
        "final_max_loading_percent": 130.0,
        "final_num_overloaded_branches": 1,
        "final_num_hard_overloaded_branches": 0,
        "discounted_return": 40.0,
    }

    assert topology_penalty(state, physics_config=physics_config) == 315.0
    assert safety_score(state, physics_config=physics_config) == 1205.0

    dc_branch = np.zeros((2, 17), dtype=float)
    dc_branch[:, BR_STATUS] = 1.0
    dc_branch[:, RATE_A] = 100.0
    dc_branch[:, PF] = [130.0, 105.0]
    dc_branch[:, PT] = [-130.0, -105.0]
    dc_score = DCActionScreener(
        physics_config=physics_config,
    )._score_dc_result(
        action=GridFMAction(
            action_id=1,
            action_type="switch_off_branch",
            branch_id=1,
            branch_pos=0,
        ),
        result_ppc={"branch": dc_branch},
        policy_prior=0.0,
    )
    assert dc_score.penalty == 41.5
    assert compute_safety_score(
        evaluation_row,
        physics_config=physics_config,
    ) == -423.0


def test_pypower_backend_does_not_patch_numpy_in1d():
    text = (ROOT / "grid_topology_ai/pypower_backend.py").read_text()
    for token in ('np.in1d =', 'np.isin', 'hasattr(np, "in1d")'):
        assert token not in text

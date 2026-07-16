from grid_topology_ai.physical_objective import assess_physical_state, stop_allowed_for_policy
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner
from tests.test_reward_logic import _state


def test_mcts_stop_policy_matches_shared_helper_for_all_policies():
    planner = object.__new__(MCTSPlanner)
    state = _state(loadings=(110.0,))
    assessment = assess_physical_state(state.metrics)
    for policy in ("never", "always", "solved_only", "no_hard_overloads"):
        planner.config = MCTSConfig(stop_policy=policy, include_stop_action=True)
        assert planner._should_include_stop_action(state) is stop_allowed_for_policy(
            assessment,
            stop_policy=policy,
            include_stop_action=True,
        )


def test_mcts_include_stop_action_false_matches_shared_helper():
    planner = object.__new__(MCTSPlanner)
    planner.config = MCTSConfig(stop_policy="always", include_stop_action=False)
    state = _state(loadings=(90.0,))
    assessment = assess_physical_state(state.metrics)
    assert planner._should_include_stop_action(state) is stop_allowed_for_policy(
        assessment,
        stop_policy="always",
        include_stop_action=False,
    )

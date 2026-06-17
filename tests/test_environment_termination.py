import numpy as np
import pytest

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowResult
from grid_topology_ai.reward import GridFMRewardBreakdown


def _state(
    scenario_id=1,
    num_overloaded_branches=1,
    num_hard_overloaded_branches=0,
    max_loading_percent=110.0,
):
    return GridFMState(
        scenario_id=scenario_id,
        load_scenario_idx=0.0,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((1, 4), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        branch_ids=np.array([10], dtype=np.int64),
        branch_status=np.array([1], dtype=np.int64),
        metrics={
            "num_overloaded_branches": int(num_overloaded_branches),
            "num_hard_overloaded_branches": int(num_hard_overloaded_branches),
            "max_loading_percent": float(max_loading_percent),
        },
        outaged_branch_ids=[],
    )


def _reward_breakdown(done, reward=0.0):
    return GridFMRewardBreakdown(
        reward=float(reward),
        before_penalty=0.0,
        after_penalty=0.0,
        improvement=0.0,
        switching_penalty=0.0,
        before_max_loading=0.0,
        after_max_loading=0.0,
        before_total_overload=0.0,
        after_total_overload=0.0,
        before_num_overloaded=0,
        after_num_overloaded=0,
        before_num_hard_overloaded=0,
        after_num_hard_overloaded=0,
        before_voltage_penalty=0.0,
        after_voltage_penalty=0.0,
        done=bool(done),
        success=True,
        message="fake reward",
    )


class FakeAdapter:
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def build_state(self, scenario_id):
        return self.initial_state


class FakeActionSpace:
    def build_all_actions(self, state):
        return [
            GridFMAction(
                action_id=0,
                action_type="do_nothing",
            ),
            GridFMAction(
                action_id=1,
                action_type="switch_off_branch",
                branch_id=10,
                branch_pos=0,
            ),
        ]

    def valid_action_mask(self, state):
        return np.array([True, True], dtype=bool)

    def valid_actions(self, state):
        return self.build_all_actions(state)


class FakeReward:
    def __init__(self, done):
        self.done = bool(done)

    def compute(
        self,
        before_state,
        after_state,
        action_is_switching,
        power_flow_success,
    ):
        if not power_flow_success:
            return _reward_breakdown(done=False, reward=-1000.0)

        return _reward_breakdown(done=self.done, reward=10.0 if self.done else 0.0)


class FakeBackend:
    def __init__(self, success=True, next_state=None):
        self.success = bool(success)
        self.next_state = next_state

    def run_power_flow_from_state(
        self,
        state,
        switched_off_branch_id,
    ):
        return GridFMPowerFlowResult(
            success=self.success,
            scenario_id=int(state.scenario_id),
            switched_off_branch_id=switched_off_branch_id,
            next_state=self.next_state,
            raw_result=None,
            message="fake power flow",
        )


def _env(
    initial_state,
    reward_done,
    backend_success=True,
    next_state=None,
    max_steps=5,
    allow_handoff_with_hard_overloads=False,
):
    return TopologySwitchingEnv(
        adapter=FakeAdapter(initial_state),
        backend=FakeBackend(
            success=backend_success,
            next_state=next_state,
        ),
        action_space=FakeActionSpace(),
        reward_fn=FakeReward(done=reward_done),
        max_steps=max_steps,
        allow_handoff_with_hard_overloads=allow_handoff_with_hard_overloads,
    )


def test_do_nothing_solved_terminates_as_solved():
    state = _state(
        num_overloaded_branches=0,
        num_hard_overloaded_branches=0,
        max_loading_percent=90.0,
    )

    env = _env(
        initial_state=state,
        reward_done=True,
    )

    env.reset(scenario_id=1)

    result = env.step(
        GridFMAction(
            action_id=0,
            action_type="do_nothing",
        )
    )

    assert result.done is True
    assert result.solved is True
    assert result.info["termination_reason"] == "solved"


def test_do_nothing_handoff_without_hard_overload():
    state = _state(
        num_overloaded_branches=1,
        num_hard_overloaded_branches=0,
        max_loading_percent=110.0,
    )

    env = _env(
        initial_state=state,
        reward_done=False,
    )

    env.reset(scenario_id=1)

    result = env.step(
        GridFMAction(
            action_id=0,
            action_type="do_nothing",
        )
    )

    assert result.done is True
    assert result.solved is False
    assert result.info["termination_reason"] == "handoff_to_redispatch"


def test_do_nothing_hard_overload_without_handoff_permission_is_unsafe_stop():
    state = _state(
        num_overloaded_branches=1,
        num_hard_overloaded_branches=1,
        max_loading_percent=130.0,
    )

    env = _env(
        initial_state=state,
        reward_done=False,
        allow_handoff_with_hard_overloads=False,
    )

    env.reset(scenario_id=1)

    result = env.step(
        GridFMAction(
            action_id=0,
            action_type="do_nothing",
        )
    )

    assert result.done is True
    assert result.solved is False
    assert result.info["termination_reason"] == "unsafe_stop_with_hard_overload"


def test_do_nothing_hard_overload_with_handoff_permission():
    state = _state(
        num_overloaded_branches=1,
        num_hard_overloaded_branches=1,
        max_loading_percent=130.0,
    )

    env = _env(
        initial_state=state,
        reward_done=False,
        allow_handoff_with_hard_overloads=True,
    )

    env.reset(scenario_id=1)

    result = env.step(
        GridFMAction(
            action_id=0,
            action_type="do_nothing",
        )
    )

    assert result.done is True
    assert result.solved is False
    assert (
        result.info["termination_reason"]
        == "handoff_to_redispatch_with_hard_overload"
    )


def test_switch_off_branch_power_flow_failure_terminates_as_failed():
    state = _state(
        num_overloaded_branches=1,
        num_hard_overloaded_branches=1,
        max_loading_percent=130.0,
    )

    env = _env(
        initial_state=state,
        reward_done=False,
        backend_success=False,
        next_state=None,
    )

    env.reset(scenario_id=1)

    result = env.step(
        GridFMAction(
            action_id=1,
            action_type="switch_off_branch",
            branch_id=10,
            branch_pos=0,
        )
    )

    assert result.done is True
    assert result.solved is False
    assert result.power_flow_success is False
    assert result.info["termination_reason"] == "power_flow_failed"


def test_switch_off_branch_solved_terminates_as_solved():
    before_state = _state(
        num_overloaded_branches=1,
        num_hard_overloaded_branches=1,
        max_loading_percent=130.0,
    )

    after_state = _state(
        num_overloaded_branches=0,
        num_hard_overloaded_branches=0,
        max_loading_percent=90.0,
    )

    env = _env(
        initial_state=before_state,
        reward_done=True,
        backend_success=True,
        next_state=after_state,
    )

    env.reset(scenario_id=1)

    result = env.step(
        GridFMAction(
            action_id=1,
            action_type="switch_off_branch",
            branch_id=10,
            branch_pos=0,
        )
    )

    assert result.done is True
    assert result.solved is True
    assert result.power_flow_success is True
    assert result.info["termination_reason"] == "solved"


def test_switch_off_branch_max_steps_reached():
    before_state = _state(
        num_overloaded_branches=1,
        num_hard_overloaded_branches=1,
        max_loading_percent=130.0,
    )

    after_state = _state(
        num_overloaded_branches=1,
        num_hard_overloaded_branches=0,
        max_loading_percent=110.0,
    )

    env = _env(
        initial_state=before_state,
        reward_done=False,
        backend_success=True,
        next_state=after_state,
        max_steps=1,
    )

    env.reset(scenario_id=1)

    result = env.step(
        GridFMAction(
            action_id=1,
            action_type="switch_off_branch",
            branch_id=10,
            branch_pos=0,
        )
    )

    assert result.done is True
    assert result.solved is False
    assert result.power_flow_success is True
    assert result.info["termination_reason"] == "max_steps_reached"


def test_step_after_done_requires_reset():
    state = _state(
        num_overloaded_branches=0,
        num_hard_overloaded_branches=0,
        max_loading_percent=90.0,
    )

    env = _env(
        initial_state=state,
        reward_done=True,
    )

    env.reset(scenario_id=1)

    env.step(
        GridFMAction(
            action_id=0,
            action_type="do_nothing",
        )
    )

    with pytest.raises(RuntimeError, match="Episode is already done"):
        env.step(
            GridFMAction(
                action_id=0,
                action_type="do_nothing",
            )
        )
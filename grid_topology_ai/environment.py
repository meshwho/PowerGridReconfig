from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from grid_topology_ai.action_space import GridFMAction, GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter, GridFMState
from grid_topology_ai.pypower_backend import (
    GridFMPowerFlowBackend,
    GridFMPowerFlowResult,
)
from grid_topology_ai.reward import GridFMReward, GridFMRewardBreakdown


@dataclass(frozen=True)
class TopologyStepResult:
    """
    Result of one environment step.

    This is the RL-style output:
        next_state, reward, done, info

    Additional fields are included for debugging and planning.
    """

    next_state: GridFMState | None
    reward: float
    done: bool
    solved: bool
    power_flow_success: bool
    action: GridFMAction
    reward_breakdown: GridFMRewardBreakdown | None
    power_flow_result: GridFMPowerFlowResult | None
    info: dict[str, Any]


class TopologySwitchingEnv:
    """
    Multi-step topology switching environment.

    This class wraps:
        GridFMAdapter
        GridFMActionSpace
        GridFMPowerFlowBackend
        GridFMReward

    Main interface:
        state = env.reset(scenario_id)
        result = env.step(action)

    Important:
        This environment applies actions to the current state,
        not always to the original scenario.

    This is the required foundation for:
        - greedy rollout;
        - beam search;
        - MCTS;
        - AlphaZero-like self-learning.
    """

    def __init__(
        self,
        adapter: GridFMAdapter,
        backend: GridFMPowerFlowBackend,
        action_space: GridFMActionSpace,
        reward_fn: GridFMReward,
        max_steps: int = 5,
    ):
        self.adapter = adapter
        self.backend = backend
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.max_steps = int(max_steps)

        self.current_state: GridFMState | None = None
        self.initial_scenario_id: int | None = None
        self.step_count: int = 0
        self.done: bool = False
        self.solved: bool = False
        self.switched_branch_ids: list[int] = []
        self.termination_reason: str | None = None

    def reset(self, scenario_id: int) -> GridFMState:
        """
        Reset environment to one emergency scenario.
        """

        self.current_state = self.adapter.build_state(int(scenario_id))
        self.initial_scenario_id = int(scenario_id)
        self.step_count = 0
        self.done = False
        self.solved = False
        self.switched_branch_ids = []
        self.termination_reason = None

        return self.current_state

    def valid_actions(self) -> list[GridFMAction]:
        """
        Return valid actions for the current state.
        """

        self._require_active_episode()

        assert self.current_state is not None

        return self.action_space.valid_actions(self.current_state)

    def valid_action_mask(self):
        """
        Return valid action mask for the current state.
        """

        self._require_active_episode()

        assert self.current_state is not None

        return self.action_space.valid_action_mask(self.current_state)

    def action_by_id(self, action_id: int) -> GridFMAction:
        """
        Convert integer action_id to GridFMAction and validate it.
        """

        self._require_active_episode()

        assert self.current_state is not None

        all_actions = self.action_space.build_all_actions(self.current_state)

        if action_id < 0 or action_id >= len(all_actions):
            raise ValueError(f"Invalid action_id: {action_id}")

        action = all_actions[action_id]

        mask = self.action_space.valid_action_mask(self.current_state)

        if not bool(mask[action_id]):
            raise ValueError(f"Action {action_id} is not valid in current state.")

        return action

    def action_by_branch_id(self, branch_id: int) -> GridFMAction:
        """
        Find a valid switch_off_branch action by original branch ID.
        """

        for action in self.valid_actions():
            if (
                action.action_type == "switch_off_branch"
                and action.branch_id == int(branch_id)
            ):
                return action

        raise ValueError(
            f"Branch {branch_id} is not a valid switch-off action "
            f"in the current state."
        )

    def step(self, action: GridFMAction | int) -> TopologyStepResult:
        """
        Apply one action to the current state.

        If action is int, it is interpreted as action_id.
        """

        self._require_active_episode()

        assert self.current_state is not None

        if isinstance(action, int):
            action = self.action_by_id(action)

        if action.action_type == "do_nothing":
            return self._step_do_nothing(action)

        if action.action_type == "switch_off_branch":
            return self._step_switch_off_branch(action)

        raise ValueError(f"Unsupported action type: {action.action_type}")

    def clone(self) -> "TopologySwitchingEnv":
        """
        Create a copy of the environment.

        This is useful for search algorithms:
            MCTS / beam search can branch without modifying the original env.
        """

        cloned = TopologySwitchingEnv(
            adapter=self.adapter,
            backend=self.backend,
            action_space=self.action_space,
            reward_fn=self.reward_fn,
            max_steps=self.max_steps,
        )

        cloned.current_state = copy.deepcopy(self.current_state)
        cloned.initial_scenario_id = self.initial_scenario_id
        cloned.step_count = self.step_count
        cloned.done = self.done
        cloned.solved = self.solved
        cloned.switched_branch_ids = list(self.switched_branch_ids)
        cloned.termination_reason = self.termination_reason

        return cloned

    def _step_do_nothing(self, action: GridFMAction) -> TopologyStepResult:
        """
        Stop the topology switching episode.

        In a multi-step environment, do_nothing is interpreted as:

        1. solved
           if all overloads are removed;

        2. handoff_to_redispatch
           if topology switching should stop but the grid is not fully solved.

        This is important for the future architecture:
            topology switching -> if not enough -> redispatch.
        """

        assert self.current_state is not None

        reward_breakdown = self.reward_fn.compute(
            before_state=self.current_state,
            after_state=self.current_state,
            action_is_switching=False,
            power_flow_success=True,
        )

        num_overloaded = int(
            self.current_state.metrics["num_overloaded_branches"]
        )
        num_hard_overloaded = int(
            self.current_state.metrics["num_hard_overloaded_branches"]
        )

        self.done = True
        self.solved = bool(reward_breakdown.done)

        if self.solved:
            self.termination_reason = "solved"
        elif num_hard_overloaded == 0:
            self.termination_reason = "handoff_to_redispatch"
        else:
            # This should normally not happen if MCTS stop-policy is configured correctly.
            self.termination_reason = "unsafe_stop_with_hard_overload"

        return TopologyStepResult(
            next_state=self.current_state,
            reward=0.0,
            done=True,
            solved=self.solved,
            power_flow_success=True,
            action=action,
            reward_breakdown=reward_breakdown,
            power_flow_result=None,
            info=self._info(),
        )

    def _step_switch_off_branch(self, action: GridFMAction) -> TopologyStepResult:
        """
        Switch off one branch and run power flow from the current state.
        """

        assert self.current_state is not None

        before_state = self.current_state

        power_flow_result = self.backend.run_power_flow_from_state(
            state=before_state,
            switched_off_branch_id=action.branch_id,
        )

        reward_breakdown = self.reward_fn.compute(
            before_state=before_state,
            after_state=power_flow_result.next_state,
            action_is_switching=True,
            power_flow_success=power_flow_result.success,
        )

        self.step_count += 1

        if action.branch_id is not None:
            self.switched_branch_ids.append(int(action.branch_id))

        if not power_flow_result.success or power_flow_result.next_state is None:
            self.done = True
            self.solved = False
            self.termination_reason = "power_flow_failed"

            return TopologyStepResult(
                next_state=None,
                reward=float(reward_breakdown.reward),
                done=True,
                solved=False,
                power_flow_success=False,
                action=action,
                reward_breakdown=reward_breakdown,
                power_flow_result=power_flow_result,
                info=self._info(),
            )

        self.current_state = power_flow_result.next_state

        self.solved = bool(reward_breakdown.done)

        if self.solved:
            self.done = True
            self.termination_reason = "solved"
        elif self.step_count >= self.max_steps:
            self.done = True
            self.termination_reason = "max_steps_reached"
        else:
            self.done = False
            self.termination_reason = None

        return TopologyStepResult(
            next_state=self.current_state,
            reward=float(reward_breakdown.reward),
            done=bool(self.done),
            solved=bool(self.solved),
            power_flow_success=True,
            action=action,
            reward_breakdown=reward_breakdown,
            power_flow_result=power_flow_result,
            info=self._info(),
        )

    def _info(self) -> dict[str, Any]:
        """
        Build debugging info dictionary.
        """

        return {
            "initial_scenario_id": self.initial_scenario_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "solved": self.solved,
            "termination_reason": self.termination_reason,
            "switched_branch_ids": list(self.switched_branch_ids),
        }

    def _require_active_episode(self) -> None:
        """
        Ensure that reset() was called and episode is not already done.
        """

        if self.current_state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        if self.done:
            raise RuntimeError(
                f"Episode is already done. "
                f"Termination reason: {self.termination_reason}. "
                f"Call reset() to start a new episode."
            )
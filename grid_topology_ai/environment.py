from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grid_topology_ai.actions import GridFMAction, GridFMActionSpace
from grid_topology_ai.data import GridFMAdapter
from grid_topology_ai.state import GridFMState
from grid_topology_ai.physics.utility import state_utility
from grid_topology_ai.physics.objective import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
    RedispatchStatus,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.power_flow.backend import (
    GridFMPowerFlowBackend,
    GridFMPowerFlowResult,
)
from grid_topology_ai.physics.objective import (
    PhysicalStateAssessment,
    assess_physical_state,
    classify_stop_outcome,
)
from grid_topology_ai.physics.utility import GridFMReward, GridFMRewardBreakdown
from grid_topology_ai.termination import (
    TerminationReason,
    termination_reason_value,
)


@dataclass(frozen=True)
class TopologyStepResult:
    """Result of one environment step."""

    next_state: GridFMState | None
    reward: float
    done: bool
    solved: bool
    power_flow_success: bool
    action: GridFMAction
    reward_breakdown: GridFMRewardBreakdown | None
    power_flow_result: GridFMPowerFlowResult | None
    terminal_outcome_evidence: TerminalOutcomeEvidence | None
    info: dict[str, Any]


class TopologySwitchingEnv:
    """Multi-step topology switching environment."""

    def __init__(
        self,
        adapter: GridFMAdapter,
        backend: GridFMPowerFlowBackend,
        action_space: GridFMActionSpace,
        reward_fn: GridFMReward,
        max_steps: int = 5,
        allow_handoff_with_hard_overloads: bool = False,
    ):
        self.adapter = adapter
        self.backend = backend
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.max_steps = int(max_steps)
        self.allow_handoff_with_hard_overloads = bool(
            allow_handoff_with_hard_overloads
        )
        self.current_state: GridFMState | None = None
        self.initial_state: GridFMState | None = None
        self.initial_scenario_id: int | None = None
        self.step_count: int = 0
        self.done: bool = False
        self.solved: bool = False
        self.switched_branch_ids: list[int] = []
        self.applied_actions: list[GridFMAction] = []
        self.termination_reason: TerminationReason | None = None
        self.terminal_outcome_evidence: TerminalOutcomeEvidence | None = None
        self._valid_actions_by_id: dict[int, GridFMAction] | None = None

    def reset(self, scenario_id: int) -> GridFMState:
        """Reset through the canonical AC power-flow backend."""

        scenario_id = int(scenario_id)
        self.current_state = None
        self.initial_state = None
        self.initial_scenario_id = scenario_id
        self.step_count = 0
        self.done = False
        self.solved = False
        self.switched_branch_ids = []
        self.applied_actions = []
        self.termination_reason = None
        self.terminal_outcome_evidence = None
        self._valid_actions_by_id = None

        initial_result = self.backend.run_power_flow(
            scenario_id=scenario_id,
            switched_off_branch_id=None,
        )
        if not initial_result.success or initial_result.next_state is None:
            self._finish_episode(
                solved=False,
                termination_reason=TerminationReason.POWER_FLOW_FAILED,
                assessment=None,
            )
            failure_kind = (
                initial_result.failure_kind.value
                if initial_result.failure_kind is not None
                else "unknown"
            )
            raise RuntimeError(
                "Environment reset could not build a canonical initial state "
                f"for scenario {scenario_id} "
                f"(failure_kind={failure_kind}): {initial_result.message}"
            )

        self.current_state = initial_result.next_state
        self.initial_state = self.current_state
        assessment = assess_physical_state(self.current_state.metrics)
        if assessment.physically_secure:
            self._finish_episode(
                solved=True,
                termination_reason=TerminationReason.SOLVED,
                assessment=assessment,
            )

        return self.current_state

    def _valid_actions_for_current_state(self) -> dict[int, GridFMAction]:
        self._require_active_episode()
        assert self.current_state is not None

        if self._valid_actions_by_id is None:
            actions = self.action_space.valid_actions(self.current_state)
            self._valid_actions_by_id = {
                int(action.action_id): action
                for action in actions
            }
        return self._valid_actions_by_id

    def valid_actions(self) -> list[GridFMAction]:
        return list(self._valid_actions_for_current_state().values())

    def structural_action_mask(self):
        self._require_active_episode()
        assert self.current_state is not None
        return self.action_space.structural_action_mask(self.current_state)

    def operational_action_mask(self):
        self._require_active_episode()
        assert self.current_state is not None
        return self.action_space.operational_action_mask(self.current_state)

    def action_by_id(self, action_id: int) -> GridFMAction:
        self._require_active_episode()
        assert self.current_state is not None

        num_branches = len(self.current_state.branch_ids)
        if action_id < 0 or action_id > num_branches:
            raise ValueError(f"Invalid action_id: {action_id}")

        action = self._valid_actions_for_current_state().get(int(action_id))
        if action is None:
            raise ValueError(
                f"Action {action_id} is not valid in current state."
            )
        return action

    def action_by_branch_id(self, branch_id: int) -> GridFMAction:
        branch_id = int(branch_id)
        for action in self.valid_actions():
            if (
                action.kind == "set_branch_status"
                and action.branch_id == branch_id
            ):
                return action
        raise ValueError(
            f"Branch {branch_id} has no valid status-change action "
            "in the current state."
        )

    def step(self, action: GridFMAction | int) -> TopologyStepResult:
        self._require_active_episode()

        if isinstance(action, int):
            action = self.action_by_id(action)
        else:
            canonical_action = self.action_by_id(int(action.action_id))
            if action != canonical_action:
                raise ValueError(
                    "Topology action is stale or does not match the current "
                    "state."
                )
            action = canonical_action

        if action.kind == "stop":
            return self._step_do_nothing(action)
        if action.kind == "set_branch_status":
            return self._step_branch_status(action)
        raise ValueError(f"Unsupported action kind: {action.kind!r}.")

    def clone(self) -> "TopologySwitchingEnv":
        cloned = TopologySwitchingEnv(
            adapter=self.adapter,
            backend=self.backend,
            action_space=self.action_space,
            reward_fn=self.reward_fn,
            max_steps=self.max_steps,
            allow_handoff_with_hard_overloads=(
                self.allow_handoff_with_hard_overloads
            ),
        )
        cloned.current_state = self.current_state
        cloned.initial_state = self.initial_state
        cloned.initial_scenario_id = self.initial_scenario_id
        cloned.step_count = self.step_count
        cloned.done = self.done
        cloned.solved = self.solved
        cloned.switched_branch_ids = list(self.switched_branch_ids)
        cloned.applied_actions = list(self.applied_actions)
        cloned.termination_reason = self.termination_reason
        cloned.terminal_outcome_evidence = self.terminal_outcome_evidence
        cloned._valid_actions_by_id = (
            None
            if self._valid_actions_by_id is None
            else dict(self._valid_actions_by_id)
        )
        return cloned

    def terminate_no_legal_action(self) -> TerminalOutcomeEvidence:
        """Finish an active episode when search has no executable action."""

        self._require_active_episode()
        assert self.current_state is not None
        assessment = assess_physical_state(self.current_state.metrics)
        return self._finish_episode(
            solved=False,
            termination_reason=TerminationReason.NO_LEGAL_ACTION,
            assessment=assessment,
        )

    def _step_do_nothing(self, action: GridFMAction) -> TopologyStepResult:
        assert self.current_state is not None

        assessment = assess_physical_state(self.current_state.metrics)
        reward_breakdown = self.reward_fn.compute(
            before_state=self.current_state,
            after_state=self.current_state,
            power_flow_success=assessment.power_flow_converged,
        )
        outcome = classify_stop_outcome(
            assessment,
            allow_handoff_with_hard_overloads=(
                self.allow_handoff_with_hard_overloads
            ),
        )
        evidence = self._finish_episode(
            solved=outcome.solved,
            termination_reason=outcome.termination_reason,
            assessment=assessment,
        )

        return TopologyStepResult(
            next_state=self.current_state,
            reward=float(reward_breakdown.reward),
            done=True,
            solved=self.solved,
            power_flow_success=assessment.power_flow_converged,
            action=action,
            reward_breakdown=reward_breakdown,
            power_flow_result=None,
            terminal_outcome_evidence=evidence,
            info=self._info(),
        )

    def _step_branch_status(
        self,
        action: GridFMAction,
    ) -> TopologyStepResult:
        assert self.current_state is not None

        before_state = self.current_state
        power_flow_result = self.backend.run_power_flow_from_state(
            state=before_state,
            action=action,
        )
        reward_breakdown = self.reward_fn.compute(
            before_state=before_state,
            after_state=power_flow_result.next_state,
            power_flow_success=power_flow_result.success,
        )

        self.step_count += 1
        self.applied_actions.append(action)
        if action.branch_id is not None:
            self.switched_branch_ids.append(int(action.branch_id))

        if not power_flow_result.success or power_flow_result.next_state is None:
            evidence = self._finish_episode(
                solved=False,
                termination_reason=TerminationReason.POWER_FLOW_FAILED,
                assessment=None,
            )
            return TopologyStepResult(
                next_state=None,
                reward=float(reward_breakdown.reward),
                done=True,
                solved=False,
                power_flow_success=False,
                action=action,
                reward_breakdown=reward_breakdown,
                power_flow_result=power_flow_result,
                terminal_outcome_evidence=evidence,
                info=self._info(),
            )

        self.current_state = power_flow_result.next_state
        self._valid_actions_by_id = None
        assessment = assess_physical_state(self.current_state.metrics)

        if assessment.physically_secure:
            evidence = self._finish_episode(
                solved=True,
                termination_reason=TerminationReason.SOLVED,
                assessment=assessment,
            )
        elif self.step_count >= self.max_steps:
            evidence = self._finish_episode(
                solved=False,
                termination_reason=TerminationReason.MAX_STEPS_REACHED,
                assessment=assessment,
            )
        else:
            self.done = False
            self.solved = False
            self.termination_reason = None
            self.terminal_outcome_evidence = None
            evidence = None

        return TopologyStepResult(
            next_state=self.current_state,
            reward=float(reward_breakdown.reward),
            done=bool(self.done),
            solved=bool(self.solved),
            power_flow_success=True,
            action=action,
            reward_breakdown=reward_breakdown,
            power_flow_result=power_flow_result,
            terminal_outcome_evidence=evidence,
            info=self._info(),
        )

    def _finish_episode(
        self,
        *,
        solved: bool,
        termination_reason: TerminationReason,
        assessment: PhysicalStateAssessment | None,
    ) -> TerminalOutcomeEvidence:
        topology_value = -1.0
        if assessment is not None:
            if self.current_state is None:
                raise RuntimeError(
                    "Terminal physical assessment requires a current state."
                )
            topology_value = state_utility(
                self.current_state,
                physics_config=self.backend.physics_config,
            )

        evidence = TerminalOutcomeEvidence(
            solved=bool(solved),
            termination_reason=termination_reason,
            assessment=assessment,
            redispatch_status=RedispatchStatus.NOT_REQUESTED,
            topology_utility=topology_value,
        )
        self.done = True
        self.solved = evidence.solved
        self.termination_reason = evidence.termination_reason
        self.terminal_outcome_evidence = evidence
        return evidence

    def _info(self) -> dict[str, Any]:
        evidence = self.terminal_outcome_evidence
        return {
            "initial_scenario_id": self.initial_scenario_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "solved": self.solved,
            "termination_reason": self.termination_reason,
            "termination_reason_value": termination_reason_value(
                self.termination_reason
            ),
            "terminal_outcome_evidence_schema_version": (
                None
                if evidence is None
                else TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
            ),
            "terminal_outcome_evidence": (
                None if evidence is None else evidence.to_dict()
            ),
            "switched_branch_ids": list(self.switched_branch_ids),
            "applied_actions": [
                {
                    "action_id": int(action.action_id),
                    "action_type": str(action.action_type),
                    "branch_id": (
                        None
                        if action.branch_id is None
                        else int(action.branch_id)
                    ),
                    "target_status": (
                        None
                        if action.target_status is None
                        else int(action.target_status)
                    ),
                }
                for action in self.applied_actions
            ],
        }

    def _require_active_episode(self) -> None:
        if self.current_state is None:
            raise RuntimeError(
                "Environment is not initialized. Call reset() first."
            )
        if self.done:
            raise RuntimeError(
                "Episode is already done. "
                f"Termination reason: {self.termination_reason}. "
                "Call reset() to start a new episode."
            )

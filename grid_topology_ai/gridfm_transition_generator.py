from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from grid_topology_ai.gridfm_action_space import GridFMAction, GridFMActionSpace
from grid_topology_ai.gridfm_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter, GridFMState
from grid_topology_ai.gridfm_pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.gridfm_reward import GridFMReward, GridFMRewardBreakdown


@dataclass(frozen=True)
class GridFMTransitionRecord:
    """
    One transition record for RL / AlphaZero-style training.

    This is the tabular metadata.

    The full state tensors will be saved separately later.
    For now this CSV is enough to analyze action quality.
    """

    transition_id: int
    scenario_id: int

    action_id: int
    action_type: str
    branch_id: int | None
    branch_pos: int | None

    power_flow_success: bool
    reward: float
    done: bool

    before_max_loading: float
    after_max_loading: float

    before_total_overload: float
    after_total_overload: float

    before_num_overloaded: int
    after_num_overloaded: int

    before_num_hard_overloaded: int
    after_num_hard_overloaded: int

    before_voltage_penalty: float
    after_voltage_penalty: float

    before_penalty: float
    after_penalty: float
    improvement: float
    switching_penalty: float

    message: str


class GridFMTransitionGenerator:
    """
    Generate transition dataset from GridFM emergency scenarios.

    Pipeline:
        GridFM scenario
        -> build state
        -> build valid actions
        -> apply each action through PYPOWER backend
        -> compute reward
        -> save transition records

    This is the first real RL dataset layer.
    """

    def __init__(
        self,
        adapter: GridFMAdapter,
        backend: GridFMPowerFlowBackend,
        action_space: GridFMActionSpace,
        reward_fn: GridFMReward,
    ):
        self.adapter = adapter
        self.backend = backend
        self.action_space = action_space
        self.reward_fn = reward_fn

    def generate_for_useful_scenarios(
        self,
        max_switch_actions_per_scenario: int | None = None,
        include_do_nothing: bool = True,
    ) -> pd.DataFrame:
        """
        Generate transitions for all useful emergency scenarios.

        Parameters
        ----------
        max_switch_actions_per_scenario:
            If None, evaluate all valid switch-off actions.
            If integer, evaluate only top-K valid switch-off actions by current loading.

        include_do_nothing:
            Whether to include the do_nothing action.

        Why top-K option exists:
            For a large dataset, evaluating every valid action can be expensive.
            For the first MVP, None is okay.
        """

        scenario_ids = self.adapter.useful_scenario_ids()

        records: list[GridFMTransitionRecord] = []
        transition_id = 0

        for scenario_id in tqdm(scenario_ids, desc="Generating transitions"):
            state = self.adapter.build_state(scenario_id)

            actions = self._select_actions(
                state=state,
                max_switch_actions=max_switch_actions_per_scenario,
                include_do_nothing=include_do_nothing,
            )

            for action in actions:
                record = self._evaluate_action(
                    transition_id=transition_id,
                    state=state,
                    action=action,
                )

                records.append(record)
                transition_id += 1

        return pd.DataFrame([asdict(record) for record in records])

    def _select_actions(
        self,
        state: GridFMState,
        max_switch_actions: int | None,
        include_do_nothing: bool,
    ) -> list[GridFMAction]:
        """
        Select actions to evaluate.

        For first experiments:
            - include do_nothing;
            - evaluate all valid switch-off actions or top-K by loading.
        """

        valid_actions = self.action_space.valid_actions(state)

        do_nothing_actions = [
            action for action in valid_actions if action.action_type == "do_nothing"
        ]

        switch_actions = [
            action for action in valid_actions if action.action_type == "switch_off_branch"
        ]

        if max_switch_actions is not None:
            loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

            switch_actions = sorted(
                switch_actions,
                key=lambda action: float(
                    state.branch_features[action.branch_pos, loading_idx]
                ),
                reverse=True,
            )

            switch_actions = switch_actions[:max_switch_actions]

        selected: list[GridFMAction] = []

        if include_do_nothing:
            selected.extend(do_nothing_actions)

        selected.extend(switch_actions)

        return selected

    def _evaluate_action(
        self,
        transition_id: int,
        state: GridFMState,
        action: GridFMAction,
    ) -> GridFMTransitionRecord:
        """
        Apply one action, run power flow, compute reward, return transition record.
        """

        if action.action_type == "do_nothing":
            # For do_nothing we intentionally do not rerun PYPOWER.
            # The next state is identical to the current state.
            # This prevents tiny backend differences from creating fake reward.
            reward = self.reward_fn.compute(
                before_state=state,
                after_state=state,
                action_is_switching=False,
                power_flow_success=True,
            )

            return self._make_record(
                transition_id=transition_id,
                state=state,
                action=action,
                reward=reward,
                power_flow_success=True,
                message="Do nothing.",
            )

        result = self.backend.run_power_flow(
            scenario_id=state.scenario_id,
            switched_off_branch_id=action.branch_id,
        )

        reward = self.reward_fn.compute(
            before_state=state,
            after_state=result.next_state,
            action_is_switching=True,
            power_flow_success=result.success,
        )

        return self._make_record(
            transition_id=transition_id,
            state=state,
            action=action,
            reward=reward,
            power_flow_success=result.success,
            message=result.message,
        )

    @staticmethod
    def _make_record(
        transition_id: int,
        state: GridFMState,
        action: GridFMAction,
        reward: GridFMRewardBreakdown,
        power_flow_success: bool,
        message: str,
    ) -> GridFMTransitionRecord:
        """
        Convert evaluated transition to a flat table row.
        """

        return GridFMTransitionRecord(
            transition_id=int(transition_id),
            scenario_id=int(state.scenario_id),
            action_id=int(action.action_id),
            action_type=str(action.action_type),
            branch_id=None if action.branch_id is None else int(action.branch_id),
            branch_pos=None if action.branch_pos is None else int(action.branch_pos),
            power_flow_success=bool(power_flow_success),
            reward=float(reward.reward),
            done=bool(reward.done),
            before_max_loading=float(reward.before_max_loading),
            after_max_loading=float(reward.after_max_loading),
            before_total_overload=float(reward.before_total_overload),
            after_total_overload=float(reward.after_total_overload),
            before_num_overloaded=int(reward.before_num_overloaded),
            after_num_overloaded=int(reward.after_num_overloaded),
            before_num_hard_overloaded=int(reward.before_num_hard_overloaded),
            after_num_hard_overloaded=int(reward.after_num_hard_overloaded),
            before_voltage_penalty=float(reward.before_voltage_penalty),
            after_voltage_penalty=float(reward.after_voltage_penalty),
            before_penalty=float(reward.before_penalty),
            after_penalty=float(reward.after_penalty),
            improvement=float(reward.improvement),
            switching_penalty=float(reward.switching_penalty),
            message=str(message),
        )


def save_transitions(transitions: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save transition table to CSV.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transitions.to_csv(output_path, index=False)
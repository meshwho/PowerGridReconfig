from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import (
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.search.root_policy import (
    normalize_policy,
    require_action_in_policy_support,
)
from grid_topology_ai.state_store import GridFMStateStore
from grid_topology_ai.termination import (
    TerminationReason,
    termination_reason_value,
    validate_outcome_invariants,
)


@dataclass(frozen=True)
class SelfPlayExample:
    """One on-policy AlphaZero-style self-play example.

    ``mcts_policy_json`` is the policy-head target. ``step_reward``,
    ``final_return``, and ``discounted_return_from_step`` are diagnostic
    potential-shaping fields only. The value-head target is the separately
    derived ``outcome_value_target`` under the discounted terminal-utility
    contract.
    """

    state_id: str
    state_path: str
    scenario_id: int
    step: int
    selected_action_id: int
    selected_branch_id: int | None
    step_reward: float
    final_return: float
    discounted_return_from_step: float
    solved: bool
    done: bool
    termination_reason: str | None
    physical_objective_schema_version: int
    outcome_value_target_contract_version: int
    physics_config_contract_version: int
    topology_action_contract_version: int
    topology_action_config: str
    topology_action_config_fingerprint: str
    action_layout: str
    action_layout_fingerprint: str
    physics_config: str
    physics_config_fingerprint: str
    visit_counts_json: str
    mcts_policy_json: str
    selection_temperature: float | None = None
    selection_mode: str | None = None
    policy_target_entropy: float | None = None
    policy_target_normalized_entropy: float | None = None
    mcts_legal_action_count: int | None = None
    mcts_considered_action_count: int | None = None
    mcts_visited_action_count: int | None = None
    mcts_action_coverage: float | None = None
    mcts_visited_action_coverage: float | None = None


class ExampleWriter:
    """Save self-play tensors plus on-policy and diagnostic metadata."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        physics_config: PhysicsConfig,
        action_space_config: ActionSpaceConfig,
    ):
        self.output_dir = Path(output_dir)
        self.physics_config = physics_config
        self.states_dir = self.output_dir / "states"
        self.examples_path = self.output_dir / "examples.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.state_store = GridFMStateStore(self.states_dir)
        self.examples: list[SelfPlayExample] = []
        self.action_space_config = action_space_config

    def add_example(
        self,
        state: GridFMState,
        state_id: str,
        action_mask,
        scenario_id: int,
        step: int,
        selected_action_id: int,
        selected_branch_id: int | None,
        step_reward: float,
        final_return: float,
        discounted_return_from_step: float,
        solved: bool,
        done: bool,
        termination_reason: TerminationReason | str | None,
        visit_counts: dict[int, int],
        mcts_policy: dict[int, float],
        selection_temperature: float | None = None,
        selection_mode: str | None = None,
        policy_target_entropy: float | None = None,
        policy_target_normalized_entropy: float | None = None,
        mcts_legal_action_count: int | None = None,
        mcts_considered_action_count: int | None = None,
        mcts_visited_action_count: int | None = None,
        mcts_action_coverage: float | None = None,
        mcts_visited_action_coverage: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save one strictly on-policy self-play example."""

        policy_context = f"self-play example {state_id!r}"
        normalized_policy = normalize_policy(
            mcts_policy,
            context=policy_context,
        )
        require_action_in_policy_support(
            selected_action_id,
            normalized_policy,
            context=policy_context,
        )

        provenance = physics_provenance(self.physics_config)
        state_metadata = dict(extra_metadata or {})
        state_metadata.update(provenance)
        state_metadata["outcome_value_target_contract_version"] = (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        )

        action_layout = (
            build_branch_action_slots(
                state.branch_ids
            )
        )
        action_provenance = (
            topology_action_provenance(
                self.action_space_config,
                action_layout,
            )
        )

        state_metadata.update(
            action_provenance
        )

        state_path = self.state_store.save_state(
            state=state,
            state_id=state_id,
            action_mask=action_mask,
            extra_metadata=state_metadata,
        )

        parsed_reason = validate_outcome_invariants(
            solved=bool(solved),
            termination_reason=termination_reason,
        )
        example = SelfPlayExample(
            state_id=state_id,
            state_path=str(state_path),
            scenario_id=int(scenario_id),
            step=int(step),
            selected_action_id=int(selected_action_id),
            selected_branch_id=(
                None if selected_branch_id is None else int(selected_branch_id)
            ),
            step_reward=float(step_reward),
            final_return=float(final_return),
            discounted_return_from_step=float(discounted_return_from_step),
            solved=bool(solved),
            done=bool(done),
            termination_reason=termination_reason_value(parsed_reason),
            physical_objective_schema_version=(
                PHYSICAL_OBJECTIVE_SCHEMA_VERSION
            ),
            outcome_value_target_contract_version=(
                OUTCOME_VALUE_TARGET_CONTRACT_VERSION
            ),
            physics_config_contract_version=int(
                provenance["physics_config_contract_version"]
            ),
            topology_action_contract_version=int(
                action_provenance[
                    "topology_action_contract_version"
                ]
            ),
            topology_action_config=json.dumps(
                action_provenance[
                    "topology_action_config"
                ],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            topology_action_config_fingerprint=str(
                action_provenance[
                    "topology_action_config_fingerprint"
                ]
            ),
            action_layout=json.dumps(
                action_provenance[
                    "action_layout"
                ],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            action_layout_fingerprint=str(
                action_provenance[
                    "action_layout_fingerprint"
                ]
            ),
            physics_config=json.dumps(
                provenance["physics_config"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            physics_config_fingerprint=str(
                provenance["physics_config_fingerprint"]
            ),
            visit_counts_json=json.dumps(
                {str(key): int(value) for key, value in visit_counts.items()}
            ),
            mcts_policy_json=json.dumps(
                {
                    str(action_id): float(probability)
                    for action_id, probability in normalized_policy.items()
                }
            ),
            selection_temperature=(
                None
                if selection_temperature is None
                else float(selection_temperature)
            ),
            selection_mode=(
                None
                if selection_mode is None
                else str(selection_mode)
            ),
            policy_target_entropy=(
                None
                if policy_target_entropy is None
                else float(policy_target_entropy)
            ),
            policy_target_normalized_entropy=(
                None
                if policy_target_normalized_entropy is None
                else float(policy_target_normalized_entropy)
            ),
            mcts_legal_action_count=(
                None
                if mcts_legal_action_count is None
                else int(mcts_legal_action_count)
            ),
            mcts_considered_action_count=(
                None
                if mcts_considered_action_count is None
                else int(mcts_considered_action_count)
            ),
            mcts_visited_action_count=(
                None
                if mcts_visited_action_count is None
                else int(mcts_visited_action_count)
            ),
            mcts_action_coverage=(
                None
                if mcts_action_coverage is None
                else float(mcts_action_coverage)
            ),
            mcts_visited_action_coverage=(
                None
                if mcts_visited_action_coverage is None
                else float(mcts_visited_action_coverage)
            ),
        )
        self.examples.append(example)

    def save(self) -> Path:
        """Save all examples to CSV."""

        df = pd.DataFrame([asdict(example) for example in self.examples])
        df.to_csv(self.examples_path, index=False)
        return self.examples_path

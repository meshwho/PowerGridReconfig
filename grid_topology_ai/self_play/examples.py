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
    physics_config: str
    physics_config_fingerprint: str
    visit_counts_json: str
    mcts_policy_json: str


class ExampleWriter:
    """Save self-play tensors plus on-policy and diagnostic metadata."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        physics_config: PhysicsConfig,
    ):
        self.output_dir = Path(output_dir)
        self.physics_config = physics_config
        self.states_dir = self.output_dir / "states"
        self.examples_path = self.output_dir / "examples.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.state_store = GridFMStateStore(self.states_dir)
        self.examples: list[SelfPlayExample] = []

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
        )
        self.examples.append(example)

    def save(self) -> Path:
        """Save all examples to CSV."""

        df = pd.DataFrame([asdict(example) for example in self.examples])
        df.to_csv(self.examples_path, index=False)
        return self.examples_path

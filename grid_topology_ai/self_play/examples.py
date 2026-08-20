from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import (
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
    topology_action_provenance,
)
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.outcome_record import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.physics.objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.reward import TERMINAL_UTILITY_GAMMA
from grid_topology_ai.search.root_policy import (
    normalize_policy,
    require_action_in_policy_support,
)
from grid_topology_ai.state.store import GridFMStateStore
from grid_topology_ai.termination import (
    TerminationReason,
    termination_reason_value,
    validate_outcome_invariants,
)
from grid_topology_ai.topology_actions import (
    ActionSpaceConfig,
    build_branch_action_slots,
)
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows


@dataclass(frozen=True)
class SelfPlayExample:
    """One on-policy AlphaZero-style self-play example."""

    state_id: str
    state_path: str
    run_id: str
    iteration: int
    episode_id: str
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
    terminal_outcome_evidence_schema_version: int
    terminal_outcome_evidence_json: str
    physical_objective_schema_version: int
    outcome_objective_version: int
    outcome_value_target_contract_version: int
    state_feature_schema_version: int
    state_feature_schema_fingerprint: str
    bus_feature_columns: str
    branch_feature_columns: str
    edge_index_semantics: str
    bus_id_semantics: str
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
    outcome_value_target: float | None = None
    outcome_class: str | None = None
    outcome_steps_to_terminal: int | None = None
    outcome_value_target_mode: str | None = None
    outcome_gamma: float | None = None
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
        self.run_id = uuid.uuid4().hex
        self._episode_ids: dict[int, str] = {}

    def add_episode(
        self,
        pending_examples: list[dict[str, Any]],
        *,
        final_return: float,
        returns_from_step: list[float],
        solved: bool,
        done: bool,
        termination_reason: TerminationReason | str | None,
        terminal_outcome_evidence: TerminalOutcomeEvidence,
        iteration: int,
    ) -> int:
        """Validate and write one complete episode as a single unit."""

        if not pending_examples:
            raise ValueError("pending_examples must not be empty.")
        if len(pending_examples) != len(returns_from_step):
            raise ValueError(
                "returns_from_step must contain one value per episode step."
            )

        iteration = self._require_iteration(iteration)
        parsed_reason = self._validate_terminal_outcome(
            solved=solved,
            done=done,
            termination_reason=termination_reason,
            evidence=terminal_outcome_evidence,
        )

        scenario_ids = {
            int(item["scenario_id"])
            for item in pending_examples
        }
        if len(scenario_ids) != 1:
            raise ValueError("One episode cannot contain multiple scenario IDs.")
        scenario_id = scenario_ids.pop()

        steps = [int(item["step"]) for item in pending_examples]
        if steps != list(range(len(pending_examples))):
            raise ValueError(
                "Episode steps must be contiguous and ordered from zero."
            )

        episode_id = uuid.uuid4().hex
        evidence_json = terminal_outcome_evidence.to_json()
        reason_value = termination_reason_value(parsed_reason)
        target_rows = [
            {
                "run_id": self.run_id,
                "iteration": iteration,
                "episode_id": episode_id,
                "scenario_id": scenario_id,
                "step": step,
                "done": True,
                "solved": bool(solved),
                "termination_reason": reason_value,
                "terminal_outcome_evidence_schema_version": (
                    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
                ),
                "terminal_outcome_evidence_json": evidence_json,
                "physical_objective_schema_version": (
                    PHYSICAL_OBJECTIVE_SCHEMA_VERSION
                ),
            }
            for step in steps
        ]
        add_outcome_value_targets_to_rows(
            target_rows,
            gamma=TERMINAL_UTILITY_GAMMA,
            group_keys=("episode_id",),
        )

        original_count = len(self.examples)
        previous_episode_id = self._episode_ids.get(scenario_id)
        created_paths: list[Path] = []
        self._episode_ids[scenario_id] = episode_id

        try:
            for item, target, return_from_step in zip(
                pending_examples,
                target_rows,
                returns_from_step,
            ):
                state_path = self.states_dir / (
                    f"{episode_id}_step_{int(item['step']):03d}.npz"
                )
                created_paths.append(state_path)
                self._store_example(
                    state=item["state"],
                    action_mask=item["action_mask"],
                    scenario_id=scenario_id,
                    step=int(item["step"]),
                    selected_action_id=int(item["selected_action_id"]),
                    selected_branch_id=item.get("selected_branch_id"),
                    step_reward=float(item["step_reward"]),
                    final_return=float(final_return),
                    discounted_return_from_step=float(return_from_step),
                    solved=bool(solved),
                    termination_reason=parsed_reason,
                    terminal_outcome_evidence=terminal_outcome_evidence,
                    visit_counts=item["visit_counts"],
                    mcts_policy=item["mcts_policy"],
                    iteration=iteration,
                    episode_id=episode_id,
                    selection_temperature=item.get("selection_temperature"),
                    selection_mode=item.get("selection_mode"),
                    policy_target_entropy=item.get("policy_target_entropy"),
                    policy_target_normalized_entropy=item.get(
                        "policy_target_normalized_entropy"
                    ),
                    mcts_legal_action_count=item.get(
                        "mcts_legal_action_count"
                    ),
                    mcts_considered_action_count=item.get(
                        "mcts_considered_action_count"
                    ),
                    mcts_visited_action_count=item.get(
                        "mcts_visited_action_count"
                    ),
                    mcts_action_coverage=item.get("mcts_action_coverage"),
                    mcts_visited_action_coverage=item.get(
                        "mcts_visited_action_coverage"
                    ),
                    extra_metadata=item.get("extra_metadata"),
                    outcome_fields=target,
                )
        except Exception:
            for path in created_paths:
                path.unlink(missing_ok=True)
            del self.examples[original_count:]
            if previous_episode_id is None:
                self._episode_ids.pop(scenario_id, None)
            else:
                self._episode_ids[scenario_id] = previous_episode_id
            raise

        return len(pending_examples)

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
        terminal_outcome_evidence: TerminalOutcomeEvidence,
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
        """Save one terminal example for legacy one-row producers."""

        del state_id
        metadata = dict(extra_metadata or {})
        iteration = self._require_iteration(
            metadata.get(
                "self_play_iteration",
                metadata.get("iteration", 1),
            )
        )
        parsed_reason = self._validate_terminal_outcome(
            solved=solved,
            done=done,
            termination_reason=termination_reason,
            evidence=terminal_outcome_evidence,
        )

        scenario_id = int(scenario_id)
        step = int(step)
        if step == 0 or scenario_id not in self._episode_ids:
            self._episode_ids[scenario_id] = uuid.uuid4().hex

        self._store_example(
            state=state,
            action_mask=action_mask,
            scenario_id=scenario_id,
            step=step,
            selected_action_id=selected_action_id,
            selected_branch_id=selected_branch_id,
            step_reward=step_reward,
            final_return=final_return,
            discounted_return_from_step=discounted_return_from_step,
            solved=solved,
            termination_reason=parsed_reason,
            terminal_outcome_evidence=terminal_outcome_evidence,
            visit_counts=visit_counts,
            mcts_policy=mcts_policy,
            iteration=iteration,
            episode_id=self._episode_ids[scenario_id],
            selection_temperature=selection_temperature,
            selection_mode=selection_mode,
            policy_target_entropy=policy_target_entropy,
            policy_target_normalized_entropy=(
                policy_target_normalized_entropy
            ),
            mcts_legal_action_count=mcts_legal_action_count,
            mcts_considered_action_count=mcts_considered_action_count,
            mcts_visited_action_count=mcts_visited_action_count,
            mcts_action_coverage=mcts_action_coverage,
            mcts_visited_action_coverage=mcts_visited_action_coverage,
            extra_metadata=metadata,
            outcome_fields=None,
        )

    @staticmethod
    def _require_iteration(value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("iteration must be a positive integer.")
        try:
            iteration = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "iteration must be a positive integer."
            ) from exc
        if iteration <= 0:
            raise ValueError("iteration must be a positive integer.")
        return iteration

    @staticmethod
    def _validate_terminal_outcome(
        *,
        solved: bool,
        done: bool,
        termination_reason: TerminationReason | str | None,
        evidence: TerminalOutcomeEvidence,
    ) -> TerminationReason:
        if not isinstance(evidence, TerminalOutcomeEvidence):
            raise TypeError(
                "terminal_outcome_evidence must be "
                "TerminalOutcomeEvidence."
            )
        if not isinstance(done, bool) or not done:
            raise ValueError(
                "Self-play examples require a completed terminal episode."
            )

        parsed_reason = validate_outcome_invariants(
            solved=bool(solved),
            termination_reason=termination_reason,
        )
        if parsed_reason is None:
            raise ValueError("termination_reason is required.")
        if (
            evidence.solved is not bool(solved)
            or evidence.termination_reason is not parsed_reason
        ):
            raise ValueError(
                "Terminal outcome evidence contradicts the example outcome."
            )
        return parsed_reason

    def _store_example(
        self,
        *,
        state: GridFMState,
        action_mask,
        scenario_id: int,
        step: int,
        selected_action_id: int,
        selected_branch_id: int | None,
        step_reward: float,
        final_return: float,
        discounted_return_from_step: float,
        solved: bool,
        termination_reason: TerminationReason,
        terminal_outcome_evidence: TerminalOutcomeEvidence,
        visit_counts: dict[int, int],
        mcts_policy: dict[int, float],
        iteration: int,
        episode_id: str,
        selection_temperature: float | None,
        selection_mode: str | None,
        policy_target_entropy: float | None,
        policy_target_normalized_entropy: float | None,
        mcts_legal_action_count: int | None,
        mcts_considered_action_count: int | None,
        mcts_visited_action_count: int | None,
        mcts_action_coverage: float | None,
        mcts_visited_action_coverage: float | None,
        extra_metadata: dict[str, Any] | None,
        outcome_fields: dict[str, object] | None,
    ) -> SelfPlayExample:
        state_id = f"{episode_id}_step_{step:03d}"
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

        evidence_json = terminal_outcome_evidence.to_json()
        metadata = dict(extra_metadata or {})
        provenance = physics_provenance(self.physics_config)
        metadata.update(
            {
                "run_id": self.run_id,
                "iteration": iteration,
                "episode_id": episode_id,
                "outcome_objective_version": OUTCOME_OBJECTIVE_VERSION,
                "outcome_value_target_contract_version": (
                    OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                ),
                "terminal_outcome_evidence_schema_version": (
                    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
                ),
                "terminal_outcome_evidence": (
                    terminal_outcome_evidence.to_dict()
                ),
            }
        )
        metadata.update(provenance)

        if outcome_fields is not None:
            for name in (
                "outcome_value_target",
                "outcome_class",
                "outcome_steps_to_terminal",
                "outcome_value_target_mode",
                "outcome_gamma",
            ):
                metadata[name] = outcome_fields[name]

        action_layout = build_branch_action_slots(state.branch_ids)
        action_provenance = topology_action_provenance(
            self.action_space_config,
            action_layout,
        )
        metadata.update(action_provenance)

        state_path = self.state_store.save_state(
            state=state,
            state_id=state_id,
            action_mask=action_mask,
            extra_metadata=metadata,
        )

        example = SelfPlayExample(
            state_id=state_id,
            state_path=str(state_path),
            run_id=self.run_id,
            iteration=iteration,
            episode_id=episode_id,
            scenario_id=int(scenario_id),
            step=int(step),
            selected_action_id=int(selected_action_id),
            selected_branch_id=(
                None
                if selected_branch_id is None
                else int(selected_branch_id)
            ),
            step_reward=float(step_reward),
            final_return=float(final_return),
            discounted_return_from_step=float(
                discounted_return_from_step
            ),
            solved=bool(solved),
            done=True,
            termination_reason=termination_reason_value(
                termination_reason
            ),
            terminal_outcome_evidence_schema_version=(
                TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
            ),
            terminal_outcome_evidence_json=evidence_json,
            physical_objective_schema_version=(
                PHYSICAL_OBJECTIVE_SCHEMA_VERSION
            ),
            outcome_objective_version=(
                OUTCOME_OBJECTIVE_VERSION
            ),
            outcome_value_target_contract_version=(
                OUTCOME_VALUE_TARGET_CONTRACT_VERSION
            ),
            state_feature_schema_version=int(
                provenance["state_feature_schema_version"]
            ),
            state_feature_schema_fingerprint=str(
                provenance["state_feature_schema_fingerprint"]
            ),
            bus_feature_columns=json.dumps(
                provenance["bus_feature_columns"],
                separators=(",", ":"),
                allow_nan=False,
            ),
            branch_feature_columns=json.dumps(
                provenance["branch_feature_columns"],
                separators=(",", ":"),
                allow_nan=False,
            ),
            edge_index_semantics=str(
                provenance["edge_index_semantics"]
            ),
            bus_id_semantics=str(provenance["bus_id_semantics"]),
            physics_config_contract_version=int(
                provenance["physics_config_contract_version"]
            ),
            topology_action_contract_version=int(
                action_provenance["topology_action_contract_version"]
            ),
            topology_action_config=json.dumps(
                action_provenance["topology_action_config"],
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
                action_provenance["action_layout"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            action_layout_fingerprint=str(
                action_provenance["action_layout_fingerprint"]
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
                {
                    str(key): int(value)
                    for key, value in visit_counts.items()
                }
            ),
            mcts_policy_json=json.dumps(
                {
                    str(action_id): float(probability)
                    for action_id, probability in normalized_policy.items()
                }
            ),
            outcome_value_target=self._optional_float(
                outcome_fields,
                "outcome_value_target",
            ),
            outcome_class=self._optional_str(
                outcome_fields,
                "outcome_class",
            ),
            outcome_steps_to_terminal=self._optional_int(
                outcome_fields,
                "outcome_steps_to_terminal",
            ),
            outcome_value_target_mode=self._optional_str(
                outcome_fields,
                "outcome_value_target_mode",
            ),
            outcome_gamma=self._optional_float(
                outcome_fields,
                "outcome_gamma",
            ),
            selection_temperature=(
                None
                if selection_temperature is None
                else float(selection_temperature)
            ),
            selection_mode=(
                None if selection_mode is None else str(selection_mode)
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
        return example

    @staticmethod
    def _optional_float(
        values: dict[str, object] | None,
        key: str,
    ) -> float | None:
        return None if values is None else float(values[key])

    @staticmethod
    def _optional_int(
        values: dict[str, object] | None,
        key: str,
    ) -> int | None:
        return None if values is None else int(values[key])

    @staticmethod
    def _optional_str(
        values: dict[str, object] | None,
        key: str,
    ) -> str | None:
        return None if values is None else str(values[key])

    def save(self) -> Path:
        columns = [field.name for field in fields(SelfPlayExample)]
        frame = pd.DataFrame(
            [asdict(example) for example in self.examples],
            columns=columns,
        )

        temporary_path = self.examples_path.with_name(
            f"{self.examples_path.name}.tmp"
        )
        try:
            frame.to_csv(temporary_path, index=False)
            temporary_path.replace(self.examples_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return self.examples_path

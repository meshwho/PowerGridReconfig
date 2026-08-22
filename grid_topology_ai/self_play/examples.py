from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.state import GridFMState
from grid_topology_ai.physics.objective import TerminalOutcomeEvidence
from grid_topology_ai.value_targets import TERMINAL_UTILITY_GAMMA
from grid_topology_ai.search.mcts import (
    normalize_policy,
    require_action_in_policy_support,
)
from grid_topology_ai.state import GridFMStateStore
from grid_topology_ai.termination import (
    TerminationReason,
    termination_reason_value,
    validate_outcome_invariants,
)
from grid_topology_ai.topology_actions import (
    ActionSpaceConfig,
    action_layout_fingerprint,
    action_layout_to_list,
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
    terminal_outcome_evidence_json: str
    physics_config: str
    topology_action_config: str
    action_layout: str
    action_layout_fingerprint: str
    visit_counts_json: str
    mcts_policy_json: str
    outcome_value_target: float | None = None
    outcome_class: str | None = None
    outcome_steps_to_terminal: int | None = None
    outcome_value_target_mode: str | None = None
    outcome_gamma: float | None = None
    selection_temperature: float | None = None
    selection_mode: str | None = None


class ExampleWriter:
    """Save self-play tensors plus the data needed to train them."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        physics_config: PhysicsConfig,
        action_space_config: ActionSpaceConfig,
        run_id: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.physics_config = physics_config
        self.states_dir = self.output_dir / "states"
        self.examples_path = self.output_dir / "examples.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.state_store = GridFMStateStore(self.states_dir)
        self.examples: list[SelfPlayExample] = []
        self._existing_frame = (
            pd.read_csv(self.examples_path)
            if self.examples_path.exists()
            else pd.DataFrame()
        )
        self.action_space_config = action_space_config
        self.run_id = str(run_id or self._existing_run_id())
        self._episode_ids: dict[int, str] = {}

    def _existing_run_id(self) -> str:
        if self._existing_frame.empty or "run_id" not in self._existing_frame:
            return uuid.uuid4().hex
        values = self._existing_frame["run_id"].dropna().astype(str).unique()
        if len(values) != 1:
            raise ValueError("Existing examples.csv has inconsistent run IDs.")
        return str(values[0])

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

        episode_id = f"iteration_{iteration:06d}_scenario_{scenario_id}"
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
                "terminal_outcome_evidence_json": evidence_json,
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
                "terminal_outcome_evidence must be TerminalOutcomeEvidence."
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
        physics_config = self.physics_config.to_dict()
        topology_action_config = self.action_space_config.to_contract_dict()
        action_layout = build_branch_action_slots(state.branch_ids)
        action_layout_data = action_layout_to_list(action_layout)
        layout_fingerprint = action_layout_fingerprint(action_layout)

        metadata = dict(extra_metadata or {})
        metadata.update(
            {
                "run_id": self.run_id,
                "iteration": iteration,
                "episode_id": episode_id,
                "physics_config": physics_config,
                "topology_action_config": topology_action_config,
                "action_layout": action_layout_data,
                "action_layout_fingerprint": layout_fingerprint,
                "terminal_outcome_evidence": (
                    terminal_outcome_evidence.to_dict()
                ),
            }
        )

        if outcome_fields is not None:
            for name in (
                "outcome_value_target",
                "outcome_class",
                "outcome_steps_to_terminal",
                "outcome_value_target_mode",
                "outcome_gamma",
            ):
                metadata[name] = outcome_fields[name]

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
            terminal_outcome_evidence_json=evidence_json,
            physics_config=json.dumps(
                physics_config,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            topology_action_config=json.dumps(
                topology_action_config,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            action_layout=json.dumps(
                action_layout_data,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            action_layout_fingerprint=layout_fingerprint,
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
        if not self._existing_frame.empty:
            frame = pd.concat(
                [self._existing_frame, frame], ignore_index=True
            )[columns]

        temporary_path = self.examples_path.with_name(
            f"{self.examples_path.name}.tmp"
        )
        try:
            frame.to_csv(temporary_path, index=False)
            temporary_path.replace(self.examples_path)
            self._existing_frame = frame
            self.examples.clear()
        finally:
            temporary_path.unlink(missing_ok=True)

        return self.examples_path

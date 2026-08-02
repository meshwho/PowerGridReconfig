from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.outcome_contract import (
    TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.self_play.examples import ExampleWriter, SelfPlayExample
from tests.outcome_evidence_helpers import terminal_evidence
from tests.topology_contract_helpers import (
    TEST_ACTION_SPACE_CONFIG,
    topology_csv_fields,
)


def test_example_writer_class_name_is_explicit() -> None:
    assert ExampleWriter.__name__ == "ExampleWriter"


def test_example_writer_uses_expected_artifact_names(tmp_path: Path) -> None:
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    )

    assert writer.states_dir == tmp_path / "states"
    assert writer.examples_path == tmp_path / "examples.csv"


def test_example_writers_use_distinct_run_ids(tmp_path: Path) -> None:
    first = ExampleWriter(
        tmp_path / "first",
        physics_config=DEFAULT_PHYSICS_CONFIG,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    )
    second = ExampleWriter(
        tmp_path / "second",
        physics_config=DEFAULT_PHYSICS_CONFIG,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    )

    assert first.run_id != second.run_id


def test_example_writer_rejects_off_policy_selected_action(
    tmp_path: Path,
) -> None:
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    )
    writer.state_store = SimpleNamespace(
        save_state=lambda **kwargs: pytest.fail(
            "state must not be written for an invalid example"
        )
    )

    with pytest.raises(ValueError, match="outside the support"):
        writer.add_example(
            state=object(),  # type: ignore[arg-type]
            state_id="state-1",
            action_mask=[True, True],
            scenario_id=1,
            step=0,
            selected_action_id=0,
            selected_branch_id=None,
            step_reward=0.0,
            final_return=0.0,
            discounted_return_from_step=0.0,
            solved=True,
            done=True,
            termination_reason="solved",
            terminal_outcome_evidence=terminal_evidence("solved"),
            visit_counts={1: 3},
            mcts_policy={1: 1.0},
        )


def test_example_writer_rejects_mismatched_terminal_evidence(
    tmp_path: Path,
) -> None:
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    )
    writer.state_store = SimpleNamespace(
        save_state=lambda **kwargs: pytest.fail(
            "state must not be written for invalid evidence"
        )
    )

    with pytest.raises(ValueError, match="contradicts"):
        writer.add_example(
            state=SimpleNamespace(branch_ids=[3, 4]),
            state_id="state-1",
            action_mask=[True, True, True],
            scenario_id=1,
            step=0,
            selected_action_id=0,
            selected_branch_id=None,
            step_reward=0.0,
            final_return=0.0,
            discounted_return_from_step=0.0,
            solved=True,
            done=True,
            termination_reason="solved",
            terminal_outcome_evidence=terminal_evidence(
                "max_steps_reached"
            ),
            visit_counts={0: 3},
            mcts_policy={0: 1.0},
        )


def test_example_writer_save_preserves_csv_schema(tmp_path: Path) -> None:
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    )
    provenance = physics_provenance(DEFAULT_PHYSICS_CONFIG)
    action_provenance = topology_csv_fields((3, 4))
    evidence = terminal_evidence("solved")
    writer.examples.append(
        SelfPlayExample(
            state_id="state-1",
            state_path="states/state-1.npz",
            run_id="run-1",
            iteration=1,
            episode_id="episode-1",
            scenario_id=1,
            step=0,
            selected_action_id=2,
            selected_branch_id=4,
            step_reward=1.0,
            final_return=1.0,
            discounted_return_from_step=1.0,
            solved=True,
            done=True,
            termination_reason="solved",
            terminal_outcome_evidence_schema_version=(
                TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
            ),
            terminal_outcome_evidence_json=evidence.to_json(),
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
            ),
            branch_feature_columns=json.dumps(
                provenance["branch_feature_columns"],
                separators=(",", ":"),
            ),
            edge_index_semantics=str(
                provenance["edge_index_semantics"]
            ),
            bus_id_semantics=str(provenance["bus_id_semantics"]),
            physics_config_contract_version=int(
                provenance["physics_config_contract_version"]
            ),
            topology_action_contract_version=int(
                action_provenance[
                    "topology_action_contract_version"
                ]
            ),
            topology_action_config=str(
                action_provenance["topology_action_config"]
            ),
            topology_action_config_fingerprint=str(
                action_provenance[
                    "topology_action_config_fingerprint"
                ]
            ),
            action_layout=str(
                action_provenance["action_layout"]
            ),
            action_layout_fingerprint=str(
                action_provenance["action_layout_fingerprint"]
            ),
            physics_config=json.dumps(
                provenance["physics_config"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            physics_config_fingerprint=str(
                provenance["physics_config_fingerprint"]
            ),
            visit_counts_json='{"2": 4}',
            mcts_policy_json='{"2": 1.0}',
        )
    )

    path = writer.save()

    assert list(pd.read_csv(path).columns) == [
        "state_id",
        "state_path",
        "run_id",
        "iteration",
        "episode_id",
        "scenario_id",
        "step",
        "selected_action_id",
        "selected_branch_id",
        "step_reward",
        "final_return",
        "discounted_return_from_step",
        "solved",
        "done",
        "termination_reason",
        "terminal_outcome_evidence_schema_version",
        "terminal_outcome_evidence_json",
        "physical_objective_schema_version",
        "outcome_objective_version",
        "outcome_value_target_contract_version",
        "state_feature_schema_version",
        "state_feature_schema_fingerprint",
        "bus_feature_columns",
        "branch_feature_columns",
        "edge_index_semantics",
        "bus_id_semantics",
        "physics_config_contract_version",
        "topology_action_contract_version",
        "topology_action_config",
        "topology_action_config_fingerprint",
        "action_layout",
        "action_layout_fingerprint",
        "physics_config",
        "physics_config_fingerprint",
        "visit_counts_json",
        "mcts_policy_json",
        "selection_temperature",
        "selection_mode",
        "policy_target_entropy",
        "policy_target_normalized_entropy",
        "mcts_legal_action_count",
        "mcts_considered_action_count",
        "mcts_visited_action_count",
        "mcts_action_coverage",
        "mcts_visited_action_coverage",
    ]

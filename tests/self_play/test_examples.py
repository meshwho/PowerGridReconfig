from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.self_play.examples import ExampleWriter, SelfPlayExample


def test_example_writer_class_name_is_explicit() -> None:
    assert ExampleWriter.__name__ == "ExampleWriter"


def test_example_writer_uses_expected_artifact_names(tmp_path: Path) -> None:
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )

    assert writer.states_dir == tmp_path / "states"
    assert writer.examples_path == tmp_path / "examples.csv"


def test_example_writer_rejects_off_policy_selected_action(
    tmp_path: Path,
) -> None:
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
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
            visit_counts={1: 3},
            mcts_policy={1: 1.0},
        )


def test_example_writer_save_preserves_csv_schema(tmp_path: Path) -> None:
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    provenance = physics_provenance(DEFAULT_PHYSICS_CONFIG)
    writer.examples.append(
        SelfPlayExample(
            state_id="state-1",
            state_path="states/state-1.npz",
            scenario_id=1,
            step=0,
            selected_action_id=2,
            selected_branch_id=3,
            step_reward=1.0,
            final_return=1.0,
            discounted_return_from_step=1.0,
            solved=True,
            done=True,
            termination_reason="solved",
            physical_objective_schema_version=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
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
        "physical_objective_schema_version",
        "outcome_value_target_contract_version",
        "physics_config_contract_version",
        "physics_config",
        "physics_config_fingerprint",
        "visit_counts_json",
        "mcts_policy_json",
    ]

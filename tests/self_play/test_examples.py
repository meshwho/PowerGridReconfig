from __future__ import annotations

from pathlib import Path

import pandas as pd

from grid_topology_ai.self_play.examples import ExampleWriter, SelfPlayExample


def test_example_writer_class_name_is_explicit() -> None:
    assert ExampleWriter.__name__ == "ExampleWriter"


def test_example_writer_uses_expected_artifact_names(tmp_path: Path) -> None:
    writer = ExampleWriter(tmp_path)

    assert writer.states_dir == tmp_path / "states"
    assert writer.examples_path == tmp_path / "examples.csv"


def test_example_writer_save_preserves_csv_schema(tmp_path: Path) -> None:
    writer = ExampleWriter(tmp_path)
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
        "visit_counts_json",
        "mcts_policy_json",
        "value_target_schema_version",
        "physical_objective_schema_version",
    ]

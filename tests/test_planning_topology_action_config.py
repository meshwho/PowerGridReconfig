from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from grid_topology_ai.topology_actions import GridFMAction
from grid_topology_ai.transition_generator import GridFMTransitionGenerator
from scripts.topology_action_cli import (
    action_space_kwargs,
    add_topology_action_arguments,
    topology_action_config_from_args,
)


def _parse_topology_args(values: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_topology_action_arguments(parser)
    return parser.parse_args(values)


def test_cli_topology_action_config_is_canonical() -> None:
    args = _parse_topology_args(
        [
            "--closeable-branch-id",
            "20",
            "--closeable-branch-id",
            "10",
            "--min-loading-for-switch-percent",
            "25",
            "--no-require-connected-after-switch",
        ]
    )

    config = topology_action_config_from_args(args)

    assert config.require_connected_after_switch is False
    assert config.min_loading_for_switch_percent == pytest.approx(25.0)
    assert config.closeable_branch_ids == (10, 20)
    assert action_space_kwargs(
        config,
        enable_cache=False,
    ) == {
        "require_connected_after_switch": False,
        "min_loading_for_switch_percent": 25.0,
        "closeable_branch_ids": (10, 20),
        "enable_cache": False,
    }


@pytest.mark.parametrize(
    "path",
    [
        "scripts/planning/run_mcts.py",
        "scripts/planning/run_mcts_episode.py",
        "scripts/planning/run_beam_search.py",
        "scripts/planning/run_impact_beam_search.py",
        "scripts/analyze_mcts_root_branches.py",
        "scripts/generate_transitions.py",
        "scripts/check_action_space.py",
    ],
)
def test_nonlegacy_cli_uses_resolved_topology_config(
    path: str,
) -> None:
    source = Path(path).read_text(encoding="utf-8")
    compact = "".join(source.split())

    assert "action_space_kwargs(" in source
    assert (
        "GridFMActionSpace("
        "require_connected_after_switch=True"
        not in compact
    )


@pytest.mark.parametrize(
    "path",
    [
        "scripts/planning/run_mcts.py",
        "scripts/planning/run_mcts_episode.py",
        "scripts/planning/run_beam_search.py",
        "scripts/planning/run_impact_beam_search.py",
        "scripts/generate_transitions.py",
        "scripts/check_action_space.py",
    ],
)
def test_configurable_cli_exposes_topology_arguments(
    path: str,
) -> None:
    source = Path(path).read_text(encoding="utf-8")
    assert "add_topology_action_arguments(" in source


def test_checkpoint_driven_clis_use_checkpoint_action_config() -> None:
    for path in (
        "scripts/planning/run_mcts_episode.py",
        "scripts/analyze_mcts_root_branches.py",
    ):
        source = Path(path).read_text(encoding="utf-8")
        assert "evaluator.topology_action_config" in source


def test_transition_top_k_keeps_all_legal_closures() -> None:
    open_action = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
    )
    close_action = GridFMAction(
        action_id=2,
        action_type="switch_on_branch",
        branch_id=20,
        branch_pos=1,
    )
    second_close_action = GridFMAction(
        action_id=3,
        action_type="switch_on_branch",
        branch_id=30,
        branch_pos=2,
    )

    class FakeActionSpace:
        def valid_actions(self, state: object) -> list[GridFMAction]:
            return [
                open_action,
                close_action,
                second_close_action,
            ]

        def loading_priority(
            self,
            state: object,
            action: GridFMAction,
        ) -> float | None:
            if action.action_type == "switch_on_branch":
                return None
            return 80.0

    generator = GridFMTransitionGenerator(
        adapter=object(),
        backend=object(),
        action_space=FakeActionSpace(),
        reward_fn=object(),
    )

    selected = generator._select_actions(
        state=SimpleNamespace(),
        max_switch_actions=1,
        include_do_nothing=False,
    )

    assert selected == [
        open_action,
        close_action,
        second_close_action,
    ]
    assert [action.target_status for action in selected] == [
        0,
        1,
        1,
    ]

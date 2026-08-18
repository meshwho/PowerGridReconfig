from __future__ import annotations

import argparse
from pathlib import Path

import pytest

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
    assert action_space_kwargs(config, enable_cache=False) == {
        "require_connected_after_switch": False,
        "min_loading_for_switch_percent": 25.0,
        "closeable_branch_ids": (10, 20),
        "enable_cache": False,
    }


@pytest.mark.parametrize(
    "path",
    [
        "scripts/analyze_mcts_root_branches.py",
        "scripts/check_action_space.py",
    ],
)
def test_remaining_topology_clis_use_resolved_config(path: str) -> None:
    source = Path(path).read_text(encoding="utf-8")
    compact = "".join(source.split())

    assert "action_space_kwargs(" in source
    assert "GridFMActionSpace(require_connected_after_switch=True" not in compact


def test_configurable_check_cli_exposes_topology_arguments() -> None:
    source = Path("scripts/check_action_space.py").read_text(encoding="utf-8")
    assert "add_topology_action_arguments(" in source


def test_checkpoint_diagnostic_uses_checkpoint_action_config() -> None:
    source = Path("scripts/analyze_mcts_root_branches.py").read_text(encoding="utf-8")
    assert "evaluator.topology_action_config" in source

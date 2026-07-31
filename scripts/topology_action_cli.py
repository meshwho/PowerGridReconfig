from __future__ import annotations

import argparse

from grid_topology_ai.topology_actions import ActionSpaceConfig


def add_topology_action_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--closeable-branch-id",
        dest="closeable_branch_ids",
        action="append",
        type=int,
        default=None,
        help=(
            "Branch ID that may be closed when currently inactive. "
            "Repeat for multiple normally-open or tie branches."
        ),
    )
    parser.add_argument(
        "--min-loading-for-switch-percent",
        type=float,
        default=None,
        help=(
            "Minimum loading for branch-opening candidates. "
            "This threshold never filters permitted closures."
        ),
    )
    connectivity = parser.add_mutually_exclusive_group()
    connectivity.add_argument(
        "--require-connected-after-switch",
        dest="require_connected_after_switch",
        action="store_true",
        help="Mask branch openings that disconnect the active grid.",
    )
    connectivity.add_argument(
        "--no-require-connected-after-switch",
        dest="require_connected_after_switch",
        action="store_false",
        help="Allow openings that disconnect the active grid.",
    )
    parser.set_defaults(require_connected_after_switch=None)


def topology_action_overrides_present(
    args: argparse.Namespace,
) -> bool:
    return any(
        (
            args.closeable_branch_ids is not None,
            args.min_loading_for_switch_percent is not None,
            args.require_connected_after_switch is not None,
        )
    )


def topology_action_config_from_args(
    args: argparse.Namespace,
    *,
    base: ActionSpaceConfig | None = None,
) -> ActionSpaceConfig:
    base_config = ActionSpaceConfig() if base is None else base
    return ActionSpaceConfig(
        require_connected_after_switch=(
            base_config.require_connected_after_switch
            if args.require_connected_after_switch is None
            else bool(args.require_connected_after_switch)
        ),
        min_loading_for_switch_percent=(
            base_config.min_loading_for_switch_percent
            if args.min_loading_for_switch_percent is None
            else float(args.min_loading_for_switch_percent)
        ),
        closeable_branch_ids=(
            base_config.closeable_branch_ids
            if args.closeable_branch_ids is None
            else tuple(args.closeable_branch_ids)
        ),
    )


def action_space_kwargs(
    config: ActionSpaceConfig,
    *,
    enable_cache: bool,
) -> dict[str, object]:
    return {
        "require_connected_after_switch": config.require_connected_after_switch,
        "min_loading_for_switch_percent": config.min_loading_for_switch_percent,
        "closeable_branch_ids": config.closeable_branch_ids,
        "enable_cache": bool(enable_cache),
    }


def print_topology_action_config(config: ActionSpaceConfig) -> None:
    print(
        "Require connected after switch: "
        f"{config.require_connected_after_switch}"
    )
    print(
        "Minimum opening loading:         "
        f"{config.min_loading_for_switch_percent}"
    )
    print(f"Closeable branch IDs:            {config.closeable_branch_ids}")

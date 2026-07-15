from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.paths import (
    SelfPlayPaths,
    discover_project_root,
)
from grid_topology_ai.self_play.pipeline import (
    PipelineRequest,
    run_self_play_pipeline,
)
from grid_topology_ai.self_play.plan import render_execution_plan
from grid_topology_ai.self_play.preflight import validate_inputs


def print_header(title: str) -> None:
    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return data


def run_loop(
    *,
    config_path: str | Path,
    validate_only: bool = False,
    plan_only: bool = False,
    resume: bool = False,
) -> None:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    project_root = discover_project_root(config_path)

    config = SelfPlayConfig.from_mapping(cfg)
    paths = SelfPlayPaths.from_config(
        config=config,
        project_root=project_root,
    )

    if plan_only:
        rendered_plan = render_execution_plan(
            config=config,
            paths=paths,
            config_path=config_path,
        )
        print(rendered_plan)
        return

    warnings = validate_inputs(
        paths,
        require_bootstrap=not validate_only,
    )

    for warning in warnings:
        print(f"WARNING: {warning}")

    if validate_only:
        print_header("Self-play config validation")
        print("Config is valid.")
        print(f"Project root: {project_root}")
        print(f"Config:       {config_path}")
        return

    run_self_play_pipeline(
        PipelineRequest(
            config=config,
            raw_config=cfg,
            paths=paths,
            resume=resume,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hybrid pool-guided self-play loop."
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to self_play_loop.yaml.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config and required paths, do not run self-play.",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Print the resolved self-play execution plan without running "
            "generation, training, or evaluation."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue after the last completed iteration. "
            "Refuse to continue if incomplete iteration "
            "directories are present."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_loop(
        config_path=args.config,
        validate_only=bool(args.validate_only),
        plan_only=bool(args.plan_only),
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import cProfile
import pstats
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile scripts.run_mcts_episode with cProfile."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/profiles/mcts_episode.prof",
        help="Output .prof file.",
    )

    parser.add_argument(
        "--sort",
        type=str,
        default="cumtime",
        choices=["cumtime", "tottime", "calls"],
        help="Sort order for printed profile statistics.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Number of functions to print.",
    )

    parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to scripts.run_mcts_episode. Use -- before them.",
    )

    args = parser.parse_args()

    target_args = list(args.target_args)

    if target_args and target_args[0] == "--":
        target_args = target_args[1:]

    if not target_args:
        raise ValueError(
            "No target arguments provided. Use: "
            "python -m scripts.profile_mcts_episode -- <run_mcts_episode args>"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Profiling MCTS episode")
    print("=" * 100)
    print(f"Output profile: {output_path}")
    print(f"Sort order:     {args.sort}")
    print(f"Print limit:    {args.limit}")
    print()
    print("Target command:")
    print("python -m scripts.run_mcts_episode " + " ".join(target_args))
    print()

    old_argv = sys.argv[:]

    try:
        sys.argv = ["scripts.planning.run_mcts_episode", *target_args]

        profiler = cProfile.Profile()
        profiler.enable()

        runpy.run_module(
            "scripts.planning.run_mcts_episode",
            run_name="__main__",
        )

        profiler.disable()
        profiler.dump_stats(output_path)

    finally:
        sys.argv = old_argv

    print("\n" + "=" * 100)
    print("Top profile results")
    print("=" * 100)

    stats = pstats.Stats(str(output_path))
    stats.strip_dirs()
    stats.sort_stats(args.sort)
    stats.print_stats(args.limit)

    print("\nSaved profile:")
    print(output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter
from grid_topology_ai.pypower_backend import (
    GridFMPowerFlowBackend,
    pf_algorithm_name,
)


def select_candidate_actions(
    state,
    action_space: GridFMActionSpace,
    top_k: int,
):
    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    valid_actions = action_space.valid_actions(state)

    switch_actions = [
        action
        for action in valid_actions
        if action.action_type == "switch_off_branch"
    ]

    switch_actions = sorted(
        switch_actions,
        key=lambda action: float(
            state.branch_features[action.branch_pos, loading_idx]
        ),
        reverse=True,
    )

    return switch_actions[:top_k]


def benchmark_algorithm(
    raw_dir: Path,
    scenario_id: int,
    pf_alg: int,
    top_k: int,
    repeats: int,
) -> dict:
    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=pf_alg,
    )
    action_space = GridFMActionSpace(require_connected_after_switch=True)

    state = adapter.build_state(scenario_id)
    actions = select_candidate_actions(
        state=state,
        action_space=action_space,
        top_k=top_k,
    )

    records = []

    for repeat in range(repeats):
        for action in actions:
            start = time.perf_counter()

            result = backend.run_power_flow_from_state(
                state=state,
                switched_off_branch_id=action.branch_id,
            )

            elapsed = time.perf_counter() - start

            if result.next_state is not None:
                metrics = result.next_state.metrics
                max_loading = float(metrics["max_loading_percent"])
                overloaded = int(metrics["num_overloaded_branches"])
                hard = int(metrics["num_hard_overloaded_branches"])
            else:
                max_loading = float("nan")
                overloaded = -1
                hard = -1

            records.append(
                {
                    "repeat": repeat,
                    "action_id": int(action.action_id),
                    "branch_id": int(action.branch_id),
                    "success": bool(result.success),
                    "elapsed_sec": float(elapsed),
                    "max_loading_percent": max_loading,
                    "num_overloaded_branches": overloaded,
                    "num_hard_overloaded_branches": hard,
                    "message": result.message,
                }
            )

    df = pd.DataFrame(records)

    success_rate = float(df["success"].mean()) if len(df) else 0.0

    return {
        "pf_alg": pf_alg,
        "pf_alg_name": pf_algorithm_name(pf_alg),
        "scenario_id": int(scenario_id),
        "top_k": int(top_k),
        "repeats": int(repeats),
        "num_runs": int(len(df)),
        "success_rate": success_rate,
        "total_time_sec": float(df["elapsed_sec"].sum()),
        "avg_time_sec": float(df["elapsed_sec"].mean()),
        "median_time_sec": float(df["elapsed_sec"].median()),
        "min_time_sec": float(df["elapsed_sec"].min()),
        "max_time_sec": float(df["elapsed_sec"].max()),
        "avg_max_loading_percent": float(df["max_loading_percent"].mean()),
        "avg_overloaded": float(df["num_overloaded_branches"].mean()),
        "avg_hard_overloaded": float(df["num_hard_overloaded_branches"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PYPOWER power flow algorithms."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument(
        "--scenario",
        type=int,
        default=7,
        help="Scenario ID.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of candidate switch actions to test.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats for each candidate action.",
    )

    parser.add_argument(
        "--algorithms",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3, 4],
        help="PF algorithms: 1=NR, 2=FDXB, 3=FDBX, 4=GS.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Benchmarking PYPOWER power flow algorithms")
    print("=" * 100)

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Scenario:      {args.scenario}")
    print(f"Top-K actions: {args.top_k}")
    print(f"Repeats:       {args.repeats}")
    print(f"Algorithms:    {args.algorithms}")

    rows = []

    for pf_alg in args.algorithms:
        print("\n" + "-" * 100)
        print(f"Testing PF_ALG={pf_alg} ({pf_algorithm_name(pf_alg)})")
        print("-" * 100)

        result = benchmark_algorithm(
            raw_dir=raw_dir,
            scenario_id=args.scenario,
            pf_alg=pf_alg,
            top_k=args.top_k,
            repeats=args.repeats,
        )

        rows.append(result)

        print(f"Success rate:     {result['success_rate']:.3f}")
        print(f"Total time:       {result['total_time_sec']:.4f} sec")
        print(f"Average time:     {result['avg_time_sec']:.6f} sec/call")
        print(f"Median time:      {result['median_time_sec']:.6f} sec/call")
        print(f"Avg max loading:  {result['avg_max_loading_percent']:.4f}%")
        print(f"Avg overloaded:   {result['avg_overloaded']:.4f}")
        print(f"Avg hard:         {result['avg_hard_overloaded']:.4f}")

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    print(
        summary[
            [
                "pf_alg",
                "pf_alg_name",
                "success_rate",
                "total_time_sec",
                "avg_time_sec",
                "median_time_sec",
                "avg_max_loading_percent",
                "avg_overloaded",
                "avg_hard_overloaded",
            ]
        ].to_string(index=False)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
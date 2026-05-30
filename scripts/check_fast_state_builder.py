from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pypower.api import ppoption, runpf

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMAdapter,
)
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def print_feature_diffs(
    name: str,
    columns: list[str],
    old: np.ndarray,
    new: np.ndarray,
    top_k: int = 20,
) -> None:
    print("\n" + "=" * 100)
    print(f"{name} feature differences")
    print("=" * 100)

    abs_diff = np.abs(old - new)

    print(f"Shape old: {old.shape}")
    print(f"Shape new: {new.shape}")
    print(f"Max abs diff: {max_abs_diff(old, new):.10f}")

    per_col = abs_diff.max(axis=0)

    rows = sorted(
        [
            (columns[i], float(per_col[i]))
            for i in range(len(columns))
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nPer-column max abs diff:")
    for col, diff in rows:
        print(f"  {col:<20} {diff:.10f}")

    flat_indices = np.argwhere(abs_diff > 1e-6)

    if len(flat_indices) == 0:
        print("\nNo element differences above 1e-6.")
        return

    print(f"\nTop element differences above 1e-6, showing first {top_k}:")
    shown = 0

    sorted_indices = sorted(
        flat_indices,
        key=lambda ij: float(abs_diff[ij[0], ij[1]]),
        reverse=True,
    )

    for row_idx, col_idx in sorted_indices[:top_k]:
        print(
            f"  row={int(row_idx):>4} | "
            f"feature={columns[int(col_idx)]:<20} | "
            f"old={float(old[row_idx, col_idx]):>14.6f} | "
            f"new={float(new[row_idx, col_idx]):>14.6f} | "
            f"diff={float(abs_diff[row_idx, col_idx]):.10f}"
        )
        shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare old pandas state builder and fast numpy state builder."
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
    )

    parser.add_argument(
        "--branch",
        type=int,
        default=105,
        help="Branch ID to switch off from the initial state.",
    )

    parser.add_argument(
        "--pf-alg",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to compare neural evaluator outputs.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("Checking fast state builder equivalence")
    print("=" * 100)

    raw_dir = Path(args.raw_dir)

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=args.pf_alg,
        enable_cache=False,
        store_raw_result=True,
    )

    state = adapter.build_state(args.scenario)

    print(f"Scenario: {args.scenario}")
    print(f"Branch:   {args.branch}")
    print(f"PF alg:   {args.pf_alg}")

    print("\nInitial metrics:")
    for key, value in state.metrics.items():
        print(f"  {key}: {value}")
    print(f"Initial outaged: {state.outaged_branch_ids}")

    # Build ppc once, run PF once, then build both states from exactly same result.
    ppc, frames = backend._build_ppc_from_state(
        state=state,
        switched_off_branch_id=args.branch,
    )

    ppopt = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=args.pf_alg,
        PF_MAX_IT=backend.max_iter,
    )

    result_ppc, success = runpf(ppc, ppopt)
    success = bool(success)

    print(f"\nPower flow success: {success}")

    if not success:
        print("Power flow did not converge. Cannot compare builders.")
        return

    old_state = backend._build_state_from_pypower_result(
        scenario_id=int(state.scenario_id),
        result_ppc=result_ppc,
        original_frames=frames,
    )

    new_state = backend._build_state_from_pypower_result_fast(
        scenario_id=int(state.scenario_id),
        result_ppc=result_ppc,
        previous_state=state,
        original_frames=frames,
    )

    print_feature_diffs(
        name="Bus",
        columns=BUS_FEATURE_COLUMNS,
        old=old_state.bus_features,
        new=new_state.bus_features,
    )

    print_feature_diffs(
        name="Branch",
        columns=BRANCH_FEATURE_COLUMNS,
        old=old_state.branch_features,
        new=new_state.branch_features,
    )

    print("\n" + "=" * 100)
    print("Metrics comparison")
    print("=" * 100)

    all_metric_keys = sorted(set(old_state.metrics) | set(new_state.metrics))

    for key in all_metric_keys:
        old_value = old_state.metrics.get(key)
        new_value = new_state.metrics.get(key)

        if isinstance(old_value, float) or isinstance(new_value, float):
            diff = abs(float(old_value) - float(new_value))
            print(
                f"{key:<35} old={float(old_value):>14.8f} "
                f"new={float(new_value):>14.8f} diff={diff:.10f}"
            )
        else:
            print(f"{key:<35} old={old_value} new={new_value}")

    print("\nOutaged comparison:")
    print(f"  old: {old_state.outaged_branch_ids}")
    print(f"  new: {new_state.outaged_branch_ids}")
    print(f"  equal: {old_state.outaged_branch_ids == new_state.outaged_branch_ids}")

    print("\nArray identity checks:")
    print(f"  edge_index equal:     {np.array_equal(old_state.edge_index, new_state.edge_index)}")
    print(f"  branch_ids equal:    {np.array_equal(old_state.branch_ids, new_state.branch_ids)}")
    print(f"  branch_status equal: {np.array_equal(old_state.branch_status, new_state.branch_status)}")

    if args.checkpoint is not None:
        print("\n" + "=" * 100)
        print("Neural evaluator comparison")
        print("=" * 100)

        action_space = GridFMActionSpace(require_connected_after_switch=True)
        evaluator = NeuralPolicyValueEvaluator(args.checkpoint)

        old_mask = action_space.valid_action_mask(old_state)
        new_mask = action_space.valid_action_mask(new_state)

        print(f"Action mask equal: {np.array_equal(old_mask, new_mask)}")
        print(f"Action mask diff count: {int(np.sum(old_mask != new_mask))}")

        old_policy, old_value = evaluator.evaluate(old_state, old_mask)
        new_policy, new_value = evaluator.evaluate(new_state, new_mask)

        print(f"Value old: {old_value:+.8f}")
        print(f"Value new: {new_value:+.8f}")
        print(f"Value diff: {abs(old_value - new_value):.10f}")
        print(f"Policy max abs diff: {max_abs_diff(old_policy, new_policy):.10f}")

        old_top = old_policy.argsort()[::-1][:10]
        new_top = new_policy.argsort()[::-1][:10]

        actions = action_space.build_all_actions(old_state)

        print("\nOld top policy actions:")
        for action_id in old_top:
            action = actions[int(action_id)]
            print(
                f"  action={int(action_id):>3} | "
                f"branch={str(action.branch_id):>4} | "
                f"p={float(old_policy[action_id]):.6f}"
            )

        print("\nNew top policy actions:")
        for action_id in new_top:
            action = actions[int(action_id)]
            print(
                f"  action={int(action_id):>3} | "
                f"branch={str(action.branch_id):>4} | "
                f"p={float(new_policy[action_id]):.6f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
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


def print_top_diffs(
    label: str,
    columns: list[str],
    old: np.ndarray,
    new: np.ndarray,
    threshold: float = 1e-6,
    top_k: int = 10,
) -> None:
    diff = np.abs(old - new)

    print(f"\n{label}:")
    print(f"  max_abs_diff = {max_abs_diff(old, new):.10f}")

    per_col = diff.max(axis=0)

    rows = sorted(
        [(columns[i], float(per_col[i])) for i in range(len(columns))],
        key=lambda x: x[1],
        reverse=True,
    )

    print("  per-column max diff:")
    for col, value in rows:
        if value > threshold:
            print(f"    {col:<20} {value:.10f}")

    indices = np.argwhere(diff > threshold)

    if len(indices) == 0:
        print(f"  no element differences above {threshold}")
        return

    sorted_indices = sorted(
        indices,
        key=lambda ij: float(diff[ij[0], ij[1]]),
        reverse=True,
    )

    print(f"  top element differences above {threshold}:")
    for row_idx, col_idx in sorted_indices[:top_k]:
        print(
            f"    row={int(row_idx):>4} | "
            f"feature={columns[int(col_idx)]:<20} | "
            f"old={float(old[row_idx, col_idx]):>14.6f} | "
            f"new={float(new[row_idx, col_idx]):>14.6f} | "
            f"diff={float(diff[row_idx, col_idx]):.10f}"
        )


def run_one_step_old_and_fast(
    backend: GridFMPowerFlowBackend,
    previous_old_state,
    previous_fast_state,
    branch_id: int,
    pf_alg: int,
):
    """
    Run one identical topology action from old_state and fast_state.

    This checks whether small differences accumulate across a sequence.
    """

    ppopt = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=pf_alg,
        PF_MAX_IT=backend.max_iter,
    )

    old_ppc, old_frames = backend._build_ppc_from_state(
        state=previous_old_state,
        switched_off_branch_id=branch_id,
    )

    old_result_ppc, old_success = runpf(old_ppc, ppopt)
    old_success = bool(old_success)

    if not old_success:
        raise RuntimeError(f"Old path power flow failed at branch {branch_id}")

    next_old_state = backend._build_state_from_pypower_result(
        scenario_id=int(previous_old_state.scenario_id),
        result_ppc=old_result_ppc,
        original_frames=old_frames,
    )

    fast_ppc, fast_frames = backend._build_ppc_from_state(
        state=previous_fast_state,
        switched_off_branch_id=branch_id,
    )

    fast_result_ppc, fast_success = runpf(fast_ppc, ppopt)
    fast_success = bool(fast_success)

    if not fast_success:
        raise RuntimeError(f"Fast path power flow failed at branch {branch_id}")

    next_fast_state = backend._build_state_from_pypower_result_fast(
        scenario_id=int(previous_fast_state.scenario_id),
        result_ppc=fast_result_ppc,
        previous_state=previous_fast_state,
        original_frames=fast_frames,
    )

    return next_old_state, next_fast_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare old and fast state builders over a branch sequence."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument("--scenario", type=int, default=7)

    parser.add_argument(
        "--branches",
        type=int,
        nargs="+",
        required=True,
        help="Branch IDs to switch off sequentially.",
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
    print("Checking fast state builder over sequence")
    print("=" * 100)

    raw_dir = Path(args.raw_dir)

    adapter = GridFMAdapter(raw_dir)

    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=args.pf_alg,
        enable_cache=False,
        store_raw_result=True,
    )

    old_state = adapter.build_state(args.scenario)
    fast_state = adapter.build_state(args.scenario)

    evaluator = None
    action_space = None

    if args.checkpoint is not None:
        evaluator = NeuralPolicyValueEvaluator(args.checkpoint)
        action_space = GridFMActionSpace(require_connected_after_switch=True)

    print(f"Scenario: {args.scenario}")
    print(f"Branches: {args.branches}")
    print(f"PF alg:   {args.pf_alg}")

    for step_idx, branch_id in enumerate(args.branches, start=1):
        print("\n" + "=" * 100)
        print(f"Step {step_idx}: switch off branch {branch_id}")
        print("=" * 100)

        old_state, fast_state = run_one_step_old_and_fast(
            backend=backend,
            previous_old_state=old_state,
            previous_fast_state=fast_state,
            branch_id=int(branch_id),
            pf_alg=args.pf_alg,
        )

        print(f"Old outaged:  {old_state.outaged_branch_ids}")
        print(f"Fast outaged: {fast_state.outaged_branch_ids}")
        print(f"Outaged equal: {old_state.outaged_branch_ids == fast_state.outaged_branch_ids}")

        print_top_diffs(
            label="Bus features",
            columns=BUS_FEATURE_COLUMNS,
            old=old_state.bus_features,
            new=fast_state.bus_features,
        )

        print_top_diffs(
            label="Branch features",
            columns=BRANCH_FEATURE_COLUMNS,
            old=old_state.branch_features,
            new=fast_state.branch_features,
        )

        print("\nMetrics:")
        for key in sorted(set(old_state.metrics) | set(fast_state.metrics)):
            old_value = old_state.metrics.get(key)
            fast_value = fast_state.metrics.get(key)

            if isinstance(old_value, float) or isinstance(fast_value, float):
                diff = abs(float(old_value) - float(fast_value))
                print(
                    f"  {key:<35} "
                    f"old={float(old_value):>14.8f} "
                    f"fast={float(fast_value):>14.8f} "
                    f"diff={diff:.10f}"
                )
            else:
                print(f"  {key:<35} old={old_value} fast={fast_value}")

        if evaluator is not None and action_space is not None:
            old_mask = action_space.valid_action_mask(old_state)
            fast_mask = action_space.valid_action_mask(fast_state)

            print("\nNeural/action comparison:")
            print(f"  action_mask_equal: {np.array_equal(old_mask, fast_mask)}")
            print(f"  action_mask_diff_count: {int(np.sum(old_mask != fast_mask))}")

            old_policy, old_value = evaluator.evaluate(old_state, old_mask)
            fast_policy, fast_value = evaluator.evaluate(fast_state, fast_mask)

            print(f"  value_old:  {old_value:+.8f}")
            print(f"  value_fast: {fast_value:+.8f}")
            print(f"  value_diff: {abs(old_value - fast_value):.10f}")
            print(f"  policy_max_abs_diff: {max_abs_diff(old_policy, fast_policy):.10f}")

            old_top = old_policy.argsort()[::-1][:8]
            fast_top = fast_policy.argsort()[::-1][:8]

            actions = action_space.build_all_actions(old_state)

            print("\n  Old top actions:")
            for action_id in old_top:
                action = actions[int(action_id)]
                print(
                    f"    action={int(action_id):>3} | "
                    f"branch={str(action.branch_id):>4} | "
                    f"p={float(old_policy[action_id]):.6f}"
                )

            print("\n  Fast top actions:")
            for action_id in fast_top:
                action = actions[int(action_id)]
                print(
                    f"    action={int(action_id):>3} | "
                    f"branch={str(action.branch_id):>4} | "
                    f"p={float(fast_policy[action_id]):.6f}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
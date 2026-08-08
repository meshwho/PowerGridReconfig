from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


def measure(callable_, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        callable_()
    return (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare canonical and fast power-flow state build cost."
    )
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--branch", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pf-alg", type=int, default=3, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")

    physics_config = replace(
        DEFAULT_PHYSICS_CONFIG,
        pf_alg=int(args.pf_alg),
    )
    adapter = GridFMAdapter(
        args.raw_dir,
        physics_config=physics_config,
    )
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=physics_config,
        enable_cache=False,
    )

    initial = backend.run_power_flow(int(args.scenario))
    if not initial.success or initial.next_state is None:
        raise RuntimeError(initial.message)

    state = initial.next_state
    ppc, frames = backend._build_ppc_from_state(
        state,
        switched_off_branch_id=int(args.branch),
    )
    result_ppc, metrics = backend._solve_ppc(
        ppc,
        context="fast-state-builder benchmark",
    )

    canonical = backend._build_state_from_pypower_result(
        scenario_id=int(args.scenario),
        result_ppc=result_ppc,
        original_frames=frames,
        physical_metrics=metrics,
    )
    fast = backend._build_state_from_pypower_result_fast(
        scenario_id=int(args.scenario),
        result_ppc=result_ppc,
        previous_state=state,
        original_frames=frames,
        physical_metrics=metrics,
    )

    np.testing.assert_allclose(
        fast.bus_features,
        canonical.bus_features,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        fast.branch_features,
        canonical.branch_features,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(fast.branch_status, canonical.branch_status)
    np.testing.assert_array_equal(fast.edge_index, canonical.edge_index)
    np.testing.assert_array_equal(fast.branch_ids, canonical.branch_ids)
    np.testing.assert_array_equal(fast.bus_ids, canonical.bus_ids)
    assert fast.outaged_branch_ids == canonical.outaged_branch_ids

    canonical_seconds = measure(
        lambda: backend._build_state_from_pypower_result(
            scenario_id=int(args.scenario),
            result_ppc=result_ppc,
            original_frames=frames,
            physical_metrics=metrics,
        ),
        int(args.iterations),
    )
    fast_seconds = measure(
        lambda: backend._build_state_from_pypower_result_fast(
            scenario_id=int(args.scenario),
            result_ppc=result_ppc,
            previous_state=state,
            original_frames=frames,
            physical_metrics=metrics,
        ),
        int(args.iterations),
    )

    print(f"Iterations:      {args.iterations}")
    print(f"Canonical/build: {canonical_seconds * 1e6:.1f} us")
    print(f"Fast/build:      {fast_seconds * 1e6:.1f} us")
    print(f"Speedup:         {canonical_seconds / fast_seconds:.2f}x")


if __name__ == "__main__":
    main()

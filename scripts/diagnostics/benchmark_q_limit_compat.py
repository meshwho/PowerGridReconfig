from __future__ import annotations

import argparse
import time
from copy import deepcopy

from pypower.api import case9, ppoption, runpf as stock_runpf
from pypower.idx_gen import QG, QMAX, QMIN

from grid_topology_ai.pypower_compat import (
    get_power_flow_workload_counters,
    reset_power_flow_workload_counters,
    runpf,
)


def _options(qlim: int) -> dict[str, object]:
    return ppoption(
        PF_ALG=3,
        PF_MAX_IT=30,
        PF_MAX_IT_FD=30,
        VERBOSE=0,
        OUT_ALL=0,
        ENFORCE_Q_LIMS=int(qlim),
    )


def _binding_case(num_limits: int) -> dict:
    ppc = case9()
    plain, success = stock_runpf(
        deepcopy(ppc),
        _options(0),
    )
    if not bool(success):
        raise RuntimeError("case9 baseline did not converge")

    num_limits = min(max(int(num_limits), 0), len(ppc["gen"]))
    for gen_index in range(num_limits):
        q_limit = float(plain["gen"][gen_index, QG]) - 1.0
        ppc["gen"][gen_index, QMAX] = q_limit
        if ppc["gen"][gen_index, QMIN] >= q_limit:
            ppc["gen"][gen_index, QMIN] = q_limit - 1000.0

    return ppc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure strict PYPOWER Q-limit solve and compatibility overhead."
        )
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--qlim", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--limits",
        type=int,
        default=1,
        help="Number of case9 generators given a deliberately binding upper Q limit.",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    ppc = _binding_case(args.limits)
    options = _options(args.qlim)
    reset_power_flow_workload_counters()

    started = time.perf_counter()
    successes = 0
    for _ in range(int(args.iterations)):
        _, success = runpf(
            deepcopy(ppc),
            options,
        )
        successes += int(bool(success))
    wall_seconds = time.perf_counter() - started

    counters = get_power_flow_workload_counters()
    iterations = int(args.iterations)
    stock_seconds = float(counters["stock_runpf_seconds"])
    bookkeeping_seconds = float(counters["q_limit_bookkeeping_seconds"])
    other_seconds = max(wall_seconds - stock_seconds - bookkeeping_seconds, 0.0)

    print(f"Iterations:                 {iterations}")
    print(f"Successful runs:            {successes}")
    print(f"Q-limit mode:               {args.qlim}")
    print(f"Forced binding generators:  {args.limits}")
    print(f"Wall / run:                 {wall_seconds / iterations * 1e3:.3f} ms")
    print(f"Stock runpf / run:          {stock_seconds / iterations * 1e3:.3f} ms")
    print(
        "Q-limit bookkeeping / run: "
        f"{bookkeeping_seconds / iterations * 1e3:.3f} ms"
    )
    print(f"Other Python / run:         {other_seconds / iterations * 1e3:.3f} ms")
    print(
        "Stock solves / run:         "
        f"{int(counters['stock_runpf_calls']) / iterations:.3f}"
    )
    print(
        "Q-limit re-solves / run:    "
        f"{int(counters['q_limit_resolves']) / iterations:.3f}"
    )
    print(
        "Q-limit failures:           "
        f"{int(counters['q_limit_failures'])}"
    )
    print(
        "Q-limit infeasible:         "
        f"{int(counters['q_limit_infeasible'])}"
    )
    print(
        "Re-solve histogram:         "
        f"{counters['q_limit_resolve_histogram']}"
    )


if __name__ == "__main__":
    main()

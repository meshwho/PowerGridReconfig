from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _counter_delta(
    before: Mapping[str, object],
    after: Mapping[str, object],
    key: str,
) -> int:
    return max(int(after.get(key, 0)) - int(before.get(key, 0)), 0)


def exact_power_flow_workload(
    before: Mapping[str, object],
    after: Mapping[str, object],
    logical_evaluations: int,
) -> dict[str, object]:
    """Return one scenario's exact-cache, warm-start and solver workload delta."""

    cache_hits = _counter_delta(before, after, "hits")
    cache_misses = _counter_delta(before, after, "misses")
    l1_hits = _counter_delta(before, after, "l1_hits")
    l1_misses = _counter_delta(before, after, "l1_misses")
    l2_hits = _counter_delta(before, after, "l2_hits")
    l2_misses = _counter_delta(before, after, "l2_misses")
    negative_hits = _counter_delta(before, after, "negative_hits")
    evictions = _counter_delta(before, after, "evictions")
    warm_start_hits = _counter_delta(before, after, "warm_start_hits")
    warm_start_misses = _counter_delta(before, after, "warm_start_misses")
    warm_start_evictions = _counter_delta(before, after, "warm_start_evictions")
    stock_runpf_calls = _counter_delta(before, after, "stock_runpf_calls")
    q_limit_resolves = _counter_delta(before, after, "q_limit_resolves")
    cache_lookups = cache_hits + cache_misses

    return {
        "logical_evaluations": int(logical_evaluations),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "cache_hit_rate": (
            float(cache_hits) / float(cache_lookups)
            if cache_lookups > 0
            else 0.0
        ),
        "l1_hits": int(l1_hits),
        "l1_misses": int(l1_misses),
        "l2_enabled": bool(after.get("l2_enabled", False)),
        "l2_hits": int(l2_hits),
        "l2_misses": int(l2_misses),
        "negative_hits": int(negative_hits),
        "evictions": int(evictions),
        "warm_start_enabled": bool(after.get("warm_start_enabled", False)),
        "warm_start_hits": int(warm_start_hits),
        "warm_start_misses": int(warm_start_misses),
        "warm_start_evictions": int(warm_start_evictions),
        "stock_runpf_calls": int(stock_runpf_calls),
        "q_limit_resolves": int(q_limit_resolves),
        "solves_per_cache_miss": (
            float(stock_runpf_calls) / float(cache_misses)
            if cache_misses > 0
            else 0.0
        ),
    }


def aggregate_exact_power_flow_workloads(
    items: Iterable[Mapping[str, object]],
) -> dict[str, Any]:
    rows = list(items)
    summed_keys = (
        "logical_evaluations",
        "cache_hits",
        "cache_misses",
        "l1_hits",
        "l1_misses",
        "l2_hits",
        "l2_misses",
        "negative_hits",
        "evictions",
        "warm_start_hits",
        "warm_start_misses",
        "warm_start_evictions",
        "stock_runpf_calls",
        "q_limit_resolves",
    )
    result: dict[str, Any] = {
        "scenarios": len(rows),
        "l2_enabled": any(bool(item.get("l2_enabled", False)) for item in rows),
        "warm_start_enabled": any(
            bool(item.get("warm_start_enabled", False)) for item in rows
        ),
    }
    for key in summed_keys:
        result[key] = sum(int(item.get(key, 0)) for item in rows)

    cache_lookups = int(result["cache_hits"]) + int(result["cache_misses"])
    result["cache_hit_rate"] = (
        float(result["cache_hits"]) / float(cache_lookups)
        if cache_lookups > 0
        else 0.0
    )
    result["solves_per_cache_miss"] = (
        float(result["stock_runpf_calls"]) / float(result["cache_misses"])
        if int(result["cache_misses"]) > 0
        else 0.0
    )
    return result


def print_exact_power_flow_workload_summary(
    items: Iterable[Mapping[str, object]],
) -> None:
    summary = aggregate_exact_power_flow_workloads(items)

    print("\n" + "=" * 100)
    print("Power-flow workload for scenarios processed in this run")
    print("=" * 100)

    if int(summary["scenarios"]) == 0:
        print("Instrumented scenarios: 0")
        print("No new scenarios were processed in this run.")
        return

    print(f"Instrumented scenarios:    {summary['scenarios']}")
    print(f"Logical evaluations:       {summary['logical_evaluations']}")
    print(f"Exact PF cache hits:       {summary['cache_hits']}")
    print(f"  L1 RAM hits:             {summary['l1_hits']}")
    if bool(summary["l2_enabled"]):
        print(f"  L2 persistent hits:      {summary['l2_hits']}")
    else:
        print("  L2 persistent cache:     disabled")
    print(f"  negative hits:           {summary['negative_hits']}")
    print(f"Exact PF cache misses:     {summary['cache_misses']}")
    print(f"  L1 RAM misses:           {summary['l1_misses']}")
    if bool(summary["l2_enabled"]):
        print(f"  L2 persistent misses:    {summary['l2_misses']}")
    print(f"  L1 evictions:            {summary['evictions']}")
    print(f"Exact cache hit rate:      {summary['cache_hit_rate']:.1%}")
    if bool(summary["warm_start_enabled"]):
        print(f"Warm-start applications:   {summary['warm_start_hits']}")
        print(f"Warm-start misses:         {summary['warm_start_misses']}")
        print(f"Warm-start evictions:      {summary['warm_start_evictions']}")
    else:
        print("PF warm start:             disabled")
    print(f"Stock PYPOWER solves:      {summary['stock_runpf_calls']}")
    print(f"Q-limit re-solves:         {summary['q_limit_resolves']}")
    print(f"Solves / cache miss:       {summary['solves_per_cache_miss']:.3f}")

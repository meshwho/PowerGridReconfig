from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from pypower.api import case9, ppoption, runpf as stock_runpf
from pypower.idx_gen import QG, QMAX, QMIN

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.pypower_compat import (
    get_power_flow_workload_counters,
    reset_power_flow_workload_counters,
    runpf,
)
from scripts.self_play import generate_impact_teacher_provenance as provenance


def _options(*, qlim: int) -> dict[str, object]:
    return ppoption(
        PF_ALG=3,
        PF_MAX_IT=30,
        PF_MAX_IT_FD=30,
        VERBOSE=0,
        OUT_ALL=0,
        ENFORCE_Q_LIMS=qlim,
    )


def _force_one_upper_q_limit(ppc: dict) -> None:
    baseline, success = stock_runpf(
        deepcopy(ppc),
        _options(qlim=0),
    )
    assert bool(success)

    q_limit = float(baseline["gen"][1, QG]) - 1.0
    ppc["gen"][1, QMAX] = q_limit
    if ppc["gen"][1, QMIN] >= q_limit:
        ppc["gen"][1, QMIN] = q_limit - 1000.0


def test_plain_runpf_counts_one_stock_solve() -> None:
    reset_power_flow_workload_counters()

    _, success = runpf(
        deepcopy(case9()),
        _options(qlim=0),
    )

    assert bool(success)
    assert get_power_flow_workload_counters() == {
        "stock_runpf_calls": 1,
        "q_limit_resolves": 0,
    }


def test_q_limit_resolve_counts_additional_stock_solves() -> None:
    ppc = case9()
    _force_one_upper_q_limit(ppc)
    reset_power_flow_workload_counters()

    _, success = runpf(
        deepcopy(ppc),
        _options(qlim=1),
    )

    assert bool(success)
    counters = get_power_flow_workload_counters()
    assert counters["stock_runpf_calls"] >= 2
    assert counters["q_limit_resolves"] == counters["stock_runpf_calls"] - 1


def test_backend_reports_its_own_solver_workload() -> None:
    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=PhysicsConfig(
            pf_alg=3,
            max_iterations=30,
        ),
    )
    reset_power_flow_workload_counters()

    backend._solve_ppc(
        deepcopy(case9()),
        context="workload accounting",
    )

    info = backend.performance_info()
    assert info["stock_runpf_calls"] == 1
    assert info["q_limit_resolves"] == 0
    assert info["hits"] == 0
    assert info["misses"] == 0
    assert info["solves_per_cache_miss"] == 0.0


def test_search_workload_uses_counter_deltas() -> None:
    before = {
        "hits": 10,
        "misses": 20,
        "stock_runpf_calls": 30,
        "q_limit_resolves": 4,
    }
    after = {
        "hits": 18,
        "misses": 25,
        "stock_runpf_calls": 37,
        "q_limit_resolves": 6,
    }

    workload = provenance._search_workload(
        before=before,
        after=after,
        logical_evaluations=15,
    )

    assert workload["logical_evaluations"] == 15
    assert workload["cache_hits"] == 8
    assert workload["cache_misses"] == 5
    assert workload["stock_runpf_calls"] == 7
    assert workload["q_limit_resolves"] == 2
    assert workload["cache_hit_rate"] == 8 / 13
    assert workload["solves_per_cache_miss"] == 7 / 5


def test_instrumented_search_records_one_scenario_delta(monkeypatch) -> None:
    snapshots = iter(
        [
            {
                "hits": 2,
                "misses": 3,
                "stock_runpf_calls": 4,
                "q_limit_resolves": 1,
            },
            {
                "hits": 7,
                "misses": 5,
                "stock_runpf_calls": 7,
                "q_limit_resolves": 2,
            },
        ]
    )

    class _Backend:
        def performance_info(self):
            return next(snapshots)

    def _fake_search(self, env, scenario_id):
        del self, env
        assert scenario_id == 42
        return SimpleNamespace(evaluated_actions=9)

    monkeypatch.setattr(provenance, "_original_planner_search", _fake_search)
    provenance._SEARCH_WORKLOAD_BY_SCENARIO.clear()

    result = provenance._instrumented_planner_search(
        SimpleNamespace(evaluated_actions=0),
        SimpleNamespace(backend=_Backend()),
        scenario_id=42,
    )

    assert result.evaluated_actions == 9
    assert provenance._SEARCH_WORKLOAD_BY_SCENARIO[42] == {
        "logical_evaluations": 9,
        "cache_hits": 5,
        "cache_misses": 2,
        "cache_hit_rate": 5 / 7,
        "stock_runpf_calls": 3,
        "q_limit_resolves": 1,
        "solves_per_cache_miss": 1.5,
    }


def test_parent_aggregation_sums_worker_scenario_results(capsys) -> None:
    provenance._PARENT_WORKLOAD_BY_SCENARIO.clear()
    provenance._PARENT_WORKLOAD_BY_SCENARIO.update(
        {
            1: {
                "logical_evaluations": 100,
                "cache_hits": 60,
                "cache_misses": 40,
                "stock_runpf_calls": 44,
                "q_limit_resolves": 4,
            },
            2: {
                "logical_evaluations": 50,
                "cache_hits": 20,
                "cache_misses": 30,
                "stock_runpf_calls": 36,
                "q_limit_resolves": 6,
            },
        }
    )

    provenance._print_power_flow_workload_summary()
    output = capsys.readouterr().out

    assert "Instrumented scenarios:    2" in output
    assert "Logical evaluations:       150" in output
    assert "PF cache hits:             80" in output
    assert "PF cache misses:           70" in output
    assert "Stock PYPOWER solves:      80" in output
    assert "Q-limit re-solves:         10" in output
    assert "Solves / cache miss:       1.143" in output

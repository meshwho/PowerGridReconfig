from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pypower.idx_bus import BUS_I, VA, VM

from grid_topology_ai.data_adapter import BUS_FEATURE_COLUMNS
from grid_topology_ai.pf_warm_shadow import (
    WarmCandidate,
    WarmStartDescriptor,
    WarmStartShadow,
)
from grid_topology_ai.pf_warm_shadow_runtime import (
    BoundedWarmStartShadow,
    BoundedWarmStartStore,
    install_runtime_warm_shadow,
)
from scripts.pipelines import run_teacher_redispatch as entrypoint
from scripts.self_play import generate_impact_teacher_redispatch_staged as staged


def _descriptor(value: float) -> WarmStartDescriptor:
    return WarmStartDescriptor(
        pd=np.array([value, 2.0 * value], dtype=np.float64),
        qd=np.array([0.5 * value, value], dtype=np.float64),
        pg=np.array([1.2 * value], dtype=np.float64),
        qg=np.array([0.1 * value], dtype=np.float64),
        gen_status=np.array([1.0], dtype=np.float64),
    )


def test_entrypoint_extracts_warm_shadow_runtime_settings() -> None:
    argv = [
        "run_teacher_redispatch.py",
        "--dataset-name",
        "case118_bootstrap_v1",
        "--pf-warm-shadow",
        "--pf-warm-shadow-rate",
        "0.25",
        "--pf-warm-shadow-max-pairs",
        "500",
        "--pf-warm-max-candidates",
        "8",
    ]

    settings = entrypoint._pop_warm_shadow_settings(argv)

    assert settings == (True, 0.25, 500, 8)
    assert "--pf-warm-shadow" not in argv
    assert "--pf-warm-shadow-rate" not in argv
    assert "--pf-warm-shadow-max-pairs" not in argv
    assert "--pf-warm-max-candidates" not in argv


def test_warm_shadow_tuning_requires_shadow_flag() -> None:
    argv = [
        "run_teacher_redispatch.py",
        "--pf-warm-shadow-rate",
        "0.25",
    ]

    with pytest.raises(ValueError, match="require --pf-warm-shadow"):
        entrypoint._pop_warm_shadow_settings(argv)


def test_warm_shadow_requires_runtime_cache_root(monkeypatch) -> None:
    argv = [
        "run_teacher_redispatch.py",
        "--dataset-name",
        "case118_bootstrap_v1",
        "--num-workers",
        "2",
        "--pf-warm-shadow",
    ]

    monkeypatch.delenv(entrypoint._PF_CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(entrypoint.sys, "argv", argv)

    with pytest.raises(SystemExit, match="requires --pf-cache-dir"):
        entrypoint.main()


def test_warm_shadow_options_are_checkpoint_neutral(monkeypatch) -> None:
    captured: dict[str, object] = {}
    argv = [
        "run_teacher_redispatch.py",
        "--dataset-name",
        "case118_bootstrap_v1",
        "--num-workers",
        "2",
        "--pf-cache-dir",
        "cache/power-flow",
        "--pf-warm-shadow",
        "--pf-warm-shadow-rate",
        "0.2",
        "--pf-warm-shadow-max-pairs",
        "500",
        "--pf-warm-max-candidates",
        "12",
    ]

    def fake_pipeline_main() -> None:
        captured["argv"] = list(entrypoint.sys.argv)
        captured["cache"] = entrypoint.os.environ[entrypoint._PF_CACHE_DIR_ENV]
        captured["enabled"] = entrypoint.os.environ[entrypoint._PF_WARM_SHADOW_ENV]
        captured["rate"] = entrypoint.os.environ[entrypoint._PF_WARM_SHADOW_RATE_ENV]
        captured["pairs"] = entrypoint.os.environ[
            entrypoint._PF_WARM_SHADOW_MAX_PAIRS_ENV
        ]
        captured["candidates"] = entrypoint.os.environ[
            entrypoint._PF_WARM_MAX_CANDIDATES_ENV
        ]

    monkeypatch.setattr(entrypoint.sys, "argv", argv)
    monkeypatch.setattr(entrypoint.pipeline, "main", fake_pipeline_main)

    entrypoint.main()

    forwarded = captured["argv"]
    assert isinstance(forwarded, list)
    for option in (
        "--pf-cache-dir",
        "--pf-warm-shadow",
        "--pf-warm-shadow-rate",
        "--pf-warm-shadow-max-pairs",
        "--pf-warm-max-candidates",
    ):
        assert option not in forwarded

    assert captured["cache"] == "cache/power-flow"
    assert captured["enabled"] == "1"
    assert captured["rate"] == "0.2"
    assert captured["pairs"] == "500"
    assert captured["candidates"] == "12"


def test_staged_worker_installs_warm_shadow_from_runtime_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeBackend:
        enable_cache = True
        _persistent_exact_cache = object()
        _warm_start_shadow = None

    class FakeShadow:
        def close(self) -> None:
            return None

    backend = FakeBackend()
    captured: dict[str, object] = {}
    shadow = FakeShadow()

    def fake_install(backend_arg, cache_root, **kwargs):
        captured["backend"] = backend_arg
        captured["cache_root"] = cache_root
        captured.update(kwargs)
        backend_arg._warm_start_shadow = shadow
        return shadow

    cache_root = tmp_path / "pf-cache"
    monkeypatch.setenv(staged._PF_CACHE_DIR_ENV, str(cache_root))
    monkeypatch.setenv(staged._PF_WARM_SHADOW_ENV, "1")
    monkeypatch.setenv(staged._PF_WARM_SHADOW_RATE_ENV, "0.15")
    monkeypatch.setenv(staged._PF_WARM_SHADOW_MAX_PAIRS_ENV, "321")
    monkeypatch.setenv(staged._PF_WARM_MAX_CANDIDATES_ENV, "9")
    monkeypatch.setattr(
        staged.redispatch.base.teacher,
        "_require_worker_context",
        lambda: {"backend": backend},
    )
    monkeypatch.setattr(staged, "install_runtime_warm_shadow", fake_install)
    registered: list[object] = []
    monkeypatch.setattr(staged.atexit, "register", registered.append)

    staged._attach_persistent_pf_cache()

    assert captured["backend"] is backend
    assert captured["cache_root"] == str(cache_root)
    assert captured["sample_rate"] == 0.15
    assert captured["max_pairs"] == 321
    assert captured["max_candidates_per_topology"] == 9
    assert registered == [shadow.close]


def test_shadow_record_limit_is_global_across_connections(tmp_path: Path) -> None:
    left = BoundedWarmStartStore(tmp_path, max_shadow_records=2)
    right = BoundedWarmStartStore(tmp_path, max_shadow_records=2)
    candidate = WarmCandidate(
        exact_key="1" * 64,
        topology_key="a" * 64,
        distance=0.1,
        state_payload=b"candidate",
    )

    try:
        first = left.record_shadow(
            request_exact_key="2" * 64,
            topology_key="a" * 64,
            candidate=candidate,
            scenario_id=1,
            record={"shadow_success": True},
        )
        second = right.record_shadow(
            request_exact_key="3" * 64,
            topology_key="a" * 64,
            candidate=candidate,
            scenario_id=2,
            record={"shadow_success": True},
        )
        third = left.record_shadow(
            request_exact_key="4" * 64,
            topology_key="a" * 64,
            candidate=candidate,
            scenario_id=3,
            record={"shadow_success": True},
        )
        _, records = right.counts()
    finally:
        right.close()
        left.close()

    assert first is True
    assert second is True
    assert third is False
    assert records == 2


def test_runtime_shadow_never_replaces_authoritative_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    authoritative_state = SimpleNamespace()
    authoritative_result = SimpleNamespace(
        success=True,
        next_state=authoritative_state,
        message="Power flow converged.",
    )

    class FakeBackend:
        def run_power_flow_from_state(
            self,
            state,
            switched_off_branch_id=None,
            *,
            action=None,
        ):
            return authoritative_result

        @staticmethod
        def _serialize_exact_state(state):
            return b"authoritative"

    backend = FakeBackend()
    shadow = install_runtime_warm_shadow(
        backend,
        tmp_path,
        sample_rate=1.0,
        max_pairs=10,
        max_candidates_per_topology=4,
    )
    shadow.store.put(
        exact_key="1" * 64,
        topology_key="a" * 64,
        descriptor=_descriptor(10.0),
        state_payload=b"candidate",
    )
    prepared = (
        {"bus": np.zeros((0, 0))},
        {},
        "f" * 64,
        "a" * 64,
        _descriptor(10.0),
    )
    monkeypatch.setattr(shadow, "_prepare", lambda *args: prepared)
    monkeypatch.setattr(
        shadow,
        "_shadow_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("different basin")
        ),
    )

    try:
        result = backend.run_power_flow_from_state(
            SimpleNamespace(scenario_id=99)
        )
        _, records = shadow.store.counts()
    finally:
        shadow.close()

    assert result is authoritative_result
    assert records == 1


def test_runtime_shadow_measures_distinct_initial_voltage_seed(monkeypatch) -> None:
    vm_col = BUS_FEATURE_COLUMNS.index("Vm")
    va_col = BUS_FEATURE_COLUMNS.index("Va")
    features = np.zeros((2, len(BUS_FEATURE_COLUMNS)), dtype=np.float64)
    features[:, vm_col] = [1.04, 0.98]
    features[:, va_col] = [3.0, -6.0]
    warm_state = SimpleNamespace(
        bus_features=features,
        bus_ids=np.array([1, 2], dtype=np.int64),
    )

    class FakeBackend:
        @staticmethod
        def _deserialize_exact_state(payload, request_state):
            return warm_state

    store = SimpleNamespace(close=lambda: None)
    shadow = BoundedWarmStartShadow(FakeBackend(), store, sample_rate=1.0)
    candidate = WarmCandidate(
        exact_key="1" * 64,
        topology_key="a" * 64,
        distance=0.1,
        state_payload=b"candidate",
    )

    bus = np.zeros((2, 13), dtype=np.float64)
    bus[:, BUS_I] = [1, 2]
    bus[:, VM] = [1.0, 1.0]
    bus[:, VA] = [0.0, 0.0]
    ppc = {"bus": bus}
    sentinel = object()

    monkeypatch.setattr(
        WarmStartShadow,
        "_shadow_state",
        lambda self, state, ppc, frames, candidate: sentinel,
    )

    result = shadow._shadow_state(
        SimpleNamespace(),
        ppc,
        {},
        candidate,
    )

    assert result is sentinel
    assert shadow._shadow_diagnostics["initial_seed_distinct"] is True
    assert shadow._shadow_diagnostics["max_initial_vm_delta_pu"] == pytest.approx(0.04)
    assert shadow._shadow_diagnostics["max_initial_va_delta_deg"] == pytest.approx(6.0)
    assert shadow._shadow_diagnostics["shadow_stock_runpf_calls"] == 0
    assert shadow._shadow_diagnostics["shadow_q_limit_resolves"] == 0


def test_runtime_shadow_measures_authoritative_path_and_legacy_warm_usage() -> None:
    authoritative_result = object()

    class FakeBackend:
        warm_start_hits = 0
        cold_start_misses = 0

        def run_power_flow_from_state(self, *args, **kwargs):
            self.warm_start_hits += 1
            return authoritative_result

    backend = FakeBackend()
    store = SimpleNamespace(close=lambda: None)
    shadow = BoundedWarmStartShadow(backend, store, sample_rate=0.0)
    shadow.install()

    result = shadow._run(SimpleNamespace())

    assert result is authoritative_result
    assert shadow._authoritative_diagnostics[
        "authoritative_used_legacy_warm_start"
    ] is True
    assert shadow._authoritative_diagnostics[
        "authoritative_used_cold_start"
    ] is False
    assert shadow._authoritative_diagnostics["authoritative_path_seconds"] >= 0.0
    assert shadow._authoritative_diagnostics["authoritative_stock_runpf_calls"] == 0
    assert shadow._authoritative_diagnostics["authoritative_q_limit_resolves"] == 0


def test_runtime_shadow_comparison_includes_path_diagnostics(monkeypatch) -> None:
    store = SimpleNamespace(close=lambda: None)
    shadow = BoundedWarmStartShadow(SimpleNamespace(), store, sample_rate=0.0)
    shadow._authoritative_diagnostics = {
        "authoritative_path_seconds": 0.2,
        "authoritative_used_legacy_warm_start": False,
    }
    shadow._shadow_diagnostics = {
        "max_initial_vm_delta_pu": 0.03,
        "max_initial_va_delta_deg": 4.0,
        "initial_seed_distinct": True,
        "shadow_path_seconds": 0.1,
    }

    monkeypatch.setattr(
        WarmStartShadow,
        "_compare",
        lambda self, authoritative, candidate: {"shadow_success": True},
    )

    record = shadow._compare(object(), object())

    assert record["shadow_success"] is True
    assert record["max_initial_vm_delta_pu"] == 0.03
    assert record["max_initial_va_delta_deg"] == 4.0
    assert record["initial_seed_distinct"] is True
    assert record["authoritative_path_seconds"] == 0.2
    assert record["shadow_path_seconds"] == 0.1

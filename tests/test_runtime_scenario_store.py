from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.power_flow_problem import build_scenario_power_flow_template
from grid_topology_ai.runtime import (
    MemoryMappedGridFMAdapter,
    MemoryMappedScenarioStore,
    ensure_runtime_scenario_store,
    validate_runtime_scenario_store,
)


def _write_raw_dataset(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    bus_rows = []
    branch_rows = []
    gen_rows = []

    for scenario, load_scale in ((1, 1.0), (2, 1.15)):
        bus_rows.extend(
            [
                {
                    "scenario": scenario,
                    "load_scenario_idx": float(scenario),
                    "bus": 1,
                    "Pd": 0.0,
                    "Qd": 0.0,
                    "Pg": 95.0 * load_scale,
                    "Qg": 15.0,
                    "Vm": 1.02,
                    "Va": 0.0,
                    "PQ": 0.0,
                    "PV": 0.0,
                    "REF": 1.0,
                    "vn_kv": 110.0,
                    "GS": 0.0,
                    "BS": 0.0,
                    "min_vm_pu": 0.95,
                    "max_vm_pu": 1.05,
                },
                {
                    "scenario": scenario,
                    "load_scenario_idx": float(scenario),
                    "bus": 2,
                    "Pd": 45.0 * load_scale,
                    "Qd": 12.0 * load_scale,
                    "Pg": 0.0,
                    "Qg": 0.0,
                    "Vm": 0.99,
                    "Va": -1.0,
                    "PQ": 1.0,
                    "PV": 0.0,
                    "REF": 0.0,
                    "vn_kv": 110.0,
                    "GS": 0.0,
                    "BS": 0.0,
                    "min_vm_pu": 0.95,
                    "max_vm_pu": 1.05,
                },
                {
                    "scenario": scenario,
                    "load_scenario_idx": float(scenario),
                    "bus": 3,
                    "Pd": 40.0 * load_scale,
                    "Qd": 10.0 * load_scale,
                    "Pg": 0.0,
                    "Qg": 0.0,
                    "Vm": 0.985,
                    "Va": -1.5,
                    "PQ": 1.0,
                    "PV": 0.0,
                    "REF": 0.0,
                    "vn_kv": 110.0,
                    "GS": 0.0,
                    "BS": 0.0,
                    "min_vm_pu": 0.95,
                    "max_vm_pu": 1.05,
                },
            ]
        )

        flows = (38.0 * load_scale, 27.0 * load_scale, 24.0 * load_scale)
        for index, (branch_id, from_bus, to_bus, pf) in enumerate(
            (
                (10, 1, 2, flows[0]),
                (11, 2, 3, flows[1]),
                (12, 1, 3, flows[2]),
            )
        ):
            branch_rows.append(
                {
                    "scenario": scenario,
                    "load_scenario_idx": float(scenario),
                    "idx": branch_id,
                    "from_bus": from_bus,
                    "to_bus": to_bus,
                    "pf": pf,
                    "qf": 4.0 + index,
                    "pt": -pf + 0.2,
                    "qt": -(4.0 + index) + 0.1,
                    "r": 0.01 + 0.001 * index,
                    "x": 0.08 + 0.01 * index,
                    "b": 0.02,
                    "tap": 0.0,
                    "shift": 0.0,
                    "rate_a": 100.0,
                    "br_status": 1.0,
                    "ang_min": -360.0,
                    "ang_max": 360.0,
                }
            )

        gen_rows.append(
            {
                "scenario": scenario,
                "idx": 100,
                "bus": 1,
                "p_mw": 95.0 * load_scale,
                "q_mvar": 15.0,
                "min_p_mw": 0.0,
                "max_p_mw": 180.0,
                "min_q_mvar": -80.0,
                "max_q_mvar": 80.0,
                "in_service": 1.0,
            }
        )

    pd.DataFrame(bus_rows).to_parquet(raw_dir / "bus_data.parquet", index=False)
    pd.DataFrame(branch_rows).to_parquet(
        raw_dir / "branch_data.parquet",
        index=False,
    )
    pd.DataFrame(gen_rows).to_parquet(raw_dir / "gen_data.parquet", index=False)


def _assert_state_equal(left, right) -> None:
    assert int(left.scenario_id) == int(right.scenario_id)
    assert float(left.load_scenario_idx) == float(right.load_scenario_idx)
    np.testing.assert_array_equal(left.bus_features, right.bus_features)
    np.testing.assert_array_equal(left.branch_features, right.branch_features)
    np.testing.assert_array_equal(left.edge_index, right.edge_index)
    np.testing.assert_array_equal(left.branch_ids, right.branch_ids)
    np.testing.assert_array_equal(left.branch_status, right.branch_status)
    np.testing.assert_array_equal(left.bus_ids, right.bus_ids)
    assert left.metrics == right.metrics
    assert list(left.outaged_branch_ids) == list(right.outaged_branch_ids)


def test_memory_mapped_adapter_matches_canonical_adapter(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_dataset(raw_dir)
    physics = PhysicsConfig()

    canonical = GridFMAdapter(
        raw_dir,
        scenario_ids=[1, 2],
        physics_config=physics,
    )
    store_dir = ensure_runtime_scenario_store(raw_dir)
    runtime = MemoryMappedGridFMAdapter(
        store_dir,
        scenario_ids=[1, 2],
        physics_config=physics,
    )

    assert runtime.scenario_ids() == canonical.scenario_ids()
    assert not hasattr(runtime, "bus_df")
    assert not hasattr(runtime, "branch_df")
    assert not hasattr(runtime, "gen_df")

    for scenario_id in (1, 2):
        _assert_state_equal(
            canonical.build_state(scenario_id),
            runtime.build_state(scenario_id),
        )


def test_runtime_frames_build_identical_power_flow_templates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_dataset(raw_dir)
    physics = PhysicsConfig()
    canonical = GridFMAdapter(raw_dir, physics_config=physics)
    runtime = MemoryMappedGridFMAdapter(
        ensure_runtime_scenario_store(raw_dir),
        physics_config=physics,
    )

    for scenario_id in (1, 2):
        canonical_bus = canonical.bus_df[
            canonical.bus_df["scenario"] == scenario_id
        ]
        canonical_branch = canonical.branch_df[
            canonical.branch_df["scenario"] == scenario_id
        ]
        canonical_gen = canonical.gen_df[
            canonical.gen_df["scenario"] == scenario_id
        ]
        runtime_frames = runtime.scenario_frames(scenario_id)

        expected = build_scenario_power_flow_template(
            scenario_id=scenario_id,
            bus_df=canonical_bus,
            branch_df=canonical_branch,
            gen_df=canonical_gen,
            base_mva=physics.base_mva,
        )
        actual = build_scenario_power_flow_template(
            scenario_id=scenario_id,
            bus_df=runtime_frames["bus"],
            branch_df=runtime_frames["branch"],
            gen_df=runtime_frames["gen"],
            base_mva=physics.base_mva,
        )

        np.testing.assert_array_equal(actual.bus_ids, expected.bus_ids)
        np.testing.assert_array_equal(actual.branch_ids, expected.branch_ids)
        np.testing.assert_array_equal(actual.generator_ids, expected.generator_ids)
        np.testing.assert_array_equal(actual.bus, expected.bus)
        np.testing.assert_array_equal(actual.branch, expected.branch)
        np.testing.assert_array_equal(actual.gen, expected.gen)


def test_runtime_store_uses_read_only_memory_maps(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_dataset(raw_dir)
    store = MemoryMappedScenarioStore(ensure_runtime_scenario_store(raw_dir))

    assert store.scenario_ids() == (1, 2)
    for array in store._arrays.values():
        assert isinstance(array, np.memmap)
        assert not array.flags.writeable


def test_runtime_store_rebuilds_after_source_change(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_dataset(raw_dir)
    store_dir = ensure_runtime_scenario_store(raw_dir)
    before = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))

    bus = pd.read_parquet(raw_dir / "bus_data.parquet")
    bus.loc[(bus["scenario"] == 2) & (bus["bus"] == 2), "Qd"] += 0.25
    bus.to_parquet(raw_dir / "bus_data.parquet", index=False)

    rebuilt_dir = ensure_runtime_scenario_store(raw_dir)
    after = json.loads((rebuilt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert rebuilt_dir == store_dir
    assert before["source_fingerprint"] != after["source_fingerprint"]

    canonical = GridFMAdapter(raw_dir, scenario_ids=[2])
    runtime = MemoryMappedGridFMAdapter(rebuilt_dir, scenario_ids=[2])
    _assert_state_equal(canonical.build_state(2), runtime.build_state(2))


def test_runtime_store_recovers_from_corrupt_payload(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_dataset(raw_dir)
    store_dir = ensure_runtime_scenario_store(raw_dir)
    manifest = validate_runtime_scenario_store(store_dir, verify_hashes=True)
    tables = manifest["tables"]
    assert isinstance(tables, dict)
    bus_meta = tables["bus"]
    assert isinstance(bus_meta, dict)
    bus_path = store_dir / str(bus_meta["file"])

    with bus_path.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 0x01]))

    rebuilt = ensure_runtime_scenario_store(raw_dir)
    validate_runtime_scenario_store(rebuilt, verify_hashes=True)


def test_production_pipeline_uses_runtime_teacher_entrypoint() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline_source = (
        root / "scripts" / "pipelines" / "run_teacher_redispatch.py"
    ).read_text(encoding="utf-8")
    runtime_source = (
        root
        / "scripts"
        / "self_play"
        / "generate_impact_teacher_redispatch_runtime.py"
    ).read_text(encoding="utf-8")

    assert "generate_impact_teacher_redispatch_runtime" in pipeline_source
    assert "ensure_runtime_scenario_store" in runtime_source
    assert "build_memory_mapped_teacher_context" in runtime_source
    assert "_RUNTIME_SCENARIO_STORE_DIR" in runtime_source

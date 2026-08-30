from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import scripts.data.build_balanced_gridfm_dataset as builder


def _args() -> Namespace:
    return Namespace(
        min_loading=105.0,
        max_loading=260.0,
        simple_min_loading=105.0,
        simple_max_loading=120.0,
        simple_max_hard=0,
        simple_max_overloaded=2,
        medium_min_loading=120.0,
        medium_max_loading=150.0,
        medium_max_hard=1,
        medium_max_overloaded=5,
        hard_min_loading=150.0,
        hard_min_hard=2,
    )


def _contract_args(**overrides) -> Namespace:
    values = {
        "dataset_name": "case118_bootstrap_v1",
        "network_name": "case118_ieee",
        "network_source": "pglib",
        "raw_network_dir_name": "case118_ieee",
        "chunk_size": 1000,
        "seed_start": 20000,
        "num_processes": 4,
        "sigma": 0.20,
        "global_range": 0.50,
        "max_scaling_factor": 2.0,
        "step_size": 0.10,
        "start_scaling_factor": 1.0,
        "topology_variants": 10,
        "topology_k": 1,
        "generation_perturbation_type": "cost_perturbation",
        "generation_perturbation_sigma": 0.15,
        "admittance_perturbation_type": "random_perturbation",
        "admittance_perturbation_sigma": 0.02,
        "min_loading": 105.0,
        "max_loading": 260.0,
        "simple_min_loading": 105.0,
        "simple_max_loading": 120.0,
        "simple_max_hard": 0,
        "simple_max_overloaded": 2,
        "medium_min_loading": 120.0,
        "medium_max_loading": 150.0,
        "medium_max_hard": 1,
        "medium_max_overloaded": 5,
        "hard_min_loading": 150.0,
        "hard_min_hard": 2,
    }
    values.update(overrides)
    return Namespace(**values)


def _metrics(loading: float, overloaded: int, hard: int) -> dict[str, float | int]:
    return {
        "max_loading_percent": loading,
        "mean_loading_percent": loading / 2.0,
        "num_overloaded_branches": overloaded,
        "num_hard_overloaded_branches": hard,
        "num_outaged_branches": 1,
        "min_vm_pu": 0.97,
        "max_vm_pu": 1.03,
        "total_low_voltage_violation": 0.0,
        "total_high_voltage_violation": 0.0,
        "total_voltage_violation": 0.0,
        "num_low_voltage_buses": 0,
        "num_high_voltage_buses": 0,
    }


def test_canonical_candidates_are_filtered_and_reclassified(monkeypatch) -> None:
    candidates = pd.DataFrame(
        [
            {
                "scenario": 1,
                "difficulty_class": "hard",
                "source_raw_dir": "raw",
                "source_scenario_id": 1,
                "max_loading_percent": 180.0,
                "num_overloaded_branches": 3,
                "num_hard_overloaded_branches": 2,
                "num_outaged_branches": 1,
            },
            {
                "scenario": 2,
                "difficulty_class": "medium",
                "source_raw_dir": "raw",
                "source_scenario_id": 2,
                "max_loading_percent": 130.0,
                "num_overloaded_branches": 1,
                "num_hard_overloaded_branches": 1,
                "num_outaged_branches": 1,
            },
            {
                "scenario": 3,
                "difficulty_class": "medium",
                "source_raw_dir": "raw",
                "source_scenario_id": 3,
                "max_loading_percent": 130.0,
                "num_overloaded_branches": 1,
                "num_hard_overloaded_branches": 1,
                "num_outaged_branches": 1,
            },
        ]
    )

    class FakeAdapter:
        def __init__(self, raw_dir, scenario_ids=None):
            self.raw_dir = raw_dir
            self.scenario_ids = scenario_ids

    class FakeBackend:
        def __init__(self, adapter, enable_cache=False):
            self.adapter = adapter

        def run_power_flow(self, scenario_id, switched_off_branch_id):
            assert switched_off_branch_id is None
            if scenario_id == 2:
                return SimpleNamespace(
                    success=False,
                    next_state=None,
                    failure_kind=SimpleNamespace(value="not_converged"),
                )
            return SimpleNamespace(
                success=True,
                next_state=SimpleNamespace(
                    metrics={
                        1: _metrics(130.0, 1, 1),
                        3: _metrics(160.0, 2, 1),
                    }[scenario_id],
                    outaged_branch_ids=[scenario_id + 10],
                ),
                failure_kind=None,
            )

    monkeypatch.setattr(builder, "GridFMAdapter", FakeAdapter)
    monkeypatch.setattr(builder, "GridFMPowerFlowBackend", FakeBackend)

    evaluated, valid = builder.evaluate_canonical_candidates(candidates, _args())

    assert evaluated["canonical_pf_ok"].tolist() == [True, False, True]
    assert evaluated.loc[1, "canonical_failure_kind"] == "not_converged"
    by_scenario = valid.set_index("source_scenario_id")
    assert set(by_scenario.index) == {1, 3}
    assert by_scenario.loc[1, "difficulty_class"] == "medium"
    assert by_scenario.loc[3, "difficulty_class"] == "hard"


def test_gridfm_variants_are_deduplicated_before_selection() -> None:
    candidates = pd.DataFrame(
        {
            "difficulty_class": ["simple"] * 5,
            "source_chunk": ["chunk00", "chunk00", "chunk00", "chunk01", "chunk00"],
            "source_scenario_id": [1, 2, 3, 4, 5],
            "load_scenario_idx": [7.0, 7.0, 7.0, 7.0, 8.0],
            "outaged_branch_ids": [[12], "[12]", [13], [12], [12]],
            "max_loading_percent": [110.0] * 5,
            "num_overloaded_branches": [1] * 5,
        }
    )

    deduplicated = builder.deduplicate_gridfm_variants(candidates)
    assert deduplicated["source_scenario_id"].tolist() == [1, 3, 4, 5]


def test_completion_markers_guard_resume_artifacts(tmp_path: Path) -> None:
    candidates_path = tmp_path / "chunk01_candidates.csv"
    pd.DataFrame(
        {
            "difficulty_class": ["simple"],
            "source_scenario_id": [1],
            "gridfm_difficulty_class": ["simple"],
            "canonical_pf_ok": [True],
        }
    ).to_csv(candidates_path, index=False)

    assert not builder.candidate_manifest_is_current(candidates_path)
    builder._write_completion_marker(
        builder.candidate_completion_marker_path(candidates_path),
        stage="canonical_candidates",
        contract_fingerprint="contract-a",
        chunk_index=1,
    )
    assert builder.candidate_manifest_is_current(
        candidates_path,
        expected_contract_fingerprint="contract-a",
        chunk_index=1,
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in builder._REQUIRED_GRIDFM_DATASETS:
        dataset_dir = raw_dir / name
        dataset_dir.mkdir()
        (dataset_dir / "part-0.parquet").write_bytes(b"data")
    (raw_dir / "n_scenarios.txt").write_text("1000", encoding="utf-8")

    assert not builder.raw_completion_is_current(
        raw_dir,
        contract_fingerprint="contract-a",
        chunk_index=0,
    )
    builder._write_completion_marker(
        raw_dir / builder._RAW_COMPLETION_MARKER,
        stage="gridfm_raw",
        contract_fingerprint="contract-a",
        chunk_index=0,
    )
    assert builder.raw_completion_is_current(
        raw_dir,
        contract_fingerprint="contract-a",
        chunk_index=0,
    )


def test_generation_contract_changes_with_gridfm_semantics(monkeypatch) -> None:
    monkeypatch.setattr(builder, "_package_version", lambda name: "1.0.5")
    baseline = builder.dataset_contract_fingerprint(_contract_args())

    assert baseline != builder.dataset_contract_fingerprint(_contract_args(seed_start=20001))
    assert baseline != builder.dataset_contract_fingerprint(_contract_args(num_processes=3))
    assert baseline != builder.dataset_contract_fingerprint(
        _contract_args(admittance_perturbation_sigma=0.03)
    )

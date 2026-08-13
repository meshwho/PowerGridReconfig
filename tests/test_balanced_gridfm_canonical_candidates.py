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
        "gridfm_command_template": (
            'python -m gridfm_datakit.cli generate "{config}"'
        ),
    }
    values.update(overrides)
    return Namespace(**values)


def _metrics(
    *,
    loading: float,
    overloaded: int,
    hard: int,
) -> dict[str, float | int]:
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
                "source_chunk": "chunk00",
                "source_raw_dir": "raw",
                "source_scenario_id": 1,
                "max_loading_percent": 180.0,
                "mean_loading_percent": 90.0,
                "num_overloaded_branches": 3,
                "num_hard_overloaded_branches": 2,
                "num_outaged_branches": 1,
                "outaged_branch_ids": [7],
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
                "total_low_voltage_violation": 0.0,
                "total_high_voltage_violation": 0.0,
                "total_voltage_violation": 0.0,
                "num_low_voltage_buses": 0,
                "num_high_voltage_buses": 0,
                "seriousness_score": 5000.0,
            },
            {
                "scenario": 2,
                "difficulty_class": "medium",
                "source_chunk": "chunk00",
                "source_raw_dir": "raw",
                "source_scenario_id": 2,
                "max_loading_percent": 130.0,
                "mean_loading_percent": 65.0,
                "num_overloaded_branches": 1,
                "num_hard_overloaded_branches": 1,
                "num_outaged_branches": 1,
                "outaged_branch_ids": [8],
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
                "total_low_voltage_violation": 0.0,
                "total_high_voltage_violation": 0.0,
                "total_voltage_violation": 0.0,
                "num_low_voltage_buses": 0,
                "num_high_voltage_buses": 0,
                "seriousness_score": 3000.0,
            },
            {
                "scenario": 3,
                "difficulty_class": "medium",
                "source_chunk": "chunk00",
                "source_raw_dir": "raw",
                "source_scenario_id": 3,
                "max_loading_percent": 130.0,
                "mean_loading_percent": 65.0,
                "num_overloaded_branches": 1,
                "num_hard_overloaded_branches": 1,
                "num_outaged_branches": 1,
                "outaged_branch_ids": [9],
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
                "total_low_voltage_violation": 0.0,
                "total_high_voltage_violation": 0.0,
                "total_voltage_violation": 0.0,
                "num_low_voltage_buses": 0,
                "num_high_voltage_buses": 0,
                "seriousness_score": 3000.0,
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
            self.enable_cache = enable_cache

        def run_power_flow(self, scenario_id, switched_off_branch_id):
            assert switched_off_branch_id is None

            if scenario_id == 2:
                return SimpleNamespace(
                    success=False,
                    next_state=None,
                    failure_kind=SimpleNamespace(value="not_converged"),
                )

            canonical = {
                1: _metrics(loading=130.0, overloaded=1, hard=1),
                3: _metrics(loading=160.0, overloaded=2, hard=1),
            }[scenario_id]

            return SimpleNamespace(
                success=True,
                next_state=SimpleNamespace(
                    metrics=canonical,
                    outaged_branch_ids=[scenario_id + 10],
                ),
                failure_kind=None,
            )

    monkeypatch.setattr(builder, "GridFMAdapter", FakeAdapter)
    monkeypatch.setattr(builder, "GridFMPowerFlowBackend", FakeBackend)

    evaluated, valid = builder.evaluate_canonical_candidates(
        candidates,
        _args(),
    )

    assert evaluated["canonical_pf_ok"].tolist() == [True, False, True]
    assert evaluated["gridfm_difficulty_class"].tolist() == [
        "hard",
        "medium",
        "medium",
    ]
    assert evaluated.loc[1, "canonical_failure_kind"] == "not_converged"

    by_scenario = valid.set_index("source_scenario_id")
    assert set(by_scenario.index) == {1, 3}
    assert by_scenario.loc[1, "difficulty_class"] == "medium"
    assert by_scenario.loc[3, "difficulty_class"] == "hard"
    assert by_scenario.loc[1, "max_loading_percent"] == 130.0
    assert by_scenario.loc[3, "max_loading_percent"] == 160.0


def test_gridfm_topology_variants_are_deduplicated_before_selection() -> None:
    def candidate(
        scenario_id: int,
        *,
        chunk: str,
        load_scenario_idx: float,
        outage,
    ) -> dict:
        return {
            "difficulty_class": "simple",
            "source_chunk": chunk,
            "source_scenario_id": scenario_id,
            "load_scenario_idx": load_scenario_idx,
            "outaged_branch_ids": outage,
            "max_loading_percent": 110.0,
            "num_overloaded_branches": 1,
        }

    candidates = pd.DataFrame(
        [
            candidate(
                1,
                chunk="chunk00",
                load_scenario_idx=7.0,
                outage=[12],
            ),
            candidate(
                2,
                chunk="chunk00",
                load_scenario_idx=7.0,
                outage="[12]",
            ),
            candidate(
                3,
                chunk="chunk00",
                load_scenario_idx=7.0,
                outage=[13],
            ),
            candidate(
                4,
                chunk="chunk01",
                load_scenario_idx=7.0,
                outage=[12],
            ),
            candidate(
                5,
                chunk="chunk00",
                load_scenario_idx=8.0,
                outage=[12],
            ),
        ]
    )

    deduplicated = builder.deduplicate_gridfm_variants(
        candidates
    )

    assert deduplicated["source_scenario_id"].tolist() == [
        1,
        3,
        4,
        5,
    ]

    selected = builder.select_balanced_manifest(
        candidates=candidates,
        targets=builder.ClassTargets(
            simple=4,
            medium=0,
            hard=0,
        ),
        seed=42,
    )

    assert len(selected) == 4
    assert set(selected["source_scenario_id"]) == {
        1,
        3,
        4,
        5,
    }


def test_resume_candidate_pool_deduplicates_old_gridfm_variants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunk00_candidates.csv"

    pd.DataFrame(
        [
            {
                "difficulty_class": "hard",
                "gridfm_difficulty_class": "hard",
                "canonical_pf_ok": True,
                "source_chunk": "chunk00",
                "source_scenario_id": 100,
                "load_scenario_idx": 25.0,
                "outaged_branch_ids": [166],
            },
            {
                "difficulty_class": "hard",
                "gridfm_difficulty_class": "hard",
                "canonical_pf_ok": True,
                "source_chunk": "chunk00",
                "source_scenario_id": 101,
                "load_scenario_idx": 25.0,
                "outaged_branch_ids": [166],
            },
        ]
    ).to_csv(path, index=False)

    builder._write_completion_marker(
        builder.candidate_completion_marker_path(path),
        stage="canonical_candidates",
        contract_fingerprint="contract-a",
        chunk_index=0,
    )

    candidates = builder.load_existing_candidates(
        tmp_path,
        contract_fingerprint="contract-a",
    )

    assert len(candidates) == 1
    assert candidates["source_scenario_id"].tolist() == [100]


def test_final_targets_respect_requested_total_when_quotas_exist() -> None:
    candidates = pd.DataFrame(
        {
            "difficulty_class": (
                ["simple"] * 35
                + ["medium"] * 33
                + ["hard"] * 25
            )
        }
    )
    requested = builder.ClassTargets(simple=3, medium=6, hard=3)

    final = builder.compute_final_targets(
        candidates=candidates,
        requested=requested,
        simple_fraction=0.25,
        medium_fraction=0.50,
        hard_fraction=0.25,
    )

    assert final == requested
    assert final.total == 12


def test_candidate_manifest_requires_matching_completion_marker(tmp_path: Path) -> None:
    old_path = tmp_path / "chunk00_candidates.csv"
    pd.DataFrame(
        {
            "difficulty_class": ["simple"],
            "source_scenario_id": [1],
        }
    ).to_csv(old_path, index=False)

    assert not builder.candidate_manifest_is_current(old_path)

    current_path = tmp_path / "chunk01_candidates.csv"
    pd.DataFrame(
        {
            "difficulty_class": ["simple"],
            "source_scenario_id": [1],
            "gridfm_difficulty_class": ["simple"],
            "canonical_pf_ok": [True],
        }
    ).to_csv(current_path, index=False)

    assert not builder.candidate_manifest_is_current(current_path)

    marker_path = builder.candidate_completion_marker_path(current_path)
    builder._write_completion_marker(
        marker_path,
        stage="canonical_candidates",
        contract_fingerprint="contract-a",
        chunk_index=1,
    )

    assert builder.candidate_manifest_is_current(
        current_path,
        expected_contract_fingerprint="contract-a",
        chunk_index=1,
    )
    assert not builder.candidate_manifest_is_current(
        current_path,
        expected_contract_fingerprint="contract-b",
        chunk_index=1,
    )


def test_partial_gridfm_raw_is_never_current_without_marker(tmp_path: Path) -> None:
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
        extra={
            "requested_scenarios": 1000,
            "generated_scenarios": 1000,
        },
    )

    assert builder.raw_completion_is_current(
        raw_dir,
        contract_fingerprint="contract-a",
        chunk_index=0,
    )
    assert not builder.raw_completion_is_current(
        raw_dir,
        contract_fingerprint="contract-b",
        chunk_index=0,
    )

    (raw_dir / "gen_data.parquet" / "part-0.parquet").unlink()

    assert not builder.raw_completion_is_current(
        raw_dir,
        contract_fingerprint="contract-a",
        chunk_index=0,
    )


def test_generation_contract_changes_with_gridfm_semantics(monkeypatch) -> None:
    monkeypatch.setattr(
        builder,
        "_package_version",
        lambda name: "1.0.5",
    )

    baseline = builder.dataset_contract_fingerprint(_contract_args())

    assert baseline != builder.dataset_contract_fingerprint(
        _contract_args(seed_start=20001)
    )
    assert baseline != builder.dataset_contract_fingerprint(
        _contract_args(num_processes=3)
    )
    assert baseline != builder.dataset_contract_fingerprint(
        _contract_args(admittance_perturbation_sigma=0.03)
    )


def test_gridfm_wrapper_streams_output_and_uses_safe_bootstrap_chunks() -> None:
    source = Path(
        "scripts/pipelines/build_gridfm_dataset.ps1"
    ).read_text(encoding="utf-8")

    assert "Output is streamed live to this console." in source
    assert 'PYTHONUNBUFFERED = "1"' in source
    assert '$ErrorActionPreference = "Stop"' in source
    assert "$PreviousErrorActionPreference = $ErrorActionPreference" in source
    assert '$ErrorActionPreference = "Continue"' in source
    assert "$ErrorActionPreference = $PreviousErrorActionPreference" in source
    assert "$ExitCode = $LASTEXITCODE" in source
    assert "$MaxSafeGridFMProcesses = 4" in source
    assert '"--gridfm-retries", "$GridFMRetries"' in source
    assert (
        '"--gridfm-inactivity-timeout-sec", '
        '"$GridFMInactivityTimeoutSec"'
    ) in source
    assert "$ChunkSize = 1000" in source
    assert "$MaxChunks = 160" in source
    assert "Output will be printed when the process finishes." not in source


def test_default_hard_threshold_keeps_single_hard_overload_medium() -> None:
    source = Path(
        "scripts/data/build_balanced_gridfm_dataset.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--hard-min-hard", type=int, default=2)' in source

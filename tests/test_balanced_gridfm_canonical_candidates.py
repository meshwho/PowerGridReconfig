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


def test_old_candidate_manifest_is_not_reused(tmp_path: Path) -> None:
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

    assert builder.candidate_manifest_is_current(current_path)


def test_default_hard_threshold_keeps_single_hard_overload_medium() -> None:
    source = Path(
        "scripts/data/build_balanced_gridfm_dataset.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--hard-min-hard", type=int, default=2)' in source

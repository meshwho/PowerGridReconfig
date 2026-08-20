from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.self_play import provenance
from grid_topology_ai.self_play import replay as replay_module
from grid_topology_ai.self_play.provenance import (
    LINEAGE_COLUMNS,
    annotate_examples_csv,
    annotate_transitions_csv,
    build_scenario_lineages,
)
from grid_topology_ai.self_play import stages


def _raw_frames() -> dict[str, pd.DataFrame]:
    bus_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    gen_rows: list[dict[str, object]] = []
    profiles = {
        1: ((10.0, 2.0), (20.0, 4.0)),
        2: ((10.0, 2.0), (20.0, 4.0)),
        3: ((11.0, 2.0), (20.0, 4.0)),
        4: ((10.0, 2.0), (20.0, 4.0)),
    }
    outages = {1: 1, 2: 1, 3: 1, 4: 2}

    for scenario_id, loads in profiles.items():
        for bus_id, (pd_mw, qd_mvar) in enumerate(loads):
            bus_rows.append(
                {
                    "scenario": scenario_id,
                    "bus": bus_id,
                    "Pd": pd_mw,
                    "Qd": qd_mvar,
                    "vn_kv": 110.0,
                    "GS": 0.0,
                    "BS": 0.0,
                }
            )
        for branch_id, (from_bus, to_bus) in enumerate(((0, 1), (1, 0)), start=1):
            branch_rows.append(
                {
                    "scenario": scenario_id,
                    "idx": branch_id,
                    "from_bus": from_bus,
                    "to_bus": to_bus,
                    "r": 0.01 * branch_id,
                    "x": 0.10 * branch_id,
                    "b": 0.001 * branch_id,
                    "tap": 1.0,
                    "shift": 0.0,
                    "rate_a": 100.0,
                    "br_status": 0.0 if branch_id == outages[scenario_id] else 1.0,
                }
            )
        gen_rows.append(
            {
                "scenario": scenario_id,
                "idx": 0,
                "bus": 0,
                "max_p_mw": 100.0,
                "min_p_mw": 0.0,
                "max_q_mvar": 50.0,
                "min_q_mvar": -50.0,
            }
        )

    return {
        "bus_data.parquet": pd.DataFrame(bus_rows),
        "branch_data.parquet": pd.DataFrame(branch_rows),
        "gen_data.parquet": pd.DataFrame(gen_rows),
    }


def _install_raw_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    frames = _raw_frames()
    for name in frames:
        (raw_dir / name).touch()

    def fake_read_parquet(
        path: str | Path,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        del kwargs
        frame = frames[Path(path).name]
        if columns is None:
            return frame.copy()
        return frame[columns].copy()

    monkeypatch.setattr(
        provenance.pd,
        "read_parquet",
        fake_read_parquet,
    )
    return raw_dir


def test_raw_physics_defines_lineage_not_scenario_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _install_raw_data(tmp_path, monkeypatch)
    lineages = build_scenario_lineages(
        raw_dir=raw_dir,
        scenario_ids=[1, 2, 3, 4],
    )

    assert lineages[1].fingerprint == lineages[2].fingerprint
    assert lineages[1].fingerprint != lineages[3].fingerprint
    assert lineages[1].fingerprint != lineages[4].fingerprint
    assert lineages[1].contingency_family_id == "branch:1"
    assert lineages[4].contingency_family_id == "branch:2"


def test_transitions_and_examples_receive_same_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _install_raw_data(tmp_path, monkeypatch)
    transitions_path = tmp_path / "transitions.csv"
    examples_path = tmp_path / "examples.csv"
    pd.DataFrame(
        {
            "scenario_id": [1, 2],
            "difficulty_class": ["medium", "medium"],
        }
    ).to_csv(transitions_path, index=False)
    pd.DataFrame(
        {
            "scenario_id": [1, 1, 2],
            "state_id": ["s1-0", "s1-1", "s2-0"],
        }
    ).to_csv(examples_path, index=False)

    annotate_transitions_csv(
        transitions_csv=transitions_path,
        raw_dir=raw_dir,
    )
    annotate_examples_csv(
        examples_csv=examples_path,
        transitions_csv=transitions_path,
    )

    transitions = pd.read_csv(transitions_path)
    examples = pd.read_csv(examples_path)
    assert set(LINEAGE_COLUMNS).issubset(transitions.columns)
    assert set(LINEAGE_COLUMNS).issubset(examples.columns)
    expected = transitions.set_index("scenario_id")[
        "physical_lineage_fingerprint"
    ].to_dict()
    assert examples["physical_lineage_fingerprint"].tolist() == [
        expected[1],
        expected[1],
        expected[2],
    ]


def test_tampered_lineage_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _install_raw_data(tmp_path, monkeypatch)
    transitions_path = tmp_path / "transitions.csv"
    pd.DataFrame({"scenario_id": [1]}).to_csv(
        transitions_path,
        index=False,
    )
    annotate_transitions_csv(
        transitions_csv=transitions_path,
        raw_dir=raw_dir,
    )
    frame = pd.read_csv(transitions_path)
    frame.loc[0, "physical_lineage_fingerprint"] = "a" * 64
    frame.to_csv(transitions_path, index=False)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        annotate_transitions_csv(
            transitions_csv=transitions_path,
            raw_dir=raw_dir,
        )


def test_partial_lineage_columns_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _install_raw_data(tmp_path, monkeypatch)
    transitions_path = tmp_path / "transitions.csv"
    pd.DataFrame(
        {
            "scenario_id": [1],
            "base_case_id": ["incomplete"],
        }
    ).to_csv(transitions_path, index=False)

    with pytest.raises(ValueError, match="partial physical lineage"):
        annotate_transitions_csv(
            transitions_csv=transitions_path,
            raw_dir=raw_dir,
        )


def test_run_generate_propagates_lineage_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _install_raw_data(tmp_path, monkeypatch)
    transitions_path = tmp_path / "pool.csv"
    pd.DataFrame(
        {
            "scenario_id": [1, 2, 3],
            "difficulty_class": ["medium", "medium", "hard"],
        }
    ).to_csv(transitions_path, index=False)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    def fake_generate(request: SimpleNamespace) -> Path:
        selected = pd.read_csv(request.transitions_csv)
        assert set(LINEAGE_COLUMNS).issubset(selected.columns)
        output = Path(request.output_dir) / "examples.csv"
        pd.DataFrame(
            {
                "scenario_id": [1, 1, 2],
                "state_id": ["s1-0", "s1-1", "s2-0"],
            }
        ).to_csv(output, index=False)
        return output

    monkeypatch.setattr(stages, "GenerationRequest", fake_request)
    monkeypatch.setattr(
        stages,
        "generate_self_play_examples",
        fake_generate,
    )

    examples_path = stages.run_generate(
        project_root=tmp_path,
        raw_dir=raw_dir,
        transitions_csv=transitions_path,
        scenario_ids=[1, 2],
        checkpoint=checkpoint,
        output_dir=tmp_path / "output",
        config=GenerationConfig(device="cpu"),
        mcts_seed=1,
        action_seed=2,
        iteration=3,
    )

    examples = pd.read_csv(examples_path)
    selected = pd.read_csv(captured["transitions_csv"])
    assert set(LINEAGE_COLUMNS).issubset(examples.columns)
    assert set(LINEAGE_COLUMNS).issubset(selected.columns)
    assert set(selected["scenario_id"]) == {1, 2}


def test_replay_chunk_serialization_preserves_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _install_raw_data(tmp_path, monkeypatch)
    transitions_path = tmp_path / "transitions.csv"
    pd.DataFrame({"scenario_id": [1]}).to_csv(
        transitions_path,
        index=False,
    )
    annotate_transitions_csv(
        transitions_csv=transitions_path,
        raw_dir=raw_dir,
    )
    row = pd.read_csv(transitions_path).iloc[0].to_dict()
    chunk = tmp_path / "chunk.jsonl.gz"

    replay_module._write_jsonl_gz(
        header={"record_type": "replay_chunk_header"},
        rows=[row],
        path=chunk,
    )
    _, loaded = replay_module._read_jsonl_gz(chunk)

    for column in LINEAGE_COLUMNS:
        assert loaded[0][column] == row[column]

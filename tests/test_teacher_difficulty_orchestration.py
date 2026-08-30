from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai import cli
from grid_topology_ai import teacher_runtime
from grid_topology_ai.self_play import example_validation


def _write_profiles(path: Path) -> dict[str, dict[str, int]]:
    profiles = {
        "simple": {
            "depth": 2,
            "beam_width": 3,
            "candidate_pool": 20,
            "top_k": 10,
            "lodf_top_k": 9,
            "max_steps": 4,
            "max_teacher_steps": 2,
            "batch_size": 5,
        },
        "medium": {
            "depth": 4,
            "beam_width": 6,
            "candidate_pool": 40,
            "top_k": 20,
            "lodf_top_k": 18,
            "max_steps": 5,
            "max_teacher_steps": 4,
            "batch_size": 4,
        },
        "hard": {
            "depth": 6,
            "beam_width": 9,
            "candidate_pool": 60,
            "top_k": 30,
            "lodf_top_k": 27,
            "max_steps": 6,
            "max_teacher_steps": 6,
            "batch_size": 3,
        },
    }
    path.write_text(json.dumps(profiles), encoding="utf-8")
    return profiles


def _value(argv: list[str], option: str) -> str:
    index = argv.index(option)
    return argv[index + 1]


def test_load_scenario_ids_filters_before_limit(tmp_path):
    transitions = tmp_path / "transitions.csv"
    pd.DataFrame(
        {
            "scenario_id": [10, 10, 20, 30, 40],
            "difficulty_class": ["Simple", "simple", "medium", "hard", " simple "],
        }
    ).to_csv(transitions, index=False)

    assert teacher_runtime.load_scenario_ids(transitions, None, "simple") == [10, 40]
    assert teacher_runtime.load_scenario_ids(transitions, 1, "simple") == [10]
    assert teacher_runtime.load_scenario_ids(transitions, None, "medium") == [20]


def test_load_scenario_ids_requires_requested_difficulty(tmp_path):
    missing_column = tmp_path / "missing_column.csv"
    pd.DataFrame({"scenario_id": [1]}).to_csv(missing_column, index=False)

    with pytest.raises(ValueError, match="difficulty_class column"):
        teacher_runtime.load_scenario_ids(missing_column, None, "simple")

    missing_class = tmp_path / "missing_class.csv"
    pd.DataFrame(
        {"scenario_id": [1], "difficulty_class": ["simple"]}
    ).to_csv(missing_class, index=False)

    with pytest.raises(ValueError, match="No 'hard' scenarios"):
        teacher_runtime.load_scenario_ids(missing_class, None, "hard")


def test_legacy_profile_matches_historical_search_budgets():
    profiles = cli._load_teacher_profiles(Path("profiles/teacher_ieee118.json"))

    assert profiles["simple"] == {
        "depth": 4,
        "beam_width": 10,
        "candidate_pool": 60,
        "top_k": 30,
        "lodf_top_k": 30,
        "max_steps": 5,
        "max_teacher_steps": 5,
        "batch_size": 3,
    }
    assert profiles["medium"]["depth"] == 5
    assert profiles["medium"]["candidate_pool"] == 160
    assert profiles["hard"]["depth"] == 6
    assert profiles["hard"]["candidate_pool"] == 220
    assert profiles["hard"]["top_k"] == 100


def test_profile_validation_rejects_missing_class_and_invalid_value(tmp_path):
    missing_class = tmp_path / "missing.json"
    profiles = _write_profiles(missing_class)
    del profiles["hard"]
    missing_class.write_text(json.dumps(profiles), encoding="utf-8")

    with pytest.raises(ValueError, match="missing difficulty classes: hard"):
        cli._load_teacher_profiles(missing_class)

    invalid = tmp_path / "invalid.json"
    profiles = _write_profiles(invalid)
    profiles["medium"]["depth"] = 0
    invalid.write_text(json.dumps(profiles), encoding="utf-8")

    with pytest.raises(ValueError, match="medium.*depth.*positive integer"):
        cli._load_teacher_profiles(invalid)


def test_directory_teacher_runs_both_splits_by_difficulty_and_merges(
    tmp_path,
    monkeypatch,
):
    transitions_dir = tmp_path / "transitions"
    transitions_dir.mkdir()
    for name in ("train", "val", "self_play"):
        (transitions_dir / f"transitions_{name}.csv").write_text(
            "scenario_id,difficulty_class\n1,simple\n",
            encoding="utf-8",
        )

    profile_path = tmp_path / "profiles.json"
    profiles = _write_profiles(profile_path)
    output_dir = tmp_path / "teacher"
    calls: list[list[str]] = []

    monkeypatch.setattr(
        example_validation,
        "validate_examples_dataframe",
        lambda frame, *, source_path: None,
    )

    def fake_teacher_main(argv: list[str]) -> int:
        calls.append(list(argv))
        difficulty = _value(argv, "--difficulty-class")
        split = "train" if "transitions_train.csv" in _value(argv, "--transitions") else "val"
        class_index = ("simple", "medium", "hard").index(difficulty)
        scenario_id = class_index + (0 if split == "train" else 100)
        class_output = Path(_value(argv, "--output-dir"))
        class_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "scenario_id": [scenario_id],
                "step": [0],
                "state_id": [f"{split}_{difficulty}"],
            }
        ).to_csv(class_output / "examples.csv", index=False)
        return 0

    monkeypatch.setattr(teacher_runtime, "main", fake_teacher_main)

    result = cli.main(
        [
            "teacher",
            str(tmp_path / "raw"),
            "--transitions",
            str(transitions_dir),
            "--profiles",
            str(profile_path),
            "--output",
            str(output_dir),
            "--workers",
            "4",
            "--use-lodf-screening",
        ]
    )

    assert result == 0
    assert [
        (
            "train" if "transitions_train.csv" in _value(call, "--transitions") else "val",
            _value(call, "--difficulty-class"),
        )
        for call in calls
    ] == [
        ("train", "simple"),
        ("train", "medium"),
        ("train", "hard"),
        ("val", "simple"),
        ("val", "medium"),
        ("val", "hard"),
    ]
    assert all("self_play" not in _value(call, "--transitions") for call in calls)

    for call in calls:
        difficulty = _value(call, "--difficulty-class")
        profile = profiles[difficulty]
        assert int(_value(call, "--depth")) == profile["depth"]
        assert int(_value(call, "--beam-width")) == profile["beam_width"]
        assert int(_value(call, "--candidate-pool")) == profile["candidate_pool"]
        assert int(_value(call, "--top-k")) == profile["top_k"]
        assert int(_value(call, "--lodf-screen-top-k")) == profile["lodf_top_k"]
        assert int(_value(call, "--max-steps")) == profile["max_steps"]
        assert int(_value(call, "--max-teacher-steps")) == profile["max_teacher_steps"]
        assert int(_value(call, "--batch-size")) == profile["batch_size"]
        assert _value(call, "--num-workers") == "4"
        assert "--use-lodf-screening" in call

    train = pd.read_csv(output_dir / "examples_train.csv")
    validation = pd.read_csv(output_dir / "examples_val.csv")
    assert train["difficulty_class"].tolist() == ["simple", "medium", "hard"]
    assert validation["difficulty_class"].tolist() == ["simple", "medium", "hard"]
    assert set(train["teacher_split"]) == {"train"}
    assert set(validation["teacher_split"]) == {"val"}
    assert train["source_examples_csv"].str.endswith("examples.csv").all()
    assert validation["source_examples_csv"].str.endswith("examples.csv").all()
    assert (output_dir / "teacher_profile.json").read_bytes() == profile_path.read_bytes()


def test_directory_mode_requires_profile(tmp_path):
    transitions_dir = tmp_path / "transitions"
    transitions_dir.mkdir()
    (transitions_dir / "transitions_train.csv").write_text("scenario_id\n1\n")
    (transitions_dir / "transitions_val.csv").write_text("scenario_id\n2\n")

    args = cli.build_parser().parse_args(
        [
            "teacher",
            str(tmp_path / "raw"),
            "--transitions",
            str(transitions_dir),
            "--output",
            str(tmp_path / "teacher"),
        ]
    )

    with pytest.raises(ValueError, match="--profiles is required"):
        cli._teacher(args)


def test_single_csv_teacher_does_not_require_profile(tmp_path, monkeypatch):
    transitions = tmp_path / "transitions.csv"
    transitions.write_text("scenario_id\n1\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_teacher_main(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(teacher_runtime, "main", fake_teacher_main)

    result = cli.main(
        [
            "teacher",
            str(tmp_path / "raw"),
            "--transitions",
            str(transitions),
            "--output",
            str(tmp_path / "teacher"),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert "--difficulty-class" not in calls[0]

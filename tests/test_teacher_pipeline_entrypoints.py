from __future__ import annotations

import sys
from pathlib import Path

from scripts.pipelines import run_teacher_by_difficulty as pipeline


PROVENANCE_TEACHER = "scripts.self_play.generate_impact_teacher_provenance"


def test_teacher_pipeline_defaults_to_provenance_module(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_teacher_by_difficulty.py", "--dataset-name", "case118_test"],
    )

    args = pipeline.parse_args()

    assert args.teacher_module == PROVENANCE_TEACHER


def test_split_transition_runner_uses_provenance_module() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runner_path = (
        project_root
        / "scripts"
        / "pipelines"
        / "run_teacher_on_split_transitions.ps1"
    )
    text = runner_path.read_text(encoding="utf-8")

    assert "python -m scripts.self_play.generate_impact_teacher_provenance" in text
    assert "python -m scripts.self_play.generate_impact_teacher_parallel_fast" not in text

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


def test_teacher_pipeline_passes_canonical_pf_alg_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_teacher_by_difficulty.py", "--dataset-name", "case118_test"],
    )

    args = pipeline.parse_args()
    command = pipeline.build_teacher_command(
        python_executable="python",
        teacher_module=args.teacher_module,
        raw_dir=Path("raw"),
        transitions_path=Path("transitions.csv"),
        output_dir=Path("output"),
        profile=pipeline.DEFAULT_TEACHER_PROFILES["simple"],
        num_workers=args.num_workers,
        pf_alg=args.pf_alg,
        use_lodf=not args.disable_lodf,
        lodf_min_candidate_count=args.lodf_min_candidate_count,
        max_worker_memory_mb=args.max_worker_memory_mb,
        min_free_system_memory_mb=args.min_free_system_memory_mb,
        auto_worker_memory_mb=args.auto_worker_memory_mb,
        auto_worker_memory_reserve_mb=args.auto_worker_memory_reserve_mb,
        value_reward_scale=args.value_reward_scale,
        quiet_success=args.quiet_success,
    )

    pf_alg_index = command.index("--pf-alg")

    assert args.pf_alg == 1
    assert command[pf_alg_index + 1] == "1"


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

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pipelines import run_teacher_by_difficulty as pipeline


EXPECTED_PRODUCTION = {
    "simple": pipeline.TeacherProfile(
        depth=4,
        beam_width=10,
        candidate_pool=60,
        top_k=30,
        lodf_top_k=30,
        max_steps=5,
        max_teacher_steps=5,
        batch_size=3,
        auto_worker_max=10,
    ),
    "medium": pipeline.TeacherProfile(
        depth=5,
        beam_width=20,
        candidate_pool=160,
        top_k=70,
        lodf_top_k=70,
        max_steps=5,
        max_teacher_steps=5,
        batch_size=2,
        auto_worker_max=10,
    ),
    "hard": pipeline.TeacherProfile(
        depth=6,
        beam_width=30,
        candidate_pool=220,
        top_k=100,
        lodf_top_k=100,
        max_steps=6,
        max_teacher_steps=6,
        batch_size=2,
        auto_worker_max=10,
    ),
}


EXPECTED_SMOKE = {
    "simple": pipeline.TeacherProfile(
        depth=2,
        beam_width=4,
        candidate_pool=24,
        top_k=12,
        lodf_top_k=12,
        max_steps=2,
        max_teacher_steps=2,
        batch_size=1,
        auto_worker_max=3,
    ),
    "medium": pipeline.TeacherProfile(
        depth=3,
        beam_width=5,
        candidate_pool=36,
        top_k=18,
        lodf_top_k=18,
        max_steps=3,
        max_teacher_steps=3,
        batch_size=1,
        auto_worker_max=3,
    ),
    "hard": pipeline.TeacherProfile(
        depth=3,
        beam_width=6,
        candidate_pool=48,
        top_k=24,
        lodf_top_k=24,
        max_steps=3,
        max_teacher_steps=3,
        batch_size=1,
        auto_worker_max=3,
    ),
}


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_full_profile_preserves_production_search_budget() -> None:
    budget_name, profiles = pipeline.resolve_teacher_profiles("full")

    assert budget_name == "production"
    assert profiles == EXPECTED_PRODUCTION
    assert pipeline.DEFAULT_TEACHER_PROFILES == EXPECTED_PRODUCTION


def test_smoke_profile_has_its_own_bounded_search_budget() -> None:
    budget_name, profiles = pipeline.resolve_teacher_profiles("smoke")

    assert budget_name == "smoke"
    assert profiles == EXPECTED_SMOKE
    assert set(profiles) == set(pipeline.DIFFICULTIES)

    for difficulty in pipeline.DIFFICULTIES:
        smoke = profiles[difficulty]
        production = pipeline.DEFAULT_TEACHER_PROFILES[difficulty]

        assert smoke.depth < production.depth
        assert smoke.beam_width < production.beam_width
        assert smoke.candidate_pool < production.candidate_pool
        assert smoke.top_k < production.top_k
        assert smoke.lodf_top_k < production.lodf_top_k
        assert smoke.max_steps <= production.max_steps
        assert smoke.max_teacher_steps <= production.max_teacher_steps


def test_smoke_hard_still_exercises_multistep_and_lodf_paths() -> None:
    _, profiles = pipeline.resolve_teacher_profiles("smoke")
    hard = profiles["hard"]

    assert hard.depth >= 3
    assert hard.max_steps >= 3
    assert hard.max_teacher_steps >= 3
    assert hard.lodf_top_k >= 8


def test_standard_smoke_sample_can_use_multiple_worker_batches() -> None:
    _, profiles = pipeline.resolve_teacher_profiles("smoke")

    for profile in profiles.values():
        assert profile.batch_size == 1
        assert profile.auto_worker_max >= 2


def test_smoke_budget_is_forwarded_to_teacher_command() -> None:
    _, profiles = pipeline.resolve_teacher_profiles("smoke")
    profile = profiles["hard"]

    command = pipeline.build_teacher_command(
        python_executable="python",
        teacher_module="teacher.module",
        raw_dir=Path("raw"),
        transitions_path=Path("transitions.csv"),
        output_dir=Path("output"),
        profile=profile,
        num_workers="auto",
        use_lodf=True,
        lodf_min_candidate_count=8,
        max_worker_memory_mb=1000.0,
        min_free_system_memory_mb=512.0,
        auto_worker_memory_mb=1200.0,
        auto_worker_memory_reserve_mb=2048.0,
        value_reward_scale="auto",
        quiet_success=True,
    )

    assert _option_value(command, "--depth") == str(profile.depth)
    assert _option_value(command, "--beam-width") == str(profile.beam_width)
    assert _option_value(command, "--candidate-pool") == str(profile.candidate_pool)
    assert _option_value(command, "--top-k") == str(profile.top_k)
    assert _option_value(command, "--batch-size") == str(profile.batch_size)
    assert _option_value(command, "--auto-worker-max") == str(profile.auto_worker_max)
    assert _option_value(command, "--lodf-screen-top-k") == str(profile.lodf_top_k)


def test_unknown_pipeline_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported teacher pipeline profile"):
        pipeline.resolve_teacher_profiles("tiny")

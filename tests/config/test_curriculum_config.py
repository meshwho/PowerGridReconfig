from dataclasses import asdict
from pathlib import Path

import pytest

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.config.pool import (
    CurriculumSamplingConfig,
    PoolConfig,
)


_EXPECTED_DEFAULTS = {
    "never_solved_min_fraction": 0.20,
    "hard_min_fraction": 0.25,
    "simple_max_fraction": 0.35,
    "frontier_max_fraction": 0.50,
    "frontier_solve_rate_min": 0.25,
    "frontier_solve_rate_max": 0.75,
    "learning_progress_weight": 1.00,
    "uncertainty_weight": 0.75,
    "staleness_weight": 0.50,
    "frontier_weight": 0.35,
    "stale_after_iterations": 3,
    "priority_floor": 0.05,
}


def test_curriculum_sampling_defaults_are_stable() -> None:
    assert asdict(CurriculumSamplingConfig()) == _EXPECTED_DEFAULTS


def test_pool_config_without_curriculum_uses_defaults() -> None:
    config = PoolConfig.from_mapping(
        {
            "transitions_csv": "pool.csv",
            "raw_dir": "pool_raw",
            "metadata_path": "pool_metadata.json",
        }
    )

    assert asdict(config.curriculum) == _EXPECTED_DEFAULTS


@pytest.mark.parametrize(
    "path",
    [
        Path("configs/self_play_loop.yaml"),
        Path("configs/self_play_loop_pilot.yaml"),
        Path("configs/self_play_loop_smoke.yaml"),
    ],
)
def test_repository_configs_expose_curriculum_contract(path: Path) -> None:
    config = SelfPlayConfig.load(path)

    assert asdict(config.pool.curriculum) == _EXPECTED_DEFAULTS


def test_pool_config_reads_custom_curriculum_values() -> None:
    config = PoolConfig.from_mapping(
        {
            "transitions_csv": "pool.csv",
            "raw_dir": "pool_raw",
            "metadata_path": "pool_metadata.json",
            "curriculum": {
                "never_solved_min_fraction": 0.30,
                "hard_min_fraction": 0.40,
                "simple_max_fraction": 0.20,
                "frontier_max_fraction": 0.45,
                "frontier_solve_rate_min": 0.10,
                "frontier_solve_rate_max": 0.90,
                "learning_progress_weight": 1.20,
                "uncertainty_weight": 0.60,
                "staleness_weight": 0.80,
                "frontier_weight": 0.15,
                "stale_after_iterations": 7,
                "priority_floor": 0.02,
            },
        }
    )

    assert asdict(config.curriculum) == {
        "never_solved_min_fraction": 0.30,
        "hard_min_fraction": 0.40,
        "simple_max_fraction": 0.20,
        "frontier_max_fraction": 0.45,
        "frontier_solve_rate_min": 0.10,
        "frontier_solve_rate_max": 0.90,
        "learning_progress_weight": 1.20,
        "uncertainty_weight": 0.60,
        "staleness_weight": 0.80,
        "frontier_weight": 0.15,
        "stale_after_iterations": 7,
        "priority_floor": 0.02,
    }


@pytest.mark.parametrize(
    "field",
    [
        "never_solved_min_fraction",
        "hard_min_fraction",
        "simple_max_fraction",
        "frontier_max_fraction",
        "frontier_solve_rate_min",
        "frontier_solve_rate_max",
    ],
)
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_curriculum_rejects_values_outside_fraction_range(
    field: str,
    value: float,
) -> None:
    values = dict(_EXPECTED_DEFAULTS)
    values[field] = value

    with pytest.raises(ValueError, match=field):
        CurriculumSamplingConfig(**values)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(0.5, 0.5), (0.8, 0.2)],
)
def test_curriculum_rejects_invalid_frontier_interval(
    lower: float,
    upper: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="frontier_solve_rate_min",
    ):
        CurriculumSamplingConfig(
            frontier_solve_rate_min=lower,
            frontier_solve_rate_max=upper,
        )


@pytest.mark.parametrize(
    "field",
    [
        "learning_progress_weight",
        "uncertainty_weight",
        "staleness_weight",
        "frontier_weight",
    ],
)
@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_curriculum_rejects_invalid_weights(
    field: str,
    value: float,
) -> None:
    values = dict(_EXPECTED_DEFAULTS)
    values[field] = value

    with pytest.raises(ValueError, match=field):
        CurriculumSamplingConfig(**values)


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_curriculum_requires_exact_positive_staleness_threshold(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="stale_after_iterations"):
        CurriculumSamplingConfig.from_mapping(
            {"stale_after_iterations": value}
        )


@pytest.mark.parametrize(
    "value",
    [0.0, -0.01, float("inf"), float("nan")],
)
def test_curriculum_rejects_invalid_priority_floor(value: float) -> None:
    with pytest.raises(ValueError, match="priority_floor"):
        CurriculumSamplingConfig(priority_floor=value)

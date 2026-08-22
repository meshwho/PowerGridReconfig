import pytest

from grid_topology_ai.config import (
    DEFAULT_PHYSICS_CONFIG,
    EvaluationConfig,
    GenerationConfig,
    PhysicsConfig,
    TrainingConfig,
)


def test_light_scientific_defaults_are_stable() -> None:
    physics = PhysicsConfig()
    generation = GenerationConfig()
    evaluation = EvaluationConfig()
    training = TrainingConfig()

    assert physics == DEFAULT_PHYSICS_CONFIG
    assert (physics.pf_alg, physics.pf_tolerance, physics.max_iterations) == (
        1,
        1e-8,
        30,
    )
    assert (
        generation.simulations,
        generation.depth,
        generation.gamma,
        generation.use_root_noise,
        generation.use_continuation_gate,
    ) == (150, 4, 0.95, True, True)
    assert (
        evaluation.random_seed,
        evaluation.num_workers,
        evaluation.policy_mode,
    ) == (42, 1, "ungated")
    assert (
        training.batch_size,
        training.learning_rate,
        training.validation_fraction,
        training.num_workers,
    ) == (64, 3e-4, 0.20, 0)


def test_evaluation_policy_mode_is_the_single_configurable_policy_input() -> None:
    ungated = EvaluationConfig()
    constrained = EvaluationConfig(policy_mode="constrained")

    assert ungated.primary_policy_mode == "ungated"
    assert ungated.use_continuation_gate is False
    assert constrained.primary_policy_mode == "constrained"
    assert constrained.use_continuation_gate is True

    with pytest.raises(ValueError, match="policy_mode"):
        EvaluationConfig(policy_mode="legacy")
    with pytest.raises(TypeError):
        EvaluationConfig(primary_policy_mode="constrained")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        EvaluationConfig(use_continuation_gate=True)  # type: ignore[call-arg]


def test_light_configs_reject_invalid_scientific_ranges() -> None:
    with pytest.raises(ValueError, match="hard_overload_limit_percent"):
        PhysicsConfig(overload_limit_percent=110.0, hard_overload_limit_percent=100.0)
    with pytest.raises(ValueError, match="stop_policy"):
        GenerationConfig(stop_policy="legacy")
    with pytest.raises(ValueError, match="random_seed"):
        EvaluationConfig(random_seed=-1)
    with pytest.raises(ValueError, match="validation_fraction"):
        TrainingConfig(validation_fraction=1.0)

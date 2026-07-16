from pathlib import Path

import pytest

from grid_topology_ai.config import SelfPlayConfig


@pytest.mark.parametrize(
    "path",
    [
        Path("configs/self_play_loop.yaml"),
        Path("configs/self_play_loop_smoke.yaml"),
        Path("configs/self_play_loop_pilot.yaml"),
    ],
)
def test_repository_config_parses(path: Path) -> None:
    config = SelfPlayConfig.load(path)

    assert config.run_name
    assert config.n_iterations > 0
    assert config.n_scenarios_per_iteration > 0


def test_pilot_config_preserves_current_values() -> None:
    config = SelfPlayConfig.load(
        "configs/self_play_loop_pilot.yaml"
    )

    assert config.run_name == "self_play_pilot"
    assert config.n_iterations == 3
    assert config.training.epochs == 3
    assert config.training.examples_per_iteration == 128
    assert config.generation.simulations == 25
    assert config.generation.pf_alg == 3
    assert config.evaluation.simulations == 50
    assert config.replay_buffer.fresh_fraction == 0.70

def test_evaluation_config_defaults_to_pf_alg_3() -> None:
    from grid_topology_ai.config import EvaluationConfig

    assert EvaluationConfig().pf_alg == 3


def test_evaluation_config_reads_pf_alg() -> None:
    from grid_topology_ai.config import EvaluationConfig

    assert EvaluationConfig.from_mapping({"pf_alg": 2}).pf_alg == 2


def test_evaluation_config_rejects_unknown_pf_alg() -> None:
    from grid_topology_ai.config import EvaluationConfig
    import pytest

    with pytest.raises(ValueError, match="evaluation.pf_alg"):
        EvaluationConfig(pf_alg=9)


def test_self_play_config_rejects_generation_evaluation_pf_alg_mismatch() -> None:
    import pytest

    raw = {
        "run_name": "mismatch",
        "seed": 1,
        "n_iterations": 1,
        "n_scenarios_per_iteration": 1,
        "pool": {"transitions_csv": "pool.csv", "raw_dir": "raw", "metadata_path": "pool.json"},
        "eval_csv": "eval.csv",
        "eval_raw_dir": "eval_raw",
        "bootstrap_checkpoint": "bootstrap.pt",
        "bootstrap_eval_metrics": "metrics.json",
        "checkpoint_dir": "runs/mismatch",
        "best_checkpoint_path": "runs/mismatch/best.pt",
        "best_metrics_path": "runs/mismatch/best_metrics.json",
        "replay_buffer": {"max_size": 1, "min_size_to_train": 1, "fresh_fraction": 1.0},
        "generation": {"pf_alg": 3},
        "training": {"examples_per_iteration": 1, "batch_size": 1, "learning_rate": 0.001},
        "evaluation": {"pf_alg": 1},
        "acceptance": {},
    }
    with pytest.raises(ValueError, match="Power-flow algorithm mismatch"):
        SelfPlayConfig.from_mapping(raw)


def test_all_repository_self_play_configs_use_matching_pf_alg() -> None:
    for path in [
        "configs/self_play_loop.yaml",
        "configs/self_play_loop_pilot.yaml",
        "configs/self_play_loop_smoke.yaml",
    ]:
        config = SelfPlayConfig.load(path)
        assert config.generation.pf_alg == config.evaluation.pf_alg


def test_training_config_validation_defaults() -> None:
    from grid_topology_ai.config import TrainingConfig

    config = TrainingConfig()
    assert config.validation_fraction == 0.20
    assert config.min_validation_scenarios == 1


def test_training_config_reads_validation_contract() -> None:
    from grid_topology_ai.config import TrainingConfig

    config = TrainingConfig.from_mapping(
        {"validation_fraction": 0.3, "min_validation_scenarios": 2}
    )
    assert config.validation_fraction == 0.3
    assert config.min_validation_scenarios == 2


def test_training_config_rejects_invalid_validation_fraction() -> None:
    from grid_topology_ai.config import TrainingConfig
    import pytest

    for value in [0.0, 1.0, -0.1]:
        with pytest.raises(ValueError, match="validation_fraction"):
            TrainingConfig(validation_fraction=value)


def test_training_config_rejects_zero_min_validation_scenarios() -> None:
    from grid_topology_ai.config import TrainingConfig
    import pytest

    with pytest.raises(ValueError, match="min_validation_scenarios"):
        TrainingConfig(min_validation_scenarios=0)


def test_repository_self_play_configs_have_validation_contract() -> None:
    for path in [
        "configs/self_play_loop.yaml",
        "configs/self_play_loop_pilot.yaml",
        "configs/self_play_loop_smoke.yaml",
    ]:
        config = SelfPlayConfig.load(path)
        assert 0.0 < config.training.validation_fraction < 1.0
        assert config.training.min_validation_scenarios > 0

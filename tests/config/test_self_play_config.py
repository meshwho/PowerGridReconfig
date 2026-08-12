from pathlib import Path

import pytest

from grid_topology_ai.config import (
    CheckpointSelectionConfig,
    SelfPlayConfig,
)


def _minimal_self_play_mapping() -> dict[str, object]:
    return {
        "run_name": "test",
        "seed": 1,
        "n_iterations": 1,
        "n_scenarios_per_iteration": 1,
        "pool": {
            "transitions_csv": "pool.csv",
            "raw_dir": "pool_raw",
            "metadata_path": "pool.json",
        },
        "eval_csv": "eval.csv",
        "eval_raw_dir": "eval_raw",
        "final_test_csv": "final.csv",
        "final_test_raw_dir": "final_raw",
        "bootstrap_checkpoint": "bootstrap.pt",
        "bootstrap_eval_metrics": "metrics.json",
        "checkpoint_dir": "runs/test",
        "best_checkpoint_path": "runs/test/best.pt",
        "best_metrics_path": "runs/test/best_metrics.json",
        "replay_buffer": {
            "max_size": 1,
            "min_size_to_train": 1,
            "fresh_fraction": 1.0,
        },
        "generation": {"pf_alg": 3},
        "training": {
            "examples_per_iteration": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "save_multiple_best": True,
        },
        "evaluation": {"pf_alg": 3},
        "acceptance": {},
    }


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
    assert config.final_test_csv
    assert config.final_test_raw_dir
    assert config.final_test_csv != config.eval_csv
    assert config.final_test_raw_dir != config.eval_raw_dir


def test_pilot_config_preserves_current_values() -> None:
    config = SelfPlayConfig.load(
        "configs/self_play_loop_pilot.yaml"
    )

    assert config.run_name == "self_play_pilot"
    assert config.n_iterations == 3
    assert config.training.epochs == 3
    assert config.training.examples_per_iteration == 128
    assert config.training.save_multiple_best is True
    assert config.generation.simulations == 25
    assert config.generation.pf_alg == 1
    assert config.evaluation.simulations == 50
    assert config.replay_buffer.fresh_fraction == 0.70


def test_checkpoint_selection_defaults_to_disabled() -> None:
    config = SelfPlayConfig.from_mapping(
        _minimal_self_play_mapping()
    )

    assert config.checkpoint_selection.enabled is False
    assert config.checkpoint_selection.tuning_csv is None
    assert config.checkpoint_selection.tuning_raw_dir is None
    assert config.checkpoint_selection.max_candidates == 4
    assert config.checkpoint_selection.metric_direction == "maximize"


def test_checkpoint_selection_reads_tuning_contract() -> None:
    raw = _minimal_self_play_mapping()
    raw["checkpoint_selection"] = {
        "enabled": True,
        "tuning_csv": "tuning.csv",
        "tuning_raw_dir": "tuning_raw",
        "candidates_per_metric": 1,
        "max_candidates": 4,
        "calibration_bins": 12,
        "metric": "failed_scenario_rate_requested",
        "metric_direction": "MINIMIZE",
        "arena": {
            "simulations": 9,
            "pf_alg": 3,
            "depth": 2,
            "max_steps": 3,
            "top_k": 7,
        },
    }

    config = SelfPlayConfig.from_mapping(raw)
    selection = config.checkpoint_selection

    assert selection.enabled is True
    assert selection.tuning_csv == Path("tuning.csv")
    assert selection.tuning_raw_dir == Path("tuning_raw")
    assert selection.candidates_per_metric == 1
    assert selection.max_candidates == 4
    assert selection.calibration_bins == 12
    assert selection.metric == "failed_scenario_rate_requested"
    assert selection.metric_direction == "minimize"
    assert selection.arena.simulations == 9
    assert selection.arena.depth == 2


def test_enabled_checkpoint_selection_requires_tuning_paths() -> None:
    with pytest.raises(ValueError, match="requires tuning_csv"):
        CheckpointSelectionConfig(enabled=True)


def test_checkpoint_selection_paths_must_be_paired() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        CheckpointSelectionConfig(tuning_csv=Path("tuning.csv"))


def test_checkpoint_selection_rejects_multiple_candidates_per_metric() -> None:
    with pytest.raises(ValueError, match="must be 1"):
        CheckpointSelectionConfig(candidates_per_metric=2)


def test_checkpoint_selection_rejects_more_than_four_candidates() -> None:
    with pytest.raises(ValueError, match="must not exceed 4"):
        CheckpointSelectionConfig(max_candidates=5)


def test_checkpoint_selection_rejects_invalid_metric_direction() -> None:
    with pytest.raises(ValueError, match="metric_direction"):
        CheckpointSelectionConfig(metric_direction="sideways")


def test_checkpoint_selection_rejects_fractional_counts() -> None:
    with pytest.raises(ValueError, match="exact integer"):
        CheckpointSelectionConfig(max_candidates=2.5)


def test_enabled_checkpoint_selection_requires_multiple_best_training() -> None:
    raw = _minimal_self_play_mapping()
    training = dict(raw["training"])
    training["save_multiple_best"] = False
    raw["training"] = training
    raw["checkpoint_selection"] = {
        "enabled": True,
        "tuning_csv": "tuning.csv",
        "tuning_raw_dir": "tuning_raw",
    }

    with pytest.raises(ValueError, match="save_multiple_best=true"):
        SelfPlayConfig.from_mapping(raw)


def test_checkpoint_selection_arena_pf_alg_must_match_physics() -> None:
    raw = _minimal_self_play_mapping()
    raw["checkpoint_selection"] = {
        "enabled": True,
        "tuning_csv": "tuning.csv",
        "tuning_raw_dir": "tuning_raw",
        "arena": {"pf_alg": 2},
    }

    with pytest.raises(
        ValueError,
        match="checkpoint_selection.arena.pf_alg",
    ):
        SelfPlayConfig.from_mapping(raw)


def test_repository_configs_enable_checkpoint_tuning() -> None:
    for path in [
        "configs/self_play_loop.yaml",
        "configs/self_play_loop_pilot.yaml",
        "configs/self_play_loop_smoke.yaml",
    ]:
        config = SelfPlayConfig.load(path)
        selection = config.checkpoint_selection
        assert selection.enabled is True
        assert selection.tuning_csv is not None
        assert selection.tuning_raw_dir is not None
        assert selection.candidates_per_metric == 1
        assert selection.max_candidates == 4
        assert selection.metric_direction == "maximize"
        assert config.training.save_multiple_best is True
        assert selection.arena.pf_alg == config.physics.pf_alg


def test_final_test_csv_must_differ_from_eval_csv() -> None:
    raw = _minimal_self_play_mapping()
    raw["final_test_csv"] = raw["eval_csv"]

    with pytest.raises(
        ValueError,
        match="final_test_csv",
    ):
        SelfPlayConfig.from_mapping(raw)


def test_final_test_raw_dir_must_differ_from_eval_raw_dir() -> None:
    raw = _minimal_self_play_mapping()
    raw["final_test_raw_dir"] = raw["eval_raw_dir"]

    with pytest.raises(
        ValueError,
        match="final_test_raw_dir",
    ):
        SelfPlayConfig.from_mapping(raw)


def test_evaluation_config_defaults_to_pf_alg_1() -> None:
    from grid_topology_ai.config import EvaluationConfig

    assert EvaluationConfig().pf_alg == 1


def test_evaluation_config_reads_pf_alg() -> None:
    from grid_topology_ai.config import EvaluationConfig

    assert EvaluationConfig.from_mapping({"pf_alg": 2}).pf_alg == 2


def test_evaluation_config_rejects_unknown_pf_alg() -> None:
    from grid_topology_ai.config import EvaluationConfig

    with pytest.raises(ValueError, match="evaluation.pf_alg"):
        EvaluationConfig(pf_alg=9)


def test_self_play_config_rejects_generation_evaluation_pf_alg_mismatch() -> None:
    raw = _minimal_self_play_mapping()
    raw["evaluation"] = {"pf_alg": 1}

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

    for value in [0.0, 1.0, -0.1]:
        with pytest.raises(ValueError, match="validation_fraction"):
            TrainingConfig(validation_fraction=value)


def test_training_config_rejects_zero_min_validation_scenarios() -> None:
    from grid_topology_ai.config import TrainingConfig

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


@pytest.mark.parametrize("value", [3.0, "3"])
def test_pf_alg_exact_integer_values_are_accepted(value: object) -> None:
    from grid_topology_ai.config import EvaluationConfig, GenerationConfig

    assert GenerationConfig.from_mapping({"pf_alg": value}).pf_alg == 3
    assert EvaluationConfig.from_mapping({"pf_alg": value}).pf_alg == 3


def test_generation_config_rejects_fractional_pf_alg() -> None:
    from grid_topology_ai.config import GenerationConfig

    with pytest.raises(ValueError, match="exact integer"):
        GenerationConfig.from_mapping({"pf_alg": 3.5})


def test_evaluation_config_rejects_fractional_pf_alg() -> None:
    from grid_topology_ai.config import EvaluationConfig

    with pytest.raises(ValueError, match="exact integer"):
        EvaluationConfig.from_mapping({"pf_alg": 3.5})


def test_evaluation_config_rejects_out_of_choice_pf_alg_from_mapping() -> None:
    from grid_topology_ai.config import EvaluationConfig

    with pytest.raises(ValueError, match="evaluation.pf_alg"):
        EvaluationConfig.from_mapping({"pf_alg": 5})


@pytest.mark.parametrize("config_name", ["GenerationConfig", "EvaluationConfig"])
@pytest.mark.parametrize("value", [True, False, 3.5])
def test_pf_alg_direct_constructors_reject_bool_and_fractional_values(
    config_name: str,
    value: object,
) -> None:
    import grid_topology_ai.config as config_module

    config_cls = getattr(config_module, config_name)
    with pytest.raises(ValueError, match="exact integer"):
        config_cls(pf_alg=value)

from __future__ import annotations

from pathlib import Path

from grid_topology_ai.config import EvaluationConfig, GenerationConfig, SelfPlayConfig
from grid_topology_ai.evaluation import checkpoint as evaluation_checkpoint
from grid_topology_ai.evaluation.checkpoint import EvaluationRequest


def test_power_flow_algorithm_static_boundaries(tmp_path: Path, monkeypatch) -> None:
    assert GenerationConfig().pf_alg == 3
    assert EvaluationConfig().pf_alg == 3

    raw = {
        "run_name": "pf_boundary",
        "seed": 1,
        "n_iterations": 1,
        "n_scenarios_per_iteration": 1,
        "pool": {"transitions_csv": "pool.csv", "raw_dir": "raw", "metadata_path": "pool.json"},
        "eval_csv": "eval.csv",
        "eval_raw_dir": "eval_raw",
        "bootstrap_checkpoint": "bootstrap.pt",
        "bootstrap_eval_metrics": "metrics.json",
        "checkpoint_dir": "runs/pf_boundary",
        "best_checkpoint_path": "runs/pf_boundary/best.pt",
        "best_metrics_path": "runs/pf_boundary/best_metrics.json",
        "replay_buffer": {"max_size": 1, "min_size_to_train": 1, "fresh_fraction": 1.0},
        "generation": {"pf_alg": 3},
        "training": {"examples_per_iteration": 1, "batch_size": 1, "learning_rate": 0.001},
        "evaluation": {"pf_alg": 3},
        "acceptance": {},
    }
    config = SelfPlayConfig.from_mapping(raw)
    assert config.generation.pf_alg == config.evaluation.pf_alg == 3

    request = EvaluationRequest(
        raw_dir=tmp_path / "raw",
        transitions_csv=tmp_path / "transitions.csv",
        checkpoint=tmp_path / "checkpoint.pt",
        config=EvaluationConfig(pf_alg=3),
        pf_alg=None,
    )

    class FakeReward:
        def __init__(
            self,
            *,
            physics_config=None,
            discount_factor: float = 0.95,
        ) -> None:
            self.physics_config = physics_config
            self.discount_factor = float(discount_factor)

        def config_dict(self) -> dict[str, object]:
            return {"discount_factor": self.discount_factor}

    monkeypatch.setattr(evaluation_checkpoint, "GridFMReward", FakeReward)
    task_config = evaluation_checkpoint._make_task_config(request)
    assert task_config["pf_alg"] == 3
    assert task_config["reward_config"]["discount_factor"] == 0.95

    assert "pf_alg=1" not in Path("grid_topology_ai/self_play/stages.py").read_text()
    cli_source = Path("scripts/evaluation/evaluate_checkpoint.py").read_text()
    pf_alg_block = cli_source[cli_source.index('"--pf-alg"') : cli_source.index('"--disable-cache"')]
    assert "default=1" not in pf_alg_block
    assert "default=3" in pf_alg_block
    assert "resolved_pf_alg" in Path("grid_topology_ai/evaluation/checkpoint.py").read_text()
    assert '"pf_alg"' in Path("grid_topology_ai/self_play/generation.py").read_text()

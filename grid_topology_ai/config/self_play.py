from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from grid_topology_ai.config._mapping import (
    ConfigMapping,
    get_section,
    require_value,
)
from grid_topology_ai.config._validation import (
    require_positive,
)
from grid_topology_ai.config.acceptance import (
    AcceptanceConfig,
)
from grid_topology_ai.config.evaluation import (
    EvaluationConfig,
)
from grid_topology_ai.config.generation import (
    GenerationConfig,
)
from grid_topology_ai.config.pool import PoolConfig
from grid_topology_ai.config.replay import ReplayBufferConfig
from grid_topology_ai.config.training import TrainingConfig


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    save_config_copy_per_iteration: bool = True
    save_dataset_hash: bool = True
    save_config_hash: bool = True
    save_parent_checkpoint: bool = True
    save_reward_config: bool = True
    save_pool_metadata_hash: bool = True

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "MetadataConfig":
        return cls(
            save_config_copy_per_iteration=bool(
                data.get(
                    "save_config_copy_per_iteration",
                    True,
                )
            ),
            save_dataset_hash=bool(
                data.get("save_dataset_hash", True)
            ),
            save_config_hash=bool(
                data.get("save_config_hash", True)
            ),
            save_parent_checkpoint=bool(
                data.get("save_parent_checkpoint", True)
            ),
            save_reward_config=bool(
                data.get("save_reward_config", True)
            ),
            save_pool_metadata_hash=bool(
                data.get(
                    "save_pool_metadata_hash",
                    True,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    run_name: str
    seed: int
    n_iterations: int
    n_scenarios_per_iteration: int

    pool: PoolConfig

    eval_csv: Path
    eval_raw_dir: Path

    bootstrap_checkpoint: Path
    bootstrap_eval_metrics: Path

    checkpoint_dir: Path
    best_checkpoint_path: Path
    best_metrics_path: Path

    replay_buffer: ReplayBufferConfig
    generation: GenerationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    acceptance: AcceptanceConfig
    metadata: MetadataConfig

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must not be empty.")

        require_positive(
            "n_iterations",
            self.n_iterations,
        )
        require_positive(
            "n_scenarios_per_iteration",
            self.n_scenarios_per_iteration,
        )
        if int(self.generation.pf_alg) != int(self.evaluation.pf_alg):
            raise ValueError(
                "Power-flow algorithm mismatch: "
                f"generation.pf_alg={self.generation.pf_alg}, "
                f"evaluation.pf_alg={self.evaluation.pf_alg}. "
                "Self-play generation and fixed evaluation must use the same PF_ALG."
            )

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
    ) -> "SelfPlayConfig":
        epochs = int(data.get("epochs_per_iteration", 10))

        return cls(
            run_name=str(require_value(data, "run_name")),
            seed=int(data.get("seed", 42)),
            n_iterations=int(
                require_value(data, "n_iterations")
            ),
            n_scenarios_per_iteration=int(
                require_value(
                    data,
                    "n_scenarios_per_iteration",
                )
            ),
            pool=PoolConfig.from_mapping(
                get_section(data, "pool")
            ),
            eval_csv=Path(
                require_value(data, "eval_csv")
            ),
            eval_raw_dir=Path(
                require_value(data, "eval_raw_dir")
            ),
            bootstrap_checkpoint=Path(
                require_value(
                    data,
                    "bootstrap_checkpoint",
                )
            ),
            bootstrap_eval_metrics=Path(
                require_value(
                    data,
                    "bootstrap_eval_metrics",
                )
            ),
            checkpoint_dir=Path(
                require_value(data, "checkpoint_dir")
            ),
            best_checkpoint_path=Path(
                require_value(
                    data,
                    "best_checkpoint_path",
                )
            ),
            best_metrics_path=Path(
                require_value(
                    data,
                    "best_metrics_path",
                )
            ),
            replay_buffer=ReplayBufferConfig.from_mapping(
                get_section(data, "replay_buffer")
            ),
            generation=GenerationConfig.from_mapping(
                get_section(data, "generation")
            ),
            training=TrainingConfig.from_mapping(
                get_section(data, "training"),
                epochs=epochs,
            ),
            evaluation=EvaluationConfig.from_mapping(
                get_section(data, "evaluation")
            ),
            acceptance=AcceptanceConfig.from_mapping(
                get_section(data, "acceptance")
            ),
            metadata=MetadataConfig.from_mapping(
                get_section(
                    data,
                    "metadata",
                    required=False,
                )
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "SelfPlayConfig":
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            data: Any = yaml.safe_load(file)

        if not isinstance(data, Mapping):
            raise ValueError(
                f"Config must be a YAML mapping: {path}"
            )

        return cls.from_mapping(data)
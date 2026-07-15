from __future__ import annotations

from dataclasses import dataclass

from grid_topology_ai.config._mapping import ConfigMapping
from grid_topology_ai.config._validation import (
    require_choice,
    require_fraction,
    require_non_negative,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    examples_per_iteration: int | None = None

    batch_size: int = 64
    learning_rate: float = 3e-4
    value_loss_weight: float = 1.0
    value_huber_delta: float = 1.0

    num_workers: int = 0
    device: str = "auto"

    model_type: str = "graph_v2"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.0

    save_multiple_best: bool = False
    no_tensorboard: bool = True

    def __post_init__(self) -> None:
        require_positive("training.epochs", self.epochs)

        if self.examples_per_iteration is not None:
            require_positive(
                "training.examples_per_iteration",
                self.examples_per_iteration,
            )

        require_positive(
            "training.batch_size",
            self.batch_size,
        )
        require_positive(
            "training.learning_rate",
            self.learning_rate,
        )
        require_non_negative(
            "training.value_loss_weight",
            self.value_loss_weight,
        )
        require_positive(
            "training.value_huber_delta",
            self.value_huber_delta,
        )
        require_non_negative(
            "training.num_workers",
            self.num_workers,
        )
        require_choice(
            "training.device",
            self.device,
            {"auto", "cpu", "cuda"},
        )
        require_choice(
            "training.model_type",
            self.model_type,
            {"graph_v1", "graph_v2"},
        )
        require_positive(
            "training.hidden_dim",
            self.hidden_dim,
        )
        require_positive(
            "training.num_layers",
            self.num_layers,
        )
        require_fraction(
            "training.dropout",
            self.dropout,
        )

    @classmethod
    def from_mapping(
        cls,
        data: ConfigMapping,
        *,
        epochs: int = 10,
    ) -> "TrainingConfig":
        examples = data.get("examples_per_iteration")

        return cls(
            epochs=int(epochs),
            examples_per_iteration=(
                None if examples is None else int(examples)
            ),
            batch_size=int(data.get("batch_size", 64)),
            learning_rate=float(
                data.get("learning_rate", 3e-4)
            ),
            value_loss_weight=float(
                data.get("value_loss_weight", 1.0)
            ),
            value_huber_delta=float(
                data.get("value_huber_delta", 1.0)
            ),
            num_workers=int(data.get("num_workers", 0)),
            device=str(data.get("device", "auto")),
            model_type=str(
                data.get("model_type", "graph_v2")
            ),
            hidden_dim=int(data.get("hidden_dim", 128)),
            num_layers=int(data.get("num_layers", 3)),
            dropout=float(data.get("dropout", 0.0)),
            save_multiple_best=bool(
                data.get("save_multiple_best", False)
            ),
            no_tensorboard=bool(
                data.get("no_tensorboard", True)
            ),
        )
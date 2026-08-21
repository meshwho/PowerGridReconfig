from __future__ import annotations

import math
import random
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import require_physics_provenance
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.graph_self_play_dataset import (
    GraphSelfPlayDataset,
    collate_graph_samples,
)
from grid_topology_ai.training.checkpoints import (
    NORMALIZATION_STAT_KEYS,
    atomic_save_checkpoint,
    checkpoint_variant_path,
    extract_normalization_stats,
    load_checkpoint_payload,
    load_initial_checkpoint_into_model,
    make_checkpoint as _make_checkpoint,
    save_checkpoint_now,
)
from grid_topology_ai.training.metrics import (
    build_value_target_diagnostics,
    compute_policy_selection_score,
    log_epoch_metrics as _log_epoch_metrics,
    print_value_target_diagnostics,
    setup_live_logging,
)
from grid_topology_ai.training.validation_diagnostics import (
    attach_validation_metrics,
    evaluate_one_epoch as _evaluate_one_epoch_diagnostics,
    log_epoch_metrics as _log_epoch_diagnostics,
)
from grid_topology_ai.topology_actions import STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT


GraphModel = GraphPolicyValueNetV2


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    project_root: Path
    examples_csv: Path
    output_path: Path
    config: TrainingConfig

    init_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    validation_examples_csv: Path | None = None

    use_amp: bool = False
    normalize_features: bool = True
    save_best: bool = False

    tensorboard_log_dir: Path | None = None
    run_name: str | None = None
    metrics_csv: Path | None = None
    seed: int = 42
    physics_config: PhysicsConfig | None = None

_CANDIDATE_SELECTORS = (
    (
        "policy_loss",
        "best_policy_loss",
        "validation_policy_loss",
    ),
    (
        "value_loss",
        "best_value_loss",
        "validation_value_loss",
    ),
    (
        "value_calibration_error",
        "best_calibration",
        "validation_value_calibration_error",
    ),
)

@dataclass(slots=True)
class _CandidateTrackingState:
    request: TrainingRequest
    training_dataset: GraphSelfPlayDataset | None = None
    epoch: int = 0
    best_values: dict[str, float] = field(
        default_factory=lambda: {
            metric_name: float("inf")
            for metric_name, _, _ in _CANDIDATE_SELECTORS
        }
    )

_TRACKING_STATE: ContextVar[_CandidateTrackingState | None] = ContextVar(
    "checkpoint_candidate_tracking_state",
    default=None,
)

@contextmanager
def checkpoint_candidate_tracking(
    request: TrainingRequest,
) -> Iterator[None]:
    if not request.config.save_multiple_best:
        yield
        return

    token = _TRACKING_STATE.set(
        _CandidateTrackingState(request=request)
    )
    try:
        yield
    finally:
        _TRACKING_STATE.reset(token)

def register_training_dataset(
    dataset: GraphSelfPlayDataset,
) -> None:
    state = _TRACKING_STATE.get()
    if state is not None:
        state.training_dataset = dataset

def _candidate_normalization_metadata(
    request: TrainingRequest,
) -> dict[str, object]:
    from_init = request.init_checkpoint is not None
    return {
        "normalization_source": (
            "init_checkpoint"
            if from_init
            else "training_dataset"
        ),
        "normalization_frozen_from_init_checkpoint": from_init,
        "normalization_source_checkpoint": (
            None
            if request.init_checkpoint is None
            else str(request.init_checkpoint)
        ),
    }

def _finite_metric(
    metrics: Mapping[str, object],
    name: str,
) -> float | None:
    if name not in metrics:
        return None

    try:
        value = float(metrics[name])
    except (TypeError, ValueError, OverflowError):
        return None

    return value if math.isfinite(value) else None

def _save_candidate(
    *,
    path: Path,
    model: torch.nn.Module,
    training_dataset: GraphSelfPlayDataset,
    validation_dataset: GraphSelfPlayDataset,
    request: TrainingRequest,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    metric_name: str,
    selector_name: str,
    selector_value: float,
    val_metrics: Mapping[str, object],
) -> None:
    checkpoint = _make_checkpoint(
        model=model,
        dataset=training_dataset,
        request=request,
        device=device,
        use_amp=use_amp,
        normalization_metadata=(
            _candidate_normalization_metadata(request)
        ),
        validation_dataset=validation_dataset,
    )

    checkpoint["saved_epoch"] = int(epoch)
    checkpoint["selector_name"] = selector_name
    checkpoint["selector_value"] = float(
        selector_value
    )
    checkpoint["checkpoint_selection_metric"] = (
        metric_name
    )
    checkpoint["val_metrics"] = {
        key: float(value)
        for key, value in val_metrics.items()
        if isinstance(value, (int, float))
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    atomic_save_checkpoint(checkpoint, path)

def record_validation_candidates(
    *,
    model: torch.nn.Module,
    validation_dataset: GraphSelfPlayDataset,
    metrics: Mapping[str, object],
    device: torch.device,
    use_amp: bool,
) -> None:
    state = _TRACKING_STATE.get()

    if state is None:
        return

    if state.training_dataset is None:
        raise RuntimeError(
            "Checkpoint candidate tracking has no "
            "training dataset."
        )

    state.epoch += 1

    for (
        metric_key,
        variant_name,
        selection_metric,
    ) in _CANDIDATE_SELECTORS:
        value = _finite_metric(
            metrics,
            metric_key,
        )

        if (
            value is None
            or value
            >= state.best_values[metric_key]
        ):
            continue

        state.best_values[metric_key] = value

        _save_candidate(
            path=checkpoint_variant_path(
                state.request.output_path,
                variant_name,
            ),
            model=model,
            training_dataset=(
                state.training_dataset
            ),
            validation_dataset=(
                validation_dataset
            ),
            request=state.request,
            device=device,
            use_amp=use_amp,
            epoch=state.epoch,
            metric_name=selection_metric,
            selector_name=f"val_{metric_key}",
            selector_value=value,
            val_metrics=metrics,
        )

def resolve_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).lower().strip()

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --device cpu."
            )
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {device_arg}. Use one of: auto, cuda, cpu."
    )


def soft_policy_loss(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(target_policy * log_probs).sum(dim=1).mean()


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = (
            value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
    return moved


def _forward_graph_model(
    model: GraphModel,
    *,
    bus_features: torch.Tensor,
    branch_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_active_mask: torch.Tensor,
    action_mask: torch.Tensor,
    node_batch: torch.Tensor | None = None,
    edge_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=edge_index,
        edge_active_mask=edge_active_mask,
        action_mask=action_mask,
        node_batch=node_batch,
        edge_batch=edge_batch,
    )


def train_one_epoch(
    model: GraphModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    value_loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    value_loss_weight: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        bus_features = batch["bus_features"]
        branch_features = batch["branch_features"]
        edge_index = batch["edge_index"]
        edge_active_mask = batch["edge_active_mask"]
        node_batch = batch.get("node_batch")
        edge_batch = batch.get("edge_batch")
        action_mask = batch["action_mask"]
        target_policy = batch["target_policy"]
        target_value = batch["target_value"]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            policy_logits, predicted_value = _forward_graph_model(
                model,
                bus_features=bus_features,
                branch_features=branch_features,
                edge_index=edge_index,
                edge_active_mask=edge_active_mask,
                action_mask=action_mask,
                node_batch=node_batch,
                edge_batch=edge_batch,
            )
            policy_loss = soft_policy_loss(
                logits=policy_logits,
                target_policy=target_policy,
            )
            value_loss = value_loss_fn(predicted_value, target_value)
            total_loss = policy_loss + float(value_loss_weight) * value_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += float(total_loss.detach().item())
        policy_loss_sum += float(policy_loss.detach().item())
        value_loss_sum += float(value_loss.detach().item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Training loader produced zero batches.")

    return (
        total_loss_sum / batches,
        policy_loss_sum / batches,
        value_loss_sum / batches,
    )


def evaluate_one_epoch(
    model: GraphModel,
    loader: DataLoader,
    value_loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    value_loss_weight: float,
) -> dict[str, float]:
    metrics = _evaluate_one_epoch_diagnostics(
        model=model,
        loader=loader,
        value_loss_fn=value_loss_fn,
        device=device,
        use_amp=use_amp,
        value_loss_weight=value_loss_weight,
    )
    record_validation_candidates(
        model=model,
        validation_dataset=loader.dataset,
        metrics=metrics,
        device=device,
        use_amp=use_amp,
    )
    return metrics


def evaluate_training_samples(
    model: GraphModel,
    dataset: GraphSelfPlayDataset,
    device: torch.device,
    max_samples: int = 20,
) -> None:
    model.eval()
    n = min(len(dataset), int(max_samples))
    print("\nFinal predictions:")

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            batch = collate_graph_samples([sample])
            batch = move_batch_to_device(batch, device)

            bus_features = batch["bus_features"]
            branch_features = batch["branch_features"]
            edge_index = batch["edge_index"]
            edge_active_mask = batch["edge_active_mask"]
            node_batch = batch["node_batch"]
            edge_batch = batch["edge_batch"]
            action_mask = batch["action_mask"]
            target_policy = batch["target_policy"]
            target_value = float(sample["target_value"].item())

            logits, value = _forward_graph_model(
                model,
                bus_features=bus_features,
                branch_features=branch_features,
                edge_index=edge_index,
                edge_active_mask=edge_active_mask,
                action_mask=action_mask,
                node_batch=node_batch,
                edge_batch=edge_batch,
            )
            probabilities = torch.softmax(logits, dim=1)
            predicted_action = int(torch.argmax(probabilities, dim=1).item())
            target_top_action = int(torch.argmax(target_policy, dim=1).item())
            predicted_value = float(value.detach().cpu().item())
            predicted_prob = float(
                probabilities[0, predicted_action].detach().cpu().item()
            )
            target_prob = float(
                target_policy[0, target_top_action].detach().cpu().item()
            )

            print(
                f"Scenario {sample['scenario_id']:>5} | "
                f"step={sample['step']:>2} | "
                f"{sample['state_id']} | "
                f"target_top={target_top_action:>3} "
                f"(pi={target_prob:.3f}) | "
                f"pred_top={predicted_action:>3} "
                f"(p={predicted_prob:.3f}) | "
                f"value target={target_value:+.3f} | "
                f"value pred={predicted_value:+.3f}"
            )


def _normalize_scenario_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""

    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def collect_scenario_ids(dataset: GraphSelfPlayDataset) -> set[str]:
    if "scenario_id" not in dataset.examples.columns:
        raise ValueError("Dataset is missing required column: scenario_id")

    scenario_ids = {
        _normalize_scenario_id(value)
        for value in dataset.examples["scenario_id"].tolist()
    }
    scenario_ids.discard("")
    if not scenario_ids:
        raise ValueError("Dataset does not contain any valid scenario_id values.")
    return scenario_ids


def validate_no_scenario_overlap(
    train_dataset: GraphSelfPlayDataset,
    val_dataset: GraphSelfPlayDataset | None,
) -> None:
    if val_dataset is None:
        return

    train_scenario_ids = collect_scenario_ids(train_dataset)
    val_scenario_ids = collect_scenario_ids(val_dataset)
    overlap = train_scenario_ids & val_scenario_ids

    print(f"Train scenarios: {len(train_scenario_ids)}")
    print(f"Val scenarios:   {len(val_scenario_ids)}")

    if overlap:
        preview = sorted(overlap)[:20]
        raise ValueError(
            "Train/validation scenario leakage detected. "
            f"{len(overlap)} scenario_id values appear in both datasets. "
            f"Examples: {preview}. Use a scenario-level split, not a row-level split."
        )


def _build_model(
    *,
    request: TrainingRequest,
    dataset: GraphSelfPlayDataset,
    device: torch.device,
) -> GraphModel:
    register_training_dataset(dataset)

    if dataset.policy_layout != STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT:
        raise ValueError(
            "Graph policy-value networks require "
            "policy_layout='stop_plus_branch_status_v1'."
        )

    return GraphPolicyValueNetV2(
        num_bus_features=dataset.num_bus_features,
        num_branch_features=dataset.num_branch_features,
        hidden_dim=request.config.hidden_dim,
        num_layers=request.config.num_layers,
        dropout=request.config.dropout,
    ).to(device)


def _normalization_provenance(
    *,
    init_checkpoint: Path | None,
) -> dict[str, object]:
    from_init = init_checkpoint is not None
    return {
        "normalization_source": "init_checkpoint" if from_init else "training_dataset",
        "normalization_frozen_from_init_checkpoint": from_init,
        "normalization_source_checkpoint": (
            str(init_checkpoint) if init_checkpoint is not None else None
        ),
    }


def _assert_same_normalization_stats(
    *,
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    checkpoint_path: Path,
) -> None:
    for key in NORMALIZATION_STAT_KEYS:
        if not np.array_equal(actual[key], expected[key]):
            raise RuntimeError(
                "Fine-tuning dataset normalization differs from init checkpoint "
                f"for {key}. Checkpoint: {checkpoint_path}"
            )


def _validate_normalization_feature_dimensions(
    *,
    normalization_stats: dict[str, np.ndarray],
    dataset: GraphSelfPlayDataset,
    checkpoint_path: Path,
) -> None:
    checks = {
        "bus_feature_mean": int(dataset.num_bus_features),
        "bus_feature_std": int(dataset.num_bus_features),
        "branch_feature_mean": int(dataset.num_branch_features),
        "branch_feature_std": int(dataset.num_branch_features),
    }
    for key, expected in checks.items():
        observed = normalization_stats[key].shape
        if observed != (expected,):
            raise ValueError(
                f"Initial checkpoint normalization dimension mismatch for {key}. "
                f"Expected dimension {expected}, observed shape {observed}. "
                f"Checkpoint: {checkpoint_path}"
            )


def make_checkpoint(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return attach_validation_metrics(_make_checkpoint(*args, **kwargs))


def log_epoch_metrics(**kwargs: Any) -> None:
    _log_epoch_diagnostics(_log_epoch_metrics, **kwargs)


def train_graph_policy_value_model(request: TrainingRequest) -> Path:
    with checkpoint_candidate_tracking(request):
        if request.init_checkpoint is not None and request.resume_checkpoint is not None:
            raise ValueError("init_checkpoint and resume_checkpoint are mutually exclusive.")
        if request.init_checkpoint is not None and not request.normalize_features:
            raise ValueError(
                "Fine-tuning from an initial checkpoint requires "
                "normalize_features=True because checkpoint weights and "
                "normalization statistics form one model contract."
            )

        if not request.examples_csv.exists():
            raise FileNotFoundError(
                f"Examples CSV not found: {request.examples_csv}"
            )

        seed = int(request.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = resolve_device(request.config.device)
        use_amp = bool(request.use_amp and device.type == "cuda")
        output_path = request.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("=" * 100)
        print("Training graph/GNN policy-value baseline")
        print("=" * 100)
        print(f"Examples CSV:  {request.examples_csv}")
        print(f"Device:        {device}")
        print(f"CUDA available:{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device:   {torch.cuda.get_device_name(0)}")
            print(f"CUDA version:  {torch.version.cuda}")
        print(f"AMP enabled:   {use_amp}")

        init_checkpoint_payload = None
        checkpoint_normalization_stats = None
        source_checkpoint = request.resume_checkpoint or request.init_checkpoint
        if source_checkpoint is not None:
            if not source_checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {source_checkpoint}"
                )
            init_checkpoint_payload = load_checkpoint_payload(
                source_checkpoint,
                map_location="cpu",
                expected_physics_config=request.physics_config,
            )
            checkpoint_normalization_stats = extract_normalization_stats(
                init_checkpoint_payload,
                source=source_checkpoint,
            )

        dataset = GraphSelfPlayDataset(
            examples_csv=request.examples_csv,
            normalize_features=request.normalize_features,
            normalization_stats=checkpoint_normalization_stats,
            physics_config=request.physics_config,
        )
        if init_checkpoint_payload is not None:
            require_physics_provenance(
                init_checkpoint_payload,
                source=str(request.init_checkpoint),
                expected_physics_config=dataset.physics_config,
            )
        effective_normalization_stats = dataset.normalization_state_dict()

        if checkpoint_normalization_stats is not None:
            _assert_same_normalization_stats(
                actual=effective_normalization_stats,
                expected=checkpoint_normalization_stats,
                checkpoint_path=request.init_checkpoint,
            )
            _validate_normalization_feature_dimensions(
                normalization_stats=effective_normalization_stats,
                dataset=dataset,
                checkpoint_path=request.init_checkpoint,
            )

        val_dataset = None
        if request.validation_examples_csv is not None:
            val_dataset = GraphSelfPlayDataset(
                examples_csv=request.validation_examples_csv,
                normalize_features=request.normalize_features,
                normalization_stats=effective_normalization_stats,
                physics_config=dataset.physics_config,
            )

        if val_dataset is not None:
            if val_dataset.policy_layout != dataset.policy_layout:
                raise ValueError(
                    "Training and validation policy layouts do not match."
                )
            if val_dataset.topology_action_config != dataset.topology_action_config:
                raise ValueError(
                    "Training and validation topology action configs do not match."
                )
            if val_dataset.num_bus_features != dataset.num_bus_features:
                raise ValueError(
                    "Training and validation bus feature dimensions do not match."
                )
            if val_dataset.num_branch_features != dataset.num_branch_features:
                raise ValueError(
                    "Training and validation branch feature dimensions do not match."
                )

        validate_no_scenario_overlap(train_dataset=dataset, val_dataset=val_dataset)

        print(f"Examples:      {len(dataset)}")
        print(f"Action layouts:     {dataset.action_layout_count}")
        print(f"Bus features:  {dataset.num_bus_features}")
        print(f"Branch feats:  {dataset.num_branch_features}")

        train_value_diagnostics = build_value_target_diagnostics(dataset=dataset)
        print_value_target_diagnostics(train_value_diagnostics)

        print(f"Batch size:    {request.config.batch_size}")
        print(f"Num workers:   {request.config.num_workers}")
        print(f"Hidden dim:    {request.config.hidden_dim}")
        print(f"Num layers:    {request.config.num_layers}")
        print(f"Dropout:       {request.config.dropout}")
        print("Model type:    graph_v2")
        print(
            f"Value loss:    HuberLoss(delta={request.config.value_huber_delta})"
        )
        if val_dataset is not None:
            print(f"Val examples:   {len(val_dataset)}")
            print(f"Val CSV:        {request.validation_examples_csv}")

        writer, metrics_csv_path = setup_live_logging(
            request=request,
            output_path=output_path,
        )

        pin_memory = device.type == "cuda"
        train_generator = torch.Generator()
        train_generator.manual_seed(int(request.seed))
        collate_fn = collate_graph_samples

        loader = DataLoader(
            dataset,
            batch_size=min(request.config.batch_size, len(dataset)),
            shuffle=True,
            num_workers=int(request.config.num_workers),
            pin_memory=pin_memory,
            generator=train_generator,
            collate_fn=collate_fn,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(request.config.batch_size, len(val_dataset)),
                shuffle=False,
                num_workers=int(request.config.num_workers),
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )

        model = _build_model(request=request, dataset=dataset, device=device)

        if source_checkpoint is not None:
            load_initial_checkpoint_into_model(
                model=model,
                checkpoint_path=source_checkpoint,
                dataset=dataset,
                hidden_dim=request.config.hidden_dim,
                num_layers=request.config.num_layers,
                dropout=request.config.dropout,
                device=device,
                checkpoint_payload=init_checkpoint_payload,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=request.config.learning_rate,
            weight_decay=1e-4,
        )
        value_loss_fn = nn.HuberLoss(
            delta=float(request.config.value_huber_delta)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_metric = float("inf")
        best_epoch = 0
        best_checkpoint = None
        best_top1 = -float("inf")
        best_top1_epoch = 0
        best_top5 = -float("inf")
        best_top5_epoch = 0
        best_switch = -float("inf")
        best_switch_epoch = 0
        best_policy_score = -float("inf")
        best_policy_score_epoch = 0

        start_epoch = 1
        if request.resume_checkpoint is not None:
            resume = init_checkpoint_payload
            assert resume is not None
            saved_config = resume.get("training_config")
            if not isinstance(saved_config, Mapping):
                raise ValueError(
                    f"Resume checkpoint is missing training_config: {request.resume_checkpoint}"
                )
            current_config = {
                "seed": int(request.seed),
                "lr": float(request.config.learning_rate),
                "batch_size": int(request.config.batch_size),
                "value_loss_weight": float(request.config.value_loss_weight),
                "value_huber_delta": float(request.config.value_huber_delta),
                "amp": bool(request.use_amp),
                "num_workers": int(request.config.num_workers),
                "no_normalize_features": bool(not request.normalize_features),
            }
            for key, expected in current_config.items():
                if saved_config.get(key) != expected:
                    raise ValueError(
                        f"Resume training configuration mismatch for {key}: "
                        f"expected {expected!r}, checkpoint has {saved_config.get(key)!r}."
                    )
            for key in ("optimizer_state_dict", "scaler_state_dict", "completed_epoch", "rng_state", "train_generator_state"):
                if key not in resume:
                    raise ValueError(f"Resume checkpoint is missing {key!r}: {request.resume_checkpoint}")
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scaler.load_state_dict(resume["scaler_state_dict"])
            train_generator.set_state(resume["train_generator_state"])
            rng_state = resume["rng_state"]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if device.type == "cuda" and "cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            start_epoch = int(resume["completed_epoch"]) + 1
            best_metric = float(resume.get("best_metric", best_metric))
            best_epoch = int(resume.get("best_epoch", best_epoch))

        for epoch in range(start_epoch, request.config.epochs + 1):
            total_loss, policy_loss, value_loss = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                value_loss_fn=value_loss_fn,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                value_loss_weight=request.config.value_loss_weight,
            )

            val_metrics = None
            if val_loader is not None:
                val_metrics = evaluate_one_epoch(
                    model=model,
                    loader=val_loader,
                    value_loss_fn=value_loss_fn,
                    device=device,
                    use_amp=use_amp,
                    value_loss_weight=request.config.value_loss_weight,
                )
                current_metric = float(val_metrics["loss"])

                if current_metric < best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_checkpoint = make_checkpoint(
                        model=model,
                        dataset=dataset,
                        request=request,
                        device=device,
                        use_amp=use_amp,
                        normalization_metadata=_normalization_provenance(
                            init_checkpoint=request.init_checkpoint
                        ),
                        validation_dataset=val_dataset,
                    )

                if request.config.save_multiple_best:
                    current_top1 = float(val_metrics["top1"])
                    current_top5 = float(val_metrics["top5"])
                    current_switch = float(val_metrics["switch_acc"])
                    current_policy_score = compute_policy_selection_score(
                        val_metrics
                    )

                    if current_metric <= best_metric:
                        save_checkpoint_now(
                            path=checkpoint_variant_path(
                                output_path, "best_loss"
                            ),
                            model=model,
                            dataset=dataset,
                            request=request,
                            device=device,
                            use_amp=use_amp,
                            epoch=epoch,
                            selector_name="val_loss",
                            selector_value=current_metric,
                            val_metrics=val_metrics,
                            normalization_metadata=_normalization_provenance(
                                init_checkpoint=request.init_checkpoint
                            ),
                            validation_dataset=val_dataset,
                        )

                    if current_top1 > best_top1:
                        best_top1 = current_top1
                        best_top1_epoch = epoch
                        save_checkpoint_now(
                            path=checkpoint_variant_path(
                                output_path, "best_top1"
                            ),
                            model=model,
                            dataset=dataset,
                            request=request,
                            device=device,
                            use_amp=use_amp,
                            epoch=epoch,
                            selector_name="val_top1",
                            selector_value=current_top1,
                            val_metrics=val_metrics,
                            normalization_metadata=_normalization_provenance(
                                init_checkpoint=request.init_checkpoint
                            ),
                            validation_dataset=val_dataset,
                        )

                    if current_top5 > best_top5:
                        best_top5 = current_top5
                        best_top5_epoch = epoch
                        save_checkpoint_now(
                            path=checkpoint_variant_path(
                                output_path, "best_top5"
                            ),
                            model=model,
                            dataset=dataset,
                            request=request,
                            device=device,
                            use_amp=use_amp,
                            epoch=epoch,
                            selector_name="val_top5",
                            selector_value=current_top5,
                            val_metrics=val_metrics,
                            normalization_metadata=_normalization_provenance(
                                init_checkpoint=request.init_checkpoint
                            ),
                            validation_dataset=val_dataset,
                        )

                    if current_switch > best_switch:
                        best_switch = current_switch
                        best_switch_epoch = epoch
                        save_checkpoint_now(
                            path=checkpoint_variant_path(
                                output_path, "best_switch"
                            ),
                            model=model,
                            dataset=dataset,
                            request=request,
                            device=device,
                            use_amp=use_amp,
                            epoch=epoch,
                            selector_name="val_switch",
                            selector_value=current_switch,
                            val_metrics=val_metrics,
                            normalization_metadata=_normalization_provenance(
                                init_checkpoint=request.init_checkpoint
                            ),
                            validation_dataset=val_dataset,
                        )

                    if current_policy_score > best_policy_score:
                        best_policy_score = current_policy_score
                        best_policy_score_epoch = epoch
                        save_checkpoint_now(
                            path=checkpoint_variant_path(
                                output_path, "best_policy"
                            ),
                            model=model,
                            dataset=dataset,
                            request=request,
                            device=device,
                            use_amp=use_amp,
                            epoch=epoch,
                            selector_name="policy_selection_score",
                            selector_value=current_policy_score,
                            val_metrics=val_metrics,
                            normalization_metadata=_normalization_provenance(
                                init_checkpoint=request.init_checkpoint
                            ),
                            validation_dataset=val_dataset,
                        )

                print(
                    f"Epoch {epoch:4d} | "
                    f"train_loss={total_loss:.6f} | "
                    f"train_policy={policy_loss:.6f} | "
                    f"train_value={value_loss:.6f} | "
                    f"val_loss={val_metrics['loss']:.6f} | "
                    f"val_policy={val_metrics['policy_loss']:.6f} | "
                    f"val_value={val_metrics['value_loss']:.6f} | "
                    f"val_top1={val_metrics['top1']:.4f} | "
                    f"val_top5={val_metrics['top5']:.4f} | "
                    f"val_stop={val_metrics['stop_acc']:.4f} | "
                    f"val_switch={val_metrics['switch_acc']:.4f} | "
                    f"best_epoch={best_epoch}"
                )
            else:
                current_metric = total_loss
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_checkpoint = make_checkpoint(
                        model=model,
                        dataset=dataset,
                        request=request,
                        device=device,
                        use_amp=use_amp,
                        normalization_metadata=_normalization_provenance(
                            init_checkpoint=request.init_checkpoint
                        ),
                        validation_dataset=val_dataset,
                    )

                if (
                    epoch == 1
                    or epoch % 25 == 0
                    or epoch == request.config.epochs
                ):
                    print(
                        f"Epoch {epoch:4d} | "
                        f"loss={total_loss:.6f} | "
                        f"policy_loss={policy_loss:.6f} | "
                        f"value_loss={value_loss:.6f} | "
                        f"best={best_metric:.6f} | "
                        f"best_epoch={best_epoch}"
                    )

            learning_rate = float(optimizer.param_groups[0]["lr"])
            log_epoch_metrics(
                tensorboard_writer=writer,
                metrics_csv_path=metrics_csv_path,
                epoch=epoch,
                train_loss=total_loss,
                train_policy=policy_loss,
                train_value=value_loss,
                val_metrics=val_metrics,
                best_epoch=best_epoch,
                best_metric=best_metric,
                learning_rate=learning_rate,
            )

            progress_checkpoint = (
                request.resume_checkpoint
                if request.resume_checkpoint is not None
                else checkpoint_variant_path(output_path, "resume")
            )
            resume_payload = make_checkpoint(
                model=model,
                dataset=dataset,
                request=request,
                device=device,
                use_amp=use_amp,
                normalization_metadata=_normalization_provenance(
                    init_checkpoint=request.init_checkpoint
                ),
                validation_dataset=val_dataset,
            )
            resume_payload.update({
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "completed_epoch": int(epoch),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "train_generator_state": train_generator.get_state(),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    **({"cuda": torch.cuda.get_rng_state_all()} if device.type == "cuda" else {}),
                },
            })
            atomic_save_checkpoint(resume_payload, progress_checkpoint)

        if request.save_best and best_checkpoint is not None:
            checkpoint = best_checkpoint
            checkpoint["best_epoch"] = int(best_epoch)
            checkpoint["best_metric"] = float(best_metric)
        else:
            checkpoint = make_checkpoint(
                model=model,
                dataset=dataset,
                request=request,
                device=device,
                use_amp=use_amp,
                normalization_metadata=_normalization_provenance(
                    init_checkpoint=request.init_checkpoint
                ),
                validation_dataset=val_dataset,
            )
            checkpoint["best_epoch"] = int(best_epoch)
            checkpoint["best_metric"] = float(best_metric)

        atomic_save_checkpoint(checkpoint, output_path)

        if request.save_best and checkpoint is best_checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        if request.config.save_multiple_best:
            last_checkpoint_path = checkpoint_variant_path(output_path, "last")
            last_checkpoint = make_checkpoint(
                model=model,
                dataset=dataset,
                request=request,
                device=device,
                use_amp=use_amp,
                normalization_metadata=_normalization_provenance(
                    init_checkpoint=request.init_checkpoint
                ),
                validation_dataset=val_dataset,
            )
            last_checkpoint["saved_epoch"] = int(request.config.epochs)
            last_checkpoint["selector_name"] = "last_epoch"
            last_checkpoint["selector_value"] = float(request.config.epochs)
            last_checkpoint["checkpoint_selection_metric"] = "last_epoch"
            atomic_save_checkpoint(last_checkpoint, last_checkpoint_path)

            print("\nSaved additional checkpoint variants:")
            print(checkpoint_variant_path(output_path, "best_loss"))
            print(checkpoint_variant_path(output_path, "best_top1"))
            print(checkpoint_variant_path(output_path, "best_top5"))
            print(checkpoint_variant_path(output_path, "best_switch"))
            print(checkpoint_variant_path(output_path, "best_policy"))
            print(last_checkpoint_path)

            print("\nBest selector epochs:")
            print(f"  best_loss epoch:   {best_epoch}")
            print(f"  best_top1 epoch:   {best_top1_epoch}")
            print(f"  best_top5 epoch:   {best_top5_epoch}")
            print(f"  best_switch epoch: {best_switch_epoch}")
            print(f"  best_policy epoch: {best_policy_score_epoch}")

        if writer is not None:
            writer.close()

        print("\nSaved graph model:")
        print(output_path)
        print(f"Best epoch:  {best_epoch}")
        print(f"Best metric: {best_metric:.6f}")

        evaluate_training_samples(
            model=model,
            dataset=dataset,
            device=device,
            max_samples=20,
        )

        print("\nDone.")
        return output_path


__all__ = [
    "Any",
    "DataLoader",
    "GraphModel",
    "GraphPolicyValueNetV2",
    "GraphSelfPlayDataset",
    "NORMALIZATION_STAT_KEYS",
    "Path",
    "PhysicsConfig",
    "STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT",
    "TrainingConfig",
    "TrainingRequest",
    "build_value_target_diagnostics",
    "checkpoint_variant_path",
    "collate_graph_samples",
    "collect_scenario_ids",
    "compute_policy_selection_score",
    "dataclass",
    "evaluate_one_epoch",
    "evaluate_training_samples",
    "extract_normalization_stats",
    "load_checkpoint_payload",
    "load_initial_checkpoint_into_model",
    "log_epoch_metrics",
    "make_checkpoint",
    "move_batch_to_device",
    "nn",
    "np",
    "print_value_target_diagnostics",
    "random",
    "require_physics_provenance",
    "resolve_device",
    "save_checkpoint_now",
    "setup_live_logging",
    "soft_policy_loss",
    "torch",
    "train_graph_policy_value_model",
    "train_one_epoch",
    "validate_no_scenario_overlap",
]

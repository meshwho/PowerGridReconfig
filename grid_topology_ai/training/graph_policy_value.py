from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.models.graph_policy_value_net import GraphPolicyValueNet
from grid_topology_ai.models.graph_policy_value_net_v2 import GraphPolicyValueNetV2
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.training.checkpoints import (
    NORMALIZATION_STAT_KEYS,
    checkpoint_variant_path,
    extract_normalization_stats,
    load_checkpoint_payload,
    load_initial_checkpoint_into_model,
    make_checkpoint,
    save_checkpoint_now,
)
from grid_topology_ai.training.metrics import (
    build_value_target_diagnostics,
    compute_policy_selection_score,
    log_epoch_metrics,
    print_value_target_diagnostics,
    setup_live_logging,
)


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    project_root: Path
    examples_csv: Path
    output_path: Path
    config: TrainingConfig

    init_checkpoint: Path | None = None
    validation_examples_csv: Path | None = None

    use_amp: bool = False
    normalize_features: bool = True
    save_best: bool = False

    tensorboard_log_dir: Path | None = None
    run_name: str | None = None
    metrics_csv: Path | None = None
    seed: int = 42


def resolve_device(device_arg: str) -> torch.device:
    """
    Resolve requested training device.
    """

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
        f"Unsupported device: {device_arg}. "
        "Use one of: auto, cuda, cpu."
    )


def soft_policy_loss(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy with soft policy target.
    """

    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(target_policy * log_probs).sum(dim=1).mean()

    return loss


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    Move tensor batch fields to selected device.
    """

    moved: dict[str, Any] = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def train_one_epoch(
    model: GraphPolicyValueNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    value_loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    value_loss_weight: float,
) -> tuple[float, float, float]:
    """
    Train one epoch.
    """

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
        action_mask = batch["action_mask"]
        target_policy = batch["target_policy"]
        target_value = batch["target_value"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            policy_logits, predicted_value = model(
                bus_features=bus_features,
                branch_features=branch_features,
                edge_index=edge_index,
                action_mask=action_mask,
            )

            policy_loss = soft_policy_loss(
                logits=policy_logits,
                target_policy=target_policy,
            )

            value_loss = value_loss_fn(
                predicted_value,
                target_value,
            )

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
    model: GraphPolicyValueNet,
    loader: DataLoader,
    value_loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    value_loss_weight: float,
) -> dict[str, float]:
    """
    Evaluate graph model on validation data.
    """

    model.eval()

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    total_examples = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    stop_total = 0
    stop_correct = 0

    switch_total = 0
    switch_correct = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)

            bus_features = batch["bus_features"]
            branch_features = batch["branch_features"]
            edge_index = batch["edge_index"]
            action_mask = batch["action_mask"]
            target_policy = batch["target_policy"]
            target_value = batch["target_value"]

            batch_size = int(bus_features.shape[0])

            with torch.amp.autocast("cuda", enabled=use_amp):
                policy_logits, predicted_value = model(
                    bus_features=bus_features,
                    branch_features=branch_features,
                    edge_index=edge_index,
                    action_mask=action_mask,
                )

                policy_loss = soft_policy_loss(
                    logits=policy_logits,
                    target_policy=target_policy,
                )

                value_loss = value_loss_fn(
                    predicted_value,
                    target_value,
                )

                total_loss = policy_loss + float(value_loss_weight) * value_loss

            total_loss_sum += float(total_loss.detach().item()) * batch_size
            policy_loss_sum += float(policy_loss.detach().item()) * batch_size
            value_loss_sum += float(value_loss.detach().item()) * batch_size

            target_action = torch.argmax(target_policy, dim=1)
            predicted_top = torch.argmax(policy_logits, dim=1)

            topk = torch.topk(
                policy_logits,
                k=min(5, policy_logits.shape[1]),
                dim=1,
            ).indices

            top1_correct += int((predicted_top == target_action).sum().item())

            top3 = topk[:, : min(3, topk.shape[1])]
            top5 = topk[:, : min(5, topk.shape[1])]

            top3_correct += int(
                (top3 == target_action.unsqueeze(1)).any(dim=1).sum().item()
            )

            top5_correct += int(
                (top5 == target_action.unsqueeze(1)).any(dim=1).sum().item()
            )

            stop_mask = target_action == 0
            switch_mask = target_action != 0

            stop_total += int(stop_mask.sum().item())
            switch_total += int(switch_mask.sum().item())

            if stop_mask.any():
                stop_correct += int(
                    (predicted_top[stop_mask] == 0).sum().item()
                )

            if switch_mask.any():
                switch_correct += int(
                    (
                        predicted_top[switch_mask]
                        == target_action[switch_mask]
                    ).sum().item()
                )

            total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("Validation loader produced zero examples.")

    return {
        "loss": total_loss_sum / total_examples,
        "policy_loss": policy_loss_sum / total_examples,
        "value_loss": value_loss_sum / total_examples,
        "top1": top1_correct / total_examples,
        "top3": top3_correct / total_examples,
        "top5": top5_correct / total_examples,
        "stop_acc": stop_correct / stop_total if stop_total > 0 else 0.0,
        "switch_acc": switch_correct / switch_total if switch_total > 0 else 0.0,
        "examples": float(total_examples),
    }


def evaluate_training_samples(
    model: GraphPolicyValueNet,
    dataset: GraphSelfPlayDataset,
    device: torch.device,
    max_samples: int = 20,
) -> None:
    """
    Print final predictions on a small subset of training data.
    """

    model.eval()
    n = min(len(dataset), int(max_samples))

    print("\nFinal predictions:")

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]

            bus_features = sample["bus_features"].unsqueeze(0).to(device)
            branch_features = sample["branch_features"].unsqueeze(0).to(device)
            edge_index = sample["edge_index"].unsqueeze(0).to(device)
            action_mask = sample["action_mask"].unsqueeze(0).to(device)
            target_policy = sample["target_policy"].unsqueeze(0).to(device)

            target_value = float(sample["target_value"].item())

            logits, value = model(
                bus_features=bus_features,
                branch_features=branch_features,
                edge_index=edge_index,
                action_mask=action_mask,
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
    """
    Normalize scenario_id for robust train/validation overlap checks.
    """

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def collect_scenario_ids(dataset: GraphSelfPlayDataset) -> set[str]:
    """
    Collect normalized scenario_id values from a GraphSelfPlayDataset.
    """

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
    """
    Ensure that train and validation datasets do not share scenario_id values.
    """

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
            f"Examples: {preview}. "
            "Use a scenario-level split, not a row-level split."
        )


def _build_model(
    *,
    request: TrainingRequest,
    dataset: GraphSelfPlayDataset,
    device: torch.device,
) -> torch.nn.Module:
    if request.config.model_type == "graph_v2":
        return GraphPolicyValueNetV2(
            num_bus_features=dataset.num_bus_features,
            num_branch_features=dataset.num_branch_features,
            num_actions=dataset.num_actions,
            hidden_dim=request.config.hidden_dim,
            num_layers=request.config.num_layers,
            dropout=request.config.dropout,
        ).to(device)

    if request.config.model_type == "graph_v1":
        return GraphPolicyValueNet(
            num_bus_features=dataset.num_bus_features,
            num_branch_features=dataset.num_branch_features,
            num_actions=dataset.num_actions,
            hidden_dim=request.config.hidden_dim,
            num_layers=request.config.num_layers,
            dropout=request.config.dropout,
        ).to(device)

    raise ValueError(
        "Unsupported training model_type: "
        f"{request.config.model_type!r}. "
        "Expected 'graph_v1' or 'graph_v2'."
    )



def _normalization_provenance(
    *,
    init_checkpoint: Path | None,
) -> dict[str, object]:
    from_init = init_checkpoint is not None
    return {
        "normalization_contract_version": 1,
        "normalization_source": "init_checkpoint" if from_init else "training_dataset",
        "normalization_frozen_from_init_checkpoint": from_init,
        "normalization_source_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
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

def train_graph_policy_value_model(
    request: TrainingRequest,
) -> Path:
    if not request.examples_csv.exists():
        raise FileNotFoundError(f"Examples CSV not found: {request.examples_csv}")

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
    if request.init_checkpoint is not None:
        if not request.init_checkpoint.exists():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {request.init_checkpoint}"
            )
        init_checkpoint_payload = load_checkpoint_payload(
            request.init_checkpoint,
            map_location="cpu",
        )
        checkpoint_normalization_stats = extract_normalization_stats(
            init_checkpoint_payload,
            source=request.init_checkpoint,
        )

    dataset = GraphSelfPlayDataset(
        examples_csv=request.examples_csv,
        normalize_features=request.normalize_features,
        normalization_stats=checkpoint_normalization_stats,
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
        )

    validate_no_scenario_overlap(train_dataset=dataset, val_dataset=val_dataset)

    print(f"Examples:      {len(dataset)}")
    print(f"Num buses:     {dataset.num_buses}")
    print(f"Num branches:  {dataset.num_branches}")
    print(f"Num actions:   {dataset.num_actions}")
    print(f"Bus features:  {dataset.num_bus_features}")
    print(f"Branch feats:  {dataset.num_branch_features}")

    train_value_diagnostics = build_value_target_diagnostics(dataset=dataset)
    print_value_target_diagnostics(train_value_diagnostics)

    print(f"Batch size:    {request.config.batch_size}")
    print(f"Num workers:   {request.config.num_workers}")
    print(f"Hidden dim:    {request.config.hidden_dim}")
    print(f"Num layers:    {request.config.num_layers}")
    print(f"Dropout:       {request.config.dropout}")
    print(f"Model type:    {request.config.model_type}")
    print(f"Value loss:    HuberLoss(delta={request.config.value_huber_delta})")

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

    loader = DataLoader(
        dataset,
        batch_size=min(request.config.batch_size, len(dataset)),
        shuffle=True,
        num_workers=int(request.config.num_workers),
        pin_memory=pin_memory,
        generator=train_generator,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(request.config.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=int(request.config.num_workers),
            pin_memory=pin_memory,
        )

    model = _build_model(request=request, dataset=dataset, device=device)

    if request.init_checkpoint is not None:
        load_initial_checkpoint_into_model(
            model=model,
            checkpoint_path=request.init_checkpoint,
            dataset=dataset,
            model_type=request.config.model_type,
            hidden_dim=request.config.hidden_dim,
            num_layers=request.config.num_layers,
            device=device,
            checkpoint_payload=init_checkpoint_payload,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=request.config.learning_rate,
        weight_decay=1e-4,
    )
    value_loss_fn = nn.HuberLoss(delta=float(request.config.value_huber_delta))
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

    for epoch in range(1, request.config.epochs + 1):
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
                    normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
                    validation_dataset=val_dataset,
                )

            if request.config.save_multiple_best:
                current_top1 = float(val_metrics["top1"])
                current_top5 = float(val_metrics["top5"])
                current_switch = float(val_metrics["switch_acc"])
                current_policy_score = compute_policy_selection_score(val_metrics)

                if current_metric <= best_metric:
                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_loss"),
                        model=model,
                        dataset=dataset,
                        request=request,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="val_loss",
                        selector_value=current_metric,
                        val_metrics=val_metrics,
                        normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
                        validation_dataset=val_dataset,
                    )

                if current_top1 > best_top1:
                    best_top1 = current_top1
                    best_top1_epoch = epoch
                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_top1"),
                        model=model,
                        dataset=dataset,
                        request=request,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="val_top1",
                        selector_value=current_top1,
                        val_metrics=val_metrics,
                        normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
                        validation_dataset=val_dataset,
                    )

                if current_top5 > best_top5:
                    best_top5 = current_top5
                    best_top5_epoch = epoch
                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_top5"),
                        model=model,
                        dataset=dataset,
                        request=request,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="val_top5",
                        selector_value=current_top5,
                        val_metrics=val_metrics,
                        normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
                        validation_dataset=val_dataset,
                    )

                if current_switch > best_switch:
                    best_switch = current_switch
                    best_switch_epoch = epoch
                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_switch"),
                        model=model,
                        dataset=dataset,
                        request=request,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="val_switch",
                        selector_value=current_switch,
                        val_metrics=val_metrics,
                        normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
                        validation_dataset=val_dataset,
                    )

                if current_policy_score > best_policy_score:
                    best_policy_score = current_policy_score
                    best_policy_score_epoch = epoch
                    save_checkpoint_now(
                        path=checkpoint_variant_path(output_path, "best_policy"),
                        model=model,
                        dataset=dataset,
                        request=request,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch,
                        selector_name="policy_selection_score",
                        selector_value=current_policy_score,
                        val_metrics=val_metrics,
                        normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
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
                    normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
                    validation_dataset=val_dataset,
                )

            if epoch == 1 or epoch % 25 == 0 or epoch == request.config.epochs:
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
            normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
            validation_dataset=val_dataset,
        )
        checkpoint["best_epoch"] = int(best_epoch)
        checkpoint["best_metric"] = float(best_metric)

    torch.save(checkpoint, output_path)

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
            normalization_metadata=_normalization_provenance(init_checkpoint=request.init_checkpoint),
            validation_dataset=val_dataset,
        )
        last_checkpoint["saved_epoch"] = int(request.config.epochs)
        last_checkpoint["selector_name"] = "last_epoch"
        last_checkpoint["selector_value"] = float(request.config.epochs)
        torch.save(last_checkpoint, last_checkpoint_path)

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

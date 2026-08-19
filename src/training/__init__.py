"""Canonical training loop with complete checkpoint and resume state."""
from __future__ import annotations

import json
import os
import pickle
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from configs.config import cfg
from src.protocol.contracts import atomic_write_json


def configure_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _trainable(parameters):
    return [parameter for parameter in parameters if parameter.requires_grad]


def build_optimizer(model: nn.Module, scenario: str) -> torch.optim.Optimizer:
    common = {
        "weight_decay": cfg.train.weight_decay,
        "betas": cfg.train.adam_betas,
        "eps": cfg.train.adam_eps,
        "amsgrad": cfg.train.amsgrad,
    }
    if scenario == "S1":
        return torch.optim.Adam(
            _trainable(model.parameters()),
            lr=cfg.train.lr_tabular,
            **common,
        )
    if scenario == "S2":
        groups = [
            {"params": _trainable(model.branch.features.parameters()), "lr": cfg.train.lr_backbone},
            {"params": _trainable(model.branch.projection.parameters()), "lr": cfg.train.lr_fusion},
            {"params": _trainable(model.classifier.parameters()), "lr": cfg.train.lr_fusion},
        ]
    elif scenario == "S3":
        groups = [
            {"params": _trainable(model.image_branch.features.parameters()), "lr": cfg.train.lr_backbone},
            {"params": _trainable(model.image_branch.projection.parameters()), "lr": cfg.train.lr_fusion},
            {"params": _trainable(model.tabular_branch.parameters()), "lr": cfg.train.lr_tabular},
            {"params": _trainable(model.fusion.parameters()), "lr": cfg.train.lr_fusion},
            {"params": _trainable(model.classifier.parameters()), "lr": cfg.train.lr_fusion},
        ]
    else:
        raise ValueError("Canonical optimizer only supports S1, S2, or S3")
    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(groups, **common)


def _model_kwargs(batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    if "image" in batch:
        result["image"] = batch["image"].to(device, non_blocking=True)
    if "tabular" in batch:
        result["tabular"] = batch["tabular"].to(device, non_blocking=True)
    return result


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    accumulation_steps: int,
) -> Dict[str, Any]:
    training = optimizer is not None
    if training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
    total_loss = 0.0
    total_examples = 0
    all_probabilities: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    amp_enabled = bool(cfg.train.use_amp and device.type == "cuda")

    with context:
        for step, batch in enumerate(loader):
            labels = batch["label_binary"].to(device, non_blocking=True).reshape(-1, 1)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(**_model_kwargs(batch, device))
                full_loss = criterion(logits, labels)
                backward_loss = full_loss / accumulation_steps
            if training:
                scaler.scale(backward_loss).backward()
                should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            batch_size = labels.shape[0]
            total_loss += float(full_loss.detach().item()) * batch_size
            total_examples += batch_size
            all_probabilities.append(torch.sigmoid(logits.detach()).cpu().numpy().reshape(-1))
            all_labels.append(labels.detach().cpu().numpy().reshape(-1))

    probabilities = np.concatenate(all_probabilities)
    labels_np = np.concatenate(all_labels).astype(np.int64)
    if len(np.unique(labels_np)) != 2:
        raise ValueError("ROC-AUC checkpoint selection requires both classes")
    return {
        "loss": total_loss / total_examples,
        "roc_auc": float(roc_auc_score(labels_np, probabilities)),
        "probabilities": probabilities,
        "labels": labels_np,
    }


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["grad_scaler_state"])
    restore_rng_state(payload["rng_state"])
    return payload


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weights: torch.Tensor,
    scenario: str,
    device: torch.device,
    *,
    run_dir: Optional[Path] = None,
    run_metadata: Optional[Mapping[str, Any]] = None,
    resume: bool = True,
) -> nn.Module:
    """Train with val-AUC model selection and val-loss scheduling."""
    configure_determinism(cfg.train.seed)
    run_dir = Path(run_dir or (cfg.paths.checkpoint_dir / scenario))
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.json"
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = build_optimizer(model, scenario)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.scheduler_factor,
        patience=cfg.train.scheduler_patience,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(cfg.train.use_amp and device.type == "cuda"),
    )
    start_epoch = 1
    best_auc = float("-inf")
    best_epoch = 0
    patience_count = 0
    history: List[Dict[str, Any]] = []
    metadata = dict(run_metadata or {})
    metadata["pos_weight"] = float(pos_weights.item())

    if resume and last_path.exists():
        payload = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if payload.get("run_metadata") != metadata:
            raise RuntimeError("Resume checkpoint metadata differs from current canonical run")
        start_epoch = int(payload["epoch"]) + 1
        best_auc = float(payload["best_validation_auc"])
        best_epoch = int(payload["best_epoch"])
        patience_count = int(payload["patience_count"])
        history = list(payload["history"])

    for epoch in range(start_epoch, cfg.train.num_epochs + 1):
        started = time.perf_counter()
        train_result = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=cfg.train.gradient_accumulation_steps,
        )
        validation_result = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            accumulation_steps=1,
        )
        scheduler.step(validation_result["loss"])
        validation_auc = float(validation_result["roc_auc"])
        improved = validation_auc > best_auc
        if improved:
            best_auc = validation_auc
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
        history.append({
            "epoch": epoch,
            "train_loss": float(train_result["loss"]),
            "train_auc": float(train_result["roc_auc"]),
            "validation_loss": float(validation_result["loss"]),
            "validation_auc": validation_auc,
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        })
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "grad_scaler_state": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "best_validation_auc": best_auc,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "history": history,
            "run_metadata": metadata,
        }
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        atomic_write_json(history_path, {"epochs": history})
        if patience_count >= cfg.train.early_stop_patience:
            break

    if not best_path.exists():
        raise RuntimeError("Training produced no best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    return model


def save_scaler(scaler: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            pickle.dump(scaler, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

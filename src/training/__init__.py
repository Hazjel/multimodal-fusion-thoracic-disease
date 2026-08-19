"""
Training loop for S1/S2/S3 binary classification.

Handles:
- Differential learning rates (backbone/tabular/fusion)
- Adam (S1) / AdamW (S2-S3)
- Weighted BCE loss with pos_weight
- Gradient accumulation, AMP, gradient clipping
- ReduceLROnPlateau scheduler
- Early stopping + best checkpoint save
"""
import os
import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import cfg


def _build_optimizer(model: nn.Module, scenario: str) -> torch.optim.Optimizer:
    """
    Build optimizer with differential learning rates.
    S1: Adam, single LR.
    S2/S3: AdamW, separate param groups for backbone / tabular / fusion+classifier.
    """
    if scenario in ("S1", "S1-ext"):
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr_tabular,
            weight_decay=cfg.train.weight_decay,
        )

    if scenario == "S2":
        param_groups = [
            {"params": model.branch.features.parameters(), "lr": cfg.train.lr_backbone},
            {"params": model.branch.projection.parameters(), "lr": cfg.train.lr_fusion},
            {"params": model.classifier.parameters(), "lr": cfg.train.lr_fusion},
        ]
    else:  # S3
        param_groups = [
            {"params": model.image_branch.features.parameters(), "lr": cfg.train.lr_backbone},
            {"params": model.image_branch.projection.parameters(), "lr": cfg.train.lr_fusion},
            {"params": model.tabular_branch.parameters(), "lr": cfg.train.lr_tabular},
            {"params": model.fusion.parameters(), "lr": cfg.train.lr_fusion},
            {"params": model.classifier.parameters(), "lr": cfg.train.lr_fusion},
        ]

    return torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.train.weight_decay,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    is_train: bool,
    accum_steps: int,
) -> Tuple[float, float]:
    """Single epoch — returns (avg_loss, avg_acc)."""
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            image   = batch["image"].to(device, non_blocking=True)
            tabular = batch["tabular"].to(device, non_blocking=True)
            labels  = batch["label_binary"].to(device, non_blocking=True).unsqueeze(1)

            with autocast(enabled=cfg.train.use_amp):
                logits = model(image=image, tabular=tabular)
                loss   = criterion(logits, labels) / accum_steps

            if is_train:
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.train.gradient_clip_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            preds   = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weights: torch.Tensor,
    scenario: str,
    device: torch.device,
) -> nn.Module:
    """
    Full training loop. Returns best model (highest val AUC proxy = lowest val loss).
    Saves checkpoint to cfg.paths.checkpoint_dir.
    """
    cfg.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights.to(device)
    )
    optimizer  = _build_optimizer(model, scenario)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.scheduler_factor,
        patience=cfg.train.scheduler_patience,
    )
    scaler       = GradScaler(enabled=cfg.train.use_amp)
    accum_steps  = cfg.train.gradient_accumulation_steps

    best_val_loss   = float("inf")
    patience_count  = 0
    ckpt_path       = cfg.paths.checkpoint_dir / f"model_{scenario.lower()}_best.pt"

    print(f"\n[Train] Scenario {scenario} | device={device} | max_epochs={cfg.train.num_epochs}")
    print(f"        batch={cfg.train.batch_size} | accum={accum_steps} | effective_batch={cfg.train.batch_size * accum_steps}")

    for epoch in range(1, cfg.train.num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, is_train=True, accum_steps=accum_steps,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            device, is_train=False, accum_steps=1,
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:3d}/{cfg.train.num_epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    [Checkpoint] Saved best model -> {ckpt_path}")
        else:
            patience_count += 1
            if patience_count >= cfg.train.early_stop_patience:
                print(f"  [EarlyStopping] No improvement for {patience_count} epochs. Stop.")
                break

    # Reload best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    print(f"[Train] Done. Best val_loss={best_val_loss:.4f}")
    return model


def save_scaler(scaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Train] Scaler saved -> {path}")

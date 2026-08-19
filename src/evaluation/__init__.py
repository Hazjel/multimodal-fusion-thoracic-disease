"""Canonical binary metrics and prediction artifact schema."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from configs.config import cfg


PREDICTION_COLUMNS = [
    "image_index", "patient_id", "fold", "split", "true_label",
    "probability", "prediction_0_5", "model", "scenario", "feature_set",
    "pretraining", "seed", "checkpoint_epoch", "protocol_hash",
    "semantic_config_hash", "run_id",
]


def _model_kwargs(batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    kwargs: Dict[str, torch.Tensor] = {}
    if "image" in batch:
        kwargs["image"] = batch["image"].to(device, non_blocking=True)
    if "tabular" in batch:
        kwargs["tabular"] = batch["tabular"].to(device, non_blocking=True)
    return kwargs


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in loader:
        logits = model(**_model_kwargs(batch, device)).reshape(-1)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["label_binary"].numpy().astype(np.int64))
    return np.concatenate(probabilities), np.concatenate(labels)


@torch.no_grad()
def collect_prediction_frame(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    fold: int,
    split: str,
    model_name: str,
    scenario: str,
    feature_set: str,
    pretraining: str,
    checkpoint_epoch: int,
    protocol_hash: str,
    semantic_config_hash: str,
    run_id: str,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        logits = model(**_model_kwargs(batch, device)).reshape(-1)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        labels = batch["label_binary"].numpy().astype(np.int64)
        patient_ids = batch["patient_id"].numpy().astype(np.int64)
        for image_index, patient_id, label, probability in zip(
            batch["image_index"], patient_ids, labels, probabilities
        ):
            rows.append({
                "image_index": str(image_index),
                "patient_id": int(patient_id),
                "fold": int(fold),
                "split": split,
                "true_label": int(label),
                "probability": float(probability),
                "prediction_0_5": int(probability >= cfg.evaluation.decision_threshold),
                "model": model_name,
                "scenario": scenario,
                "feature_set": feature_set,
                "pretraining": pretraining,
                "seed": cfg.train.seed,
                "checkpoint_epoch": int(checkpoint_epoch),
                "protocol_hash": protocol_hash,
                "semantic_config_hash": semantic_config_hash,
                "run_id": run_id,
            })
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    validate_prediction_frame(frame)
    return frame


def compute_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if len(np.unique(labels)) != 2:
        raise ValueError("Canonical binary metrics require both classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def calibration_table(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Uniform [0,1] bins, retaining empty bins as NaN rows."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        rows.append({
            "bin": bin_id,
            "lower": float(edges[bin_id]),
            "upper": float(edges[bin_id + 1]),
            "count": int(mask.sum()),
            "mean_score": float(probabilities[mask].mean()) if mask.any() else np.nan,
            "fraction_positive": float(labels[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Prediction schema mismatch: {list(frame.columns)}")
    key = ["run_id", "split", "image_index"]
    if frame.duplicated(key).any():
        raise ValueError("Duplicate image prediction within run/split")
    if not frame["true_label"].isin([0, 1]).all():
        raise ValueError("true_label must be binary")
    if not frame["prediction_0_5"].isin([0, 1]).all():
        raise ValueError("prediction_0_5 must be binary")
    if not frame["probability"].between(0, 1).all():
        raise ValueError("probability must be in [0,1]")


def validate_oof_coverage(frame: pd.DataFrame, expected_image_indices) -> None:
    validate_prediction_frame(frame)
    expected = set(map(str, expected_image_indices))
    actual = set(frame["image_index"].astype(str))
    if actual != expected or len(frame) != len(expected):
        raise ValueError(
            f"OOF coverage mismatch: expected={len(expected)}, rows={len(frame)}, unique={len(actual)}"
        )


def write_prediction_frame(frame: pd.DataFrame, path: Path) -> None:
    validate_prediction_frame(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "validation",
    threshold: float = 0.5,
) -> Dict[str, float]:
    probabilities, labels = collect_predictions(model, loader, device)
    return compute_metrics(probabilities, labels, threshold)

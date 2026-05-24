"""
Evaluation metrics for binary classification (Normal vs Abnormal).

Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC.
Threshold: 0.5 for binary decision.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import cfg


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on loader. Returns (probs, labels) as numpy arrays.
    probs: sigmoid probabilities for positive class (Abnormal).
    """
    model.eval()
    all_probs  = []
    all_labels = []

    for batch in loader:
        image   = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        labels  = batch["label_binary"].numpy()

        logits = model(image=image, tabular=tabular).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Returns dict with: accuracy, precision, recall, f1, auc_roc.
    """
    preds = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    return {
        "accuracy":  accuracy_score(labels_int, preds),
        "precision": precision_score(labels_int, preds, zero_division=0),
        "recall":    recall_score(labels_int, preds, zero_division=0),
        "f1":        f1_score(labels_int, preds, zero_division=0),
        "auc_roc":   roc_auc_score(labels_int, probs),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "test",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Full evaluation pipeline — collect predictions then compute metrics.
    Prints formatted results table.
    """
    probs, labels = collect_predictions(model, loader, device)
    metrics = compute_metrics(probs, labels, threshold)

    print(f"\n[Eval] {split_name.upper()} SET RESULTS")
    print(f"  {'Metric':<12} {'Value':>8}")
    print(f"  {'-'*22}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:>8.4f}")

    return metrics


def plot_roc_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    scenario: str,
    save_path: Path,
) -> None:
    """Plot and save ROC curve for a scenario."""
    fpr, tpr, _ = roc_curve(labels.astype(int), probs)
    auc = roc_auc_score(labels.astype(int), probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{scenario} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {scenario}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] ROC curve saved → {save_path}")


def compare_scenarios(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print comparison table for S1/S2/S3.
    Computes delta_AUC = AUC_S3 - max(AUC_S1, AUC_S2).
    """
    metrics_order = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    scenarios     = list(results.keys())

    header = f"  {'Metric':<12}" + "".join(f"  {s:>10}" for s in scenarios)
    print(f"\n[Eval] SCENARIO COMPARISON")
    print(header)
    print(f"  {'-'*( 12 + 12*len(scenarios) )}")

    for m in metrics_order:
        row = f"  {m:<12}" + "".join(f"  {results[s][m]:>10.4f}" for s in scenarios)
        print(row)

    if "S1" in results and "S2" in results and "S3" in results:
        delta_auc = results["S3"]["auc_roc"] - max(
            results["S1"]["auc_roc"], results["S2"]["auc_roc"]
        )
        sign = "+" if delta_auc >= 0 else ""
        print(f"\n  ΔAUC (S3 vs best unimodal) = {sign}{delta_auc:.4f}")
        if delta_auc > 0:
            print("  → Multimodal fusion outperforms unimodal baselines.")
        else:
            print("  → Multimodal fusion does NOT improve over best unimodal baseline.")

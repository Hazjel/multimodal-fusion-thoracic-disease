"""
Patient-level k-fold cross-validation on the TRAINING data.

Why this exists
---------------
The official NIH test set (``test_list.txt``) stays untouched as the final,
independent hold-out. On top of the training pool (``train_val_list.txt``) we
run k-fold CV *grouped by Patient ID* so that no patient's follow-up images
leak across folds. Each fold trains a fresh model, evaluates on its validation
fold, and we report mean +/- std of every metric across folds — a more robust
estimate than a single split.

The reviewer's two requests are answered as follows:
  * stability / generalisation across folds  -> mean +/- std here
  * confidence interval + significance test   -> done on the test set in
    ``evaluation.stats`` (bootstrap CI + DeLong), NOT inside CV.

Usage
-----
    from src.training.cv import run_cross_validation
    summary = run_cross_validation("S3", k=5, device=DEVICE)
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import cfg
from src.data.dataset import (
    build_image_index,
    load_and_prepare_metadata,
    compute_pos_weight,
    get_transforms,
    NIHChestXrayDataset,
)
from src.models.architectures import build_model
from src.training import train
from src.evaluation import collect_predictions, compute_metrics
from src.evaluation.stats import summarize_folds, wilcoxon_folds


def _load_train_pool():
    """Return (df_train_pool, image_index): the official NIH train_val pool only."""
    image_index = build_image_index(cfg.paths.image_dirs)
    df = load_and_prepare_metadata(cfg.paths.csv_path, image_index)

    train_images = set(
        __import__("pandas").read_csv(cfg.paths.train_list_path, header=None)[0].tolist()
    )
    df_pool = df[df["Image Index"].isin(train_images)].reset_index(drop=True)
    return df_pool, image_index


def _make_loaders(df_tr, df_va, image_index, batch_size):
    """Build train/val loaders for one fold; scaler is fit on the fold's train split."""
    train_ds = NIHChestXrayDataset(
        df_tr, image_index,
        transform=get_transforms(is_training=True),
        fit_scaler=True,
    )
    val_ds = NIHChestXrayDataset(
        df_va, image_index,
        transform=get_transforms(is_training=False),
        scaler=train_ds.scaler,
    )
    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader


def run_cross_validation(
    scenario: str,
    k: int = 5,
    device: torch.device = None,
    batch_size: int = None,
) -> Dict:
    """
    Run patient-level k-fold CV for one scenario.

    Returns dict:
        {
          "scenario": str,
          "per_fold": [metric_dict, ...],   # one per fold
          "summary":  {metric: {mean, std}},
          "fold_auc": [auc, ...],           # convenience for Wilcoxon later
        }
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or cfg.train.batch_size

    df_pool, image_index = _load_train_pool()
    groups = df_pool["Patient ID"].values
    gkf = GroupKFold(n_splits=k)

    print(f"\n{'='*60}\n  {k}-FOLD PATIENT-LEVEL CV — SCENARIO {scenario}\n{'='*60}")
    print(f"  Pool: {len(df_pool)} images | {df_pool['Patient ID'].nunique()} patients")

    per_fold: List[Dict[str, float]] = []
    fold_auc: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df_pool, groups=groups), start=1):
        df_tr = df_pool.iloc[tr_idx].reset_index(drop=True)
        df_va = df_pool.iloc[va_idx].reset_index(drop=True)
        print(f"\n[CV] Fold {fold}/{k} — train {len(df_tr)} | val {len(df_va)} "
              f"(patients: {df_tr['Patient ID'].nunique()}/{df_va['Patient ID'].nunique()})")

        train_loader, val_loader = _make_loaders(df_tr, df_va, image_index, batch_size)
        pos_weights = compute_pos_weight(df_tr)

        model = build_model(scenario).to(device)
        model = train(model, train_loader, val_loader, pos_weights,
                      f"{scenario}_fold{fold}", device)

        probs, labels = collect_predictions(model, val_loader, device)
        metrics = compute_metrics(probs, labels)
        per_fold.append(metrics)
        fold_auc.append(metrics["auc_roc"])
        print(f"[CV] Fold {fold} AUC = {metrics['auc_roc']:.4f}")

    summary = summarize_folds(per_fold)

    print(f"\n[CV] {scenario} — mean +/- std across {k} folds:")
    for m, s in summary.items():
        print(f"  {m:<12} {s['mean']:.4f} +/- {s['std']:.4f}")

    return {
        "scenario": scenario,
        "per_fold": per_fold,
        "summary": summary,
        "fold_auc": fold_auc,
    }


def compare_cv_scenarios(cv_results: Dict[str, Dict]) -> None:
    """
    Given {scenario: cv_result}, run Wilcoxon signed-rank on per-fold AUCs
    for S3 vs each unimodal baseline.
    """
    if not all(s in cv_results for s in ("S1", "S2", "S3")):
        return
    print(f"\n[CV-Stats] Wilcoxon signed-rank on per-fold AUC — H0: S3 == baseline")
    for base in ("S1", "S2"):
        w = wilcoxon_folds(cv_results["S3"]["fold_auc"], cv_results[base]["fold_auc"])
        star = "*" if (w["p_value"] == w["p_value"] and w["p_value"] < 0.05) else " "
        pval = "nan" if w["p_value"] != w["p_value"] else f"{w['p_value']:.4g}"
        print(f"  S3 ({w['mean_a']:.4f}) vs {base} ({w['mean_b']:.4f}):  p = {pval} {star}")
    print("  (* p < 0.05; n = number of folds, low power for small k)")

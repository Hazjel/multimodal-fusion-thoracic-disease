"""
Statistical inference for AUC comparison.

Two tools the reviewer asked for:
  1. bootstrap_auc_ci  -> 95% confidence interval for each scenario's AUC
                          (resampling on the official NIH test set, no retraining).
  2. delong_test       -> paired significance test for the difference between two
                          correlated AUCs evaluated on the SAME test set
                          (e.g. S3 vs S1, S3 vs S2).

For the cross-validation route, wilcoxon_folds compares per-fold AUCs between
two scenarios (n = number of folds).

All functions operate on (probs, labels) numpy arrays produced by
``evaluation.collect_predictions`` — no model retraining required for the
test-set statistics.
"""
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────────────────────────────
# Bootstrap confidence interval for a single AUC
# ──────────────────────────────────────────────────────────────────────
def bootstrap_auc_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Stratified bootstrap percentile CI for AUC-ROC.

    Resamples test predictions with replacement n_boot times, keeping the
    positive/negative ratio fixed (stratified) so every resample is a valid
    ROC problem, then takes the alpha/2 and 1-alpha/2 percentiles.

    Returns dict: {auc, ci_low, ci_high, se}.
    """
    labels = labels.astype(int)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng = np.random.RandomState(seed)

    point_auc = roc_auc_score(labels, probs)
    boot_aucs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        pos_s = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_s = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_s, neg_s])
        boot_aucs[b] = roc_auc_score(labels[idx], probs[idx])

    lo = float(np.percentile(boot_aucs, 100 * alpha / 2))
    hi = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))

    return {
        "auc": float(point_auc),
        "ci_low": lo,
        "ci_high": hi,
        "se": float(boot_aucs.std(ddof=1)),
    }


# ──────────────────────────────────────────────────────────────────────
# DeLong test for two correlated AUCs
# ──────────────────────────────────────────────────────────────────────
# Implementation follows Sun & Xu (2014), "Fast Implementation of DeLong's
# Algorithm for Comparing the Areas Under Correlated ROC Curves", IEEE SPL.
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank (average rank for ties) of a 1-D array."""
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    midrank_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        midrank_sorted[i:j] = 0.5 * (i + j - 1) + 1  # 1-based average rank
        i = j
    midrank = np.empty(n, dtype=np.float64)
    midrank[order] = midrank_sorted
    return midrank


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong covariance. Expects predictions ordered positives-first.

    predictions_sorted_transposed: shape (k, n) for k classifiers, n samples,
        with the first label_1_count columns being the positive cases.
    Returns (aucs, covariance_matrix).
    """
    m = label_1_count                       # positives
    n = predictions_sorted_transposed.shape[1] - m   # negatives
    k = predictions_sorted_transposed.shape[0]

    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    # np.cov returns a scalar for k=1 vectors; force 2-D
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    DeLong test for H0: AUC_a == AUC_b on the same test set (paired).

    Returns dict: {auc_a, auc_b, delta_auc, z, p_value}.
    p_value is two-sided.
    """
    labels = labels.astype(int)
    order = np.argsort(-labels, kind="mergesort")  # positives (label 1) first
    label_1_count = int(labels.sum())

    preds = np.vstack([probs_a[order], probs_b[order]])
    aucs, cov = _fast_delong(preds, label_1_count)

    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        # Identical predictions or degenerate variance.
        z = 0.0
        p = 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var)
        p = 2.0 * stats.norm.sf(abs(z))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc": auc_a - auc_b,
        "z": float(z),
        "p_value": float(p),
    }


# ──────────────────────────────────────────────────────────────────────
# Wilcoxon signed-rank across CV folds
# ──────────────────────────────────────────────────────────────────────
def wilcoxon_folds(
    auc_a: Sequence[float],
    auc_b: Sequence[float],
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test on paired per-fold AUCs (scenario A vs B).

    Use when you have k>=5 folds. Returns {mean_a, mean_b, statistic, p_value}.
    Falls back gracefully when scipy cannot run (e.g. all differences zero).
    """
    a = np.asarray(auc_a, dtype=np.float64)
    b = np.asarray(auc_b, dtype=np.float64)
    out = {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "statistic": float("nan"),
        "p_value": float("nan"),
    }
    try:
        res = stats.wilcoxon(a, b)
        out["statistic"] = float(res.statistic)
        out["p_value"] = float(res.pvalue)
    except ValueError:
        # e.g. zero differences across all folds
        pass
    return out


def summarize_folds(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-fold metric dicts into mean ± std.

    Input: list of metric dicts (one per fold), each with the same keys.
    Output: {metric: {mean, std}}.
    """
    if not fold_metrics:
        return {}
    keys = fold_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics], dtype=np.float64)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)}
    return summary

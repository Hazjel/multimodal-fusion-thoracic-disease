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
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
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


def paired_patient_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    probability_a: str,
    probability_b: str,
    patient_column: str = "patient_id",
    label_column: str = "true_label",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired AUC-difference CI by resampling patients, without retraining."""
    required = {patient_column, label_column, probability_a, probability_b}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cluster bootstrap missing columns: {sorted(missing)}")
    patients = np.asarray(sorted(frame[patient_column].unique()))
    if len(patients) < 2:
        raise ValueError("Cluster bootstrap requires at least two patients")
    grouped_indices = {
        patient: frame.index[frame[patient_column] == patient].to_numpy()
        for patient in patients
    }
    labels = frame[label_column].to_numpy(dtype=np.int64)
    probs_a = frame[probability_a].to_numpy(dtype=np.float64)
    probs_b = frame[probability_b].to_numpy(dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise ValueError("Point estimate requires both classes")
    auc_a = float(roc_auc_score(labels, probs_a))
    auc_b = float(roc_auc_score(labels, probs_b))
    rng = np.random.RandomState(seed)
    deltas = []
    attempts = 0
    max_attempts = n_boot * 20
    while len(deltas) < n_boot and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([grouped_indices[patient] for patient in sampled])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) != 2:
            continue
        deltas.append(
            roc_auc_score(sampled_labels, probs_a[indices])
            - roc_auc_score(sampled_labels, probs_b[indices])
        )
    if len(deltas) != n_boot:
        raise RuntimeError(f"Only generated {len(deltas)}/{n_boot} valid bootstrap replicates")
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc": auc_a - auc_b,
        "ci_low": float(np.percentile(values, 100 * alpha / 2)),
        "ci_high": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "se": float(values.std(ddof=1)),
        "n_boot": int(n_boot),
        "resampling_unit": patient_column,
        "conditional_on_fitted_cv_models": True,
    }


def select_cnn_candidate(
    predictions: Mapping[str, pd.DataFrame],
    fold_aucs: Mapping[str, Sequence[float]],
    trainable_parameters: Mapping[str, int],
    median_training_seconds: Mapping[str, float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Apply the frozen no-clear-separation candidate-set heuristic."""
    names = sorted(predictions)
    if set(names) != set(fold_aucs) or set(names) != set(trainable_parameters) or set(names) != set(median_training_seconds):
        raise ValueError("Selection inputs must contain the same candidates")
    pooled_auc = {}
    for name in names:
        candidate = predictions[name]
        pooled_auc[name] = float(roc_auc_score(candidate["true_label"], candidate["probability"]))
    top = max(names, key=lambda name: (pooled_auc[name], name))
    top_frame = predictions[top][["image_index", "patient_id", "true_label", "probability"]].rename(
        columns={"probability": "probability_top"}
    )
    candidate_set = [top]
    comparisons: Dict[str, Any] = {}
    for index, name in enumerate(names):
        if name == top:
            continue
        other = predictions[name][["image_index", "patient_id", "true_label", "probability"]].rename(
            columns={"probability": "probability_other"}
        )
        merged = top_frame.merge(
            other,
            on=["image_index", "patient_id", "true_label"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(top_frame) or len(merged) != len(other):
            raise ValueError(f"OOF predictions are not aligned for {top} vs {name}")
        result = paired_patient_cluster_bootstrap(
            merged,
            probability_a="probability_top",
            probability_b="probability_other",
            n_boot=n_boot,
            seed=seed + index,
        )
        comparisons[f"{top}_vs_{name}"] = result
        if result["ci_low"] <= 0.0 <= result["ci_high"]:
            candidate_set.append(name)

    selected = min(
        candidate_set,
        key=lambda name: (
            float(np.std(fold_aucs[name], ddof=1)),
            int(trainable_parameters[name]),
            float(median_training_seconds[name]),
            name,
        ),
    )
    return {
        "top_by_pooled_auc": top,
        "pooled_auc": pooled_auc,
        "heuristic_candidate_set": sorted(candidate_set),
        "comparisons": comparisons,
        "selected": selected,
        "tie_break": ["fold_auc_sd", "trainable_parameters", "median_training_seconds"],
        "interpretation": "available OOF evidence did not clearly separate members of the heuristic set",
    }

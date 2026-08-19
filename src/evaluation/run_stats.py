"""
Run bootstrap CI + DeLong test on the already-computed S1/S2/S3 predictions.

Reads results/tables/predictions_s{1,2,3}.csv (no retraining needed) and
prints/saves the statistical inference results the SINTA-2 reviewer asked
for: 95% bootstrap CI per scenario, and pairwise DeLong tests between
scenarios (S3 vs S2, S3 vs S1, S2 vs S1).

Usage:
    python -m src.evaluation.run_stats
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.stats import bootstrap_auc_ci, delong_test

TABLES_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT_PATH = TABLES_DIR / "stats_significance.json"

SCENARIOS = {
    "S1": ("predictions_s1.csv", "prob_s1"),
    "S2": ("predictions_s2.csv", "prob_s2"),
    "S3": ("predictions_s3.csv", "prob_s3"),
}


def load_predictions():
    data = {}
    labels_ref = None
    for name, (fname, prob_col) in SCENARIOS.items():
        df = pd.read_csv(TABLES_DIR / fname)
        labels = df["true_label"].to_numpy()
        probs = df[prob_col].to_numpy()
        if labels_ref is None:
            labels_ref = labels
        else:
            # Sanity check: all scenarios must share the same test set / order
            assert len(labels) == len(labels_ref), f"{name}: row count mismatch"
            assert np.array_equal(labels, labels_ref), (
                f"{name}: true_label differs from other scenarios — "
                "predictions are not aligned on the same test set/order, "
                "DeLong test would be invalid."
            )
        data[name] = probs
    return data, labels_ref


def main():
    probs, labels = load_predictions()
    print(f"Loaded {len(labels)} test samples, {int(labels.sum())} positive.\n")

    results = {"bootstrap_ci": {}, "delong": {}}

    print("=" * 60)
    print("1. Bootstrap 95% CI per scenario (n_boot=2000)")
    print("=" * 60)
    for name, p in probs.items():
        ci = bootstrap_auc_ci(p, labels)
        results["bootstrap_ci"][name] = ci
        print(f"{name}: AUC={ci['auc']:.4f}  95% CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  SE={ci['se']:.4f}")

    print()
    print("=" * 60)
    print("2. DeLong test — pairwise AUC comparison (paired, same test set)")
    print("=" * 60)
    pairs = [("S3", "S2"), ("S3", "S1"), ("S2", "S1")]
    for a, b in pairs:
        res = delong_test(probs[a], probs[b], labels)
        results["delong"][f"{a}_vs_{b}"] = res
        sig = "SIGNIFICANT (p<0.05)" if res["p_value"] < 0.05 else "not significant (p>=0.05)"
        print(
            f"{a} vs {b}: AUC {res['auc_a']:.4f} vs {res['auc_b']:.4f}  "
            f"dAUC={res['delta_auc']:+.4f}  z={res['z']:.3f}  p={res['p_value']:.4f}  -> {sig}"
        )

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

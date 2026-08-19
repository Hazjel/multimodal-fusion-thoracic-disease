"""
Feature-engineering exploration: retrain S1 (tabular-only MLP) with the
extended 6-feature set (baseline 4 + visit_count + pixel_spacing_x) to see
whether the engineered features improve on the AUC=0.6182 baseline.

Does not touch the S1 baseline scenario, its checkpoint, or its config
default — this is a standalone comparison run.

Usage:
    python -m src.training.train_s1_extended
"""
import torch

from configs.config import cfg
from src.data.dataset import create_dataloaders, NIHChestXrayDataset
from src.models.architectures import TabularMLP
from src.training import train, save_scaler
from src.evaluation import evaluate, collect_predictions
from src.evaluation.stats import delong_test, bootstrap_auc_ci

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCENARIO = "S1-ext"


def main():
    print(f"[Setup] Device: {DEVICE}")
    print(f"[Data] Loading dataset with extended tabular features...")
    print(f"        Baseline cols: {NIHChestXrayDataset.TABULAR_COLS_BASELINE}")
    print(f"        Extended cols: {NIHChestXrayDataset.TABULAR_COLS_EXTENDED}")

    train_loader, val_loader, test_loader, scaler, pos_weights = create_dataloaders(
        tabular_cols=NIHChestXrayDataset.TABULAR_COLS_EXTENDED,
    )

    n_features = len(NIHChestXrayDataset.TABULAR_COLS_EXTENDED)
    model = TabularMLP(num_classes=1, input_dim=n_features).to(DEVICE)
    print(f"[Model] S1-ext — tabular_input_dim={n_features}")

    model = train(model, train_loader, val_loader, pos_weights, SCENARIO, DEVICE)

    scaler_path = cfg.paths.checkpoint_dir / "scaler_s1ext.pkl"
    save_scaler(scaler, scaler_path)

    metrics = evaluate(model, test_loader, DEVICE, split_name="S1-ext_test")
    probs, labels = collect_predictions(model, test_loader, DEVICE)
    print(f"[Eval] S1-ext test metrics: {metrics}")

    # Save predictions in the same format used by run_stats.py
    import pandas as pd
    df = pd.DataFrame({
        "true_label": labels.astype(int),
        "prob_s1ext": probs,
        "pred_s1ext": (probs >= 0.5).astype(int),
    })
    tables_dir = cfg.paths.results_dir / "tables"
    df.to_csv(tables_dir / "predictions_s1ext.csv", index=False)
    pd.DataFrame([metrics]).to_csv(tables_dir / "results_s1ext.csv", index=False)
    print(f"[Eval] Saved -> {tables_dir / 'predictions_s1ext.csv'}")

    # Compare directly against the S1 baseline
    baseline_path = tables_dir / "predictions_s1.csv"
    if baseline_path.exists():
        s1_df = pd.read_csv(baseline_path)
        s1_probs = s1_df["prob_s1"].to_numpy()
        s1_labels = s1_df["true_label"].to_numpy()

        if len(s1_labels) == len(labels) and (s1_labels == labels.astype(int)).all():
            print("\n" + "=" * 60)
            print("S1-ext (6 features) vs S1 baseline (4 features)")
            print("=" * 60)
            ci_ext = bootstrap_auc_ci(probs, labels)
            ci_base = bootstrap_auc_ci(s1_probs, s1_labels)
            print(f"S1 baseline: AUC={ci_base['auc']:.4f}  95% CI=[{ci_base['ci_low']:.4f}, {ci_base['ci_high']:.4f}]")
            print(f"S1-ext:      AUC={ci_ext['auc']:.4f}  95% CI=[{ci_ext['ci_low']:.4f}, {ci_ext['ci_high']:.4f}]")

            d = delong_test(probs, s1_probs, labels)
            sig = "SIGNIFICANT (p<0.05)" if d["p_value"] < 0.05 else "not significant (p>=0.05)"
            print(f"DeLong S1-ext vs S1: dAUC={d['delta_auc']:+.4f}  z={d['z']:.3f}  p={d['p_value']:.4f}  -> {sig}")
        else:
            print("[Warn] S1 baseline predictions not aligned with current test set — skipping DeLong comparison.")
    else:
        print(f"[Warn] No baseline predictions found at {baseline_path} — skipping comparison.")

    print("\n[Done] S1-ext training and evaluation complete.")


if __name__ == "__main__":
    main()

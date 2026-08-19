"""
Run inference on the official test set for a trained checkpoint and save
predictions in the same format as predictions_s{1,2,3}.csv, so it can be
compared with existing scenarios via run_stats.py / delong_test.

Does not retrain — loads an existing checkpoint and evaluates once.

Usage:
    python -m src.evaluation.eval_checkpoint --scenario S3-gated
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from configs.config import cfg
from src.data.dataset import create_dataloaders
from src.models.architectures import build_model
from src.evaluation import collect_predictions, compute_metrics

TABLES_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        help="Scenario name matching a saved checkpoint, e.g. S3-gated")
    args = parser.parse_args()
    scenario = args.scenario
    slug = scenario.lower()

    ckpt_path = cfg.paths.checkpoint_dir / f"model_{slug}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    print(f"[Eval] Loading dataset...")
    _, _, test_loader, _, _ = create_dataloaders()

    print(f"[Eval] Loading checkpoint: {ckpt_path}")
    model = build_model(scenario).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    print(f"[Eval] Running inference on test set ({scenario})...")
    probs, labels = collect_predictions(model, test_loader, DEVICE)

    metrics = compute_metrics(probs, labels)
    print(f"[Eval] {scenario} test metrics: {metrics}")

    # Save predictions CSV in the same format as predictions_s{1,2,3}.csv
    prob_col = f"prob_{slug.replace('-', '')}"
    pred_col = f"pred_{slug.replace('-', '')}"
    df = pd.DataFrame({
        "true_label": labels.astype(int),
        prob_col: probs,
        pred_col: (probs >= 0.5).astype(int),
    })
    out_path = TABLES_DIR / f"predictions_{slug}.csv"
    df.to_csv(out_path, index=False)
    print(f"[Eval] Saved -> {out_path}")

    # Save results summary row
    results_path = TABLES_DIR / f"results_{slug}.csv"
    pd.DataFrame([metrics]).to_csv(results_path, index=False)
    print(f"[Eval] Saved -> {results_path}")


if __name__ == "__main__":
    main()

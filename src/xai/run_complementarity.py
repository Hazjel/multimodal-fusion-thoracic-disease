"""
Run the SHAP<->Grad-CAM complementarity analysis on the trained S3 (concat
fusion) checkpoint, without retraining.

Loads the existing S3 checkpoint (results/checkpoints/model_s3_multimodal_binary.pt),
builds a SHAP KernelExplainer on the tabular branch, and runs
analyze_complementarity() over a sample of the test set.

Usage:
    python -m src.xai.run_complementarity
    python -m src.xai.run_complementarity --n-samples 200 --checkpoint model_s3_multimodal_binary.pt
"""
import argparse
from pathlib import Path

import torch

from configs.config import cfg
from src.data.dataset import create_dataloaders
from src.models.architectures import build_model
from src.xai import build_shap_background, make_shap_explainer
from src.xai.complementarity import (
    analyze_complementarity, save_per_sample_csv,
    plot_complementarity_scatter, print_summary,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model_s3_multimodal_binary.pt",
                         help="Checkpoint filename inside results/checkpoints/")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--shap-nsamples", type=int, default=128,
                         help="SHAP KernelExplainer nsamples per prediction (perf/accuracy tradeoff)")
    args = parser.parse_args()

    ckpt_path = cfg.paths.checkpoint_dir / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    print(f"[Setup] Device: {DEVICE}")
    print("[Data] Loading dataset...")
    train_loader, _, test_loader, _, _ = create_dataloaders()
    train_ds = train_loader.dataset

    print(f"[Model] Loading S3 checkpoint: {ckpt_path}")
    model = build_model("S3").to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    print("[XAI] Building SHAP background + explainer...")
    background = build_shap_background(train_ds, n_samples=100)
    explainer = make_shap_explainer(model, background, DEVICE)
    target_layer = model.get_cam_target_layer()

    print(f"[XAI] Running complementarity analysis (n_samples={args.n_samples})...")
    comp = analyze_complementarity(
        model, test_loader, explainer, target_layer, DEVICE,
        n_samples=args.n_samples, shap_nsamples=args.shap_nsamples,
    )

    save_per_sample_csv(comp, cfg.paths.xai_dir / "complementarity.csv")
    plot_complementarity_scatter(comp, cfg.paths.xai_dir / "complementarity_scatter.png")
    print_summary(comp)
    print("[XAI] Done.")


if __name__ == "__main__":
    main()

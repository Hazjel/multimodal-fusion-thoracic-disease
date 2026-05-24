"""
Main experiment runner — trains S1/S2/S3, evaluates, runs XAI.

Usage:
    python run_experiment.py --scenario S1
    python run_experiment.py --scenario S2
    python run_experiment.py --scenario S3
    python run_experiment.py --scenario all
    python run_experiment.py --scenario S3 --xai-only  (skip training, load checkpoint)
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import cfg
from src.data.dataset import create_dataloaders
from src.models.architectures import build_model
from src.training import train, save_scaler
from src.evaluation import evaluate, plot_roc_curve, compare_scenarios, collect_predictions
from src.xai import (
    build_shap_background, save_shap_background, load_shap_background,
    make_shap_explainer, compute_shap_values,
    plot_shap_summary, plot_shap_waterfall,
    compute_gradcam, plot_gradcam_grid,
)


SCENARIOS = ["S1", "S2", "S3"]
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(scenario: str, train_loader, val_loader, test_loader, pos_weights, scaler):
    print(f"\n{'='*60}")
    print(f"  SCENARIO {scenario}")
    print(f"{'='*60}")

    model = build_model(scenario).to(DEVICE)
    model = train(model, train_loader, val_loader, pos_weights, scenario, DEVICE)

    # Save scaler (needed for inference/XAI)
    if scenario in ("S1", "S3"):
        scaler_path = cfg.paths.checkpoint_dir / "scaler.pkl"
        save_scaler(scaler, scaler_path)

    return model


def run_evaluation(scenario: str, model, test_loader):
    metrics = evaluate(model, test_loader, DEVICE, split_name=f"{scenario}_test")
    probs, labels = collect_predictions(model, test_loader, DEVICE)
    plot_roc_curve(
        probs, labels, scenario,
        cfg.paths.figures_dir / f"roc_{scenario.lower()}.png",
    )
    return metrics


def run_xai(model, train_ds, test_loader):
    """Run dual XAI on S3 model."""
    print("\n[XAI] Running dual XAI on S3...")

    # ── SHAP ──────────────────────────────────────────────
    bg_path = cfg.paths.xai_dir / "shap_background.npy"
    if bg_path.exists():
        print(f"[XAI] Loading existing SHAP background from {bg_path}")
        background = load_shap_background(bg_path)
    else:
        print("[XAI] Building SHAP background (100 train samples)...")
        background = build_shap_background(train_ds, n_samples=100)
        save_shap_background(background, bg_path)

    explainer = make_shap_explainer(model, background, DEVICE)

    # Collect tabular from first 200 test samples for summary plot
    test_tab = []
    for batch in test_loader:
        test_tab.append(batch["tabular"].numpy())
        if sum(len(t) for t in test_tab) >= 200:
            break
    import numpy as np
    X_tab_test = np.concatenate(test_tab)[:200]

    print("[XAI] Computing SHAP values (this may take several minutes)...")
    shap_vals = compute_shap_values(explainer, X_tab_test, nsamples=128)

    plot_shap_summary(
        shap_vals, X_tab_test,
        cfg.paths.xai_dir / "shap_summary.png",
    )

    # Waterfall for first sample
    plot_shap_waterfall(
        shap_vals[0], float(explainer.expected_value),
        cfg.paths.xai_dir / "shap_waterfall_sample0.png",
        title="SHAP Waterfall — Sample 0",
    )

    # ── Grad-CAM ──────────────────────────────────────────
    print("[XAI] Computing Grad-CAM on denseblock4...")
    target_layer = model.image_branch.features.denseblock4

    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((cfg.data.image_size, cfg.data.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cam_images, cam_heatmaps, cam_titles = [], [], []
    n_cam = 8  # visualize first 8 test samples

    for batch in test_loader:
        for i in range(min(n_cam, len(batch["image"]))):
            img_tensor = batch["image"][i:i+1].to(DEVICE)
            tab_tensor = batch["tabular"][i:i+1].to(DEVICE)
            label      = int(batch["label_binary"][i].item())
            label_str  = "Abnormal" if label == 1 else "Normal"

            heatmap = compute_gradcam(model, img_tensor, tab_tensor, target_layer)
            # Reconstruct approximate PIL image for overlay (denormalize)
            img_np = batch["image"][i].permute(1, 2, 0).numpy()
            mean   = [0.485, 0.456, 0.406]
            std    = [0.229, 0.224, 0.225]
            img_np = (img_np * std + mean).clip(0, 1)
            pil_img = Image.fromarray((img_np * 255).astype("uint8"))

            cam_images.append(pil_img)
            cam_heatmaps.append(heatmap)
            cam_titles.append(f"GT: {label_str}")

        if len(cam_images) >= n_cam:
            break

    plot_gradcam_grid(
        cam_images[:n_cam], cam_heatmaps[:n_cam], cam_titles[:n_cam],
        cfg.paths.xai_dir / "gradcam_grid.png",
    )
    print("[XAI] Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all",
                        choices=["S1", "S2", "S3", "all"])
    parser.add_argument("--xai-only", action="store_true",
                        help="Skip training, load existing S3 checkpoint for XAI")
    args = parser.parse_args()

    print(f"[Setup] Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}")
        print(f"        VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n[Data] Loading dataset...")
    train_loader, val_loader, test_loader, scaler, pos_weights = create_dataloaders()

    # Reference to train dataset for SHAP background
    train_ds = train_loader.dataset

    target_scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    all_results = {}

    for scenario in target_scenarios:
        ckpt_path = cfg.paths.checkpoint_dir / f"model_{scenario.lower()}_best.pt"

        if args.xai_only and scenario == "S3":
            print(f"\n[XAI-only] Loading S3 checkpoint from {ckpt_path}")
            model = build_model("S3").to(DEVICE)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            )
        else:
            model = run_training(
                scenario, train_loader, val_loader, test_loader, pos_weights, scaler
            )

        metrics = run_evaluation(scenario, model, test_loader)
        all_results[scenario] = metrics

        if scenario == "S3":
            run_xai(model, train_ds, test_loader)

    if len(all_results) > 1:
        compare_scenarios(all_results)

    print("\n[Done] All experiments complete.")
    print(f"       Results saved to: {cfg.paths.results_dir}")


if __name__ == "__main__":
    main()

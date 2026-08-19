"""
Explainable AI module: SHAP KernelExplainer (tabular) + Grad-CAM (image).

SHAP:  tabular branch of S3 — background=100 train samples, zero image to isolate tabular.
Grad-CAM: denseblock4 of DenseNet-121 — overlaid on original X-ray.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import shap
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import cfg


# ── SHAP ──────────────────────────────────────────────────────────────────────

TABULAR_FEATURE_NAMES = ["Usia", "Jenis Kelamin", "Posisi (PA)", "Follow-up #"]


def build_shap_background(
    dataset,
    n_samples: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample n_samples tabular vectors from dataset (training set) as SHAP background.
    Background is fit to training distribution — standard SHAP practice.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    background = np.stack([dataset[i]["tabular"].numpy() for i in indices])
    return background


def save_shap_background(background: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, background)
    print(f"[XAI] SHAP background ({len(background)} samples) saved -> {path}")


def load_shap_background(path: Path) -> np.ndarray:
    return np.load(path)


def make_shap_explainer(
    model: nn.Module,
    background: np.ndarray,
    device: torch.device,
) -> shap.KernelExplainer:
    """
    Build SHAP KernelExplainer wrapping the tabular branch of S3.
    Image is zeroed out to isolate tabular feature contribution.
    """
    zero_image = torch.zeros(
        1, 3, cfg.data.image_size, cfg.data.image_size, device=device
    )

    @torch.no_grad()
    def _predict(X_tab: np.ndarray) -> np.ndarray:
        results = []
        batch_size = 64
        for i in range(0, len(X_tab), batch_size):
            tab = torch.tensor(X_tab[i : i + batch_size], dtype=torch.float32, device=device)
            img = zero_image.expand(len(tab), -1, -1, -1)
            logits = model(image=img, tabular=tab).squeeze(1)
            results.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(results)

    return shap.KernelExplainer(_predict, background)


def compute_shap_values(
    explainer: shap.KernelExplainer,
    X_tab: np.ndarray,
    nsamples: int = 128,
) -> np.ndarray:
    """Compute SHAP values. Returns array shape (n_samples, n_features)."""
    return explainer.shap_values(X_tab, nsamples=nsamples)


def plot_shap_summary(
    shap_values: np.ndarray,
    X_tab: np.ndarray,
    save_path: Path,
) -> None:
    """Beeswarm summary plot for SHAP values across all samples."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(
        shap_values, X_tab,
        feature_names=TABULAR_FEATURE_NAMES,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[XAI] SHAP summary plot saved -> {save_path}")


def plot_shap_waterfall(
    shap_values_single: np.ndarray,
    base_value: float,
    save_path: Path,
    title: str = "SHAP Waterfall",
) -> None:
    """Waterfall plot for a single sample."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    expl = shap.Explanation(
        values=shap_values_single,
        base_values=base_value,
        feature_names=TABULAR_FEATURE_NAMES,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    shap.waterfall_plot(expl, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[XAI] SHAP waterfall saved -> {save_path}")


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(
    model: nn.Module,
    img_tensor: torch.Tensor,
    tab_tensor: torch.Tensor,
    target_layer: nn.Module,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap from target_layer (denseblock4).

    Steps (as per proposal §3.7.2):
    1. Forward pass — register activation hook on target_layer
    2. Backward pass — register gradient hook
    3. Weight = GAP of gradients per channel
    4. Heatmap = ReLU(sum_k(weight_k * activation_k))
    5. Normalize to [0, 1]

    Returns heatmap (H, W) normalized float in [0, 1].
    """
    model.eval()
    activations: List[torch.Tensor] = []
    gradients:   List[torch.Tensor] = []

    h_fwd = target_layer.register_forward_hook(
        lambda _m, _i, o: activations.append(o.detach())
    )
    h_bwd = target_layer.register_full_backward_hook(
        lambda _m, _gi, go: gradients.append(go[0].detach())
    )

    logit = model(image=img_tensor, tabular=tab_tensor).squeeze()
    model.zero_grad()
    logit.backward()

    h_fwd.remove()
    h_bwd.remove()

    # weights: GAP over spatial dims per channel
    act = activations[0].squeeze(0)  # (C, H, W)
    grd = gradients[0].squeeze(0)    # (C, H, W)
    weights = grd.mean(dim=(1, 2))   # (C,)

    heatmap = torch.relu((weights[:, None, None] * act).sum(0))  # (H, W)
    heatmap = heatmap.cpu().numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def overlay_gradcam(
    pil_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    img_size: int = None,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original X-ray image.
    Returns RGB numpy array (H, W, 3).
    """
    if img_size is None:
        img_size = cfg.data.image_size

    img = np.array(pil_img.resize((img_size, img_size)))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    h = cv2.resize(heatmap, (img_size, img_size))
    colored = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)


def plot_gradcam_grid(
    images: List[Image.Image],
    heatmaps: List[np.ndarray],
    titles: List[str],
    save_path: Path,
    ncols: int = 4,
) -> None:
    """
    Plot grid of Grad-CAM overlays for multiple samples.
    Each cell: original | overlay side-by-side.
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 5, nrows * 3))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        row, col = divmod(i, ncols)
        orig    = np.array(images[i].resize((cfg.data.image_size, cfg.data.image_size)))
        overlay = overlay_gradcam(images[i], heatmaps[i])

        ax_orig    = axes[row, col * 2]
        ax_overlay = axes[row, col * 2 + 1]

        ax_orig.imshow(orig, cmap="gray" if orig.ndim == 2 else None)
        ax_orig.set_title(f"{titles[i]}\n(original)", fontsize=8)
        ax_orig.axis("off")

        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f"{titles[i]}\n(Grad-CAM)", fontsize=8)
        ax_overlay.axis("off")

    # Hide unused axes
    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[XAI] Grad-CAM grid saved -> {save_path}")

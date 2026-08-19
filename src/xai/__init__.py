"""OOF image-conditioned SHAP and backbone-aware Grad-CAM utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from PIL import Image

from configs.config import cfg


TABULAR_FEATURE_NAMES = ["Age", "Gender", "View Position", "Follow-up #"]


def build_shap_background(dataset, n_samples: int = 100, seed: int = 42) -> np.ndarray:
    """Sample fold-training vectors already transformed by that fold's scaler."""
    if getattr(dataset, "scaler", None) is None:
        raise ValueError("SHAP background requires the fold-specific fitted StandardScaler")
    if "tabular" not in getattr(dataset, "modalities", set()):
        raise ValueError("SHAP background dataset must expose the tabular modality")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    return np.stack([dataset[index]["tabular"].numpy() for index in indices]).astype(np.float32)


def save_shap_background(background: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(background, dtype=np.float32))


def load_shap_background(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def make_shap_explainer(
    model: nn.Module,
    background: np.ndarray,
    device: torch.device,
    *,
    fixed_image: torch.Tensor,
) -> shap.KernelExplainer:
    """Explain scaled metadata conditional on one actual preprocessed OOF image."""
    background = np.asarray(background, dtype=np.float32)
    if background.ndim != 2:
        raise ValueError("SHAP background must be a 2-D scaled metadata matrix")
    if fixed_image.ndim != 4 or fixed_image.shape[0] != 1:
        raise ValueError("fixed_image must have shape (1,C,H,W)")
    fixed_image = fixed_image.detach().to(device)

    @torch.no_grad()
    def predict(scaled_tabular: np.ndarray) -> np.ndarray:
        outputs = []
        for start in range(0, len(scaled_tabular), 64):
            tabular = torch.as_tensor(
                scaled_tabular[start:start + 64], dtype=torch.float32, device=device
            )
            image = fixed_image.expand(len(tabular), -1, -1, -1)
            outputs.append(torch.sigmoid(model(image=image, tabular=tabular).reshape(-1)).cpu().numpy())
        return np.concatenate(outputs)

    return shap.KernelExplainer(predict, background)


def compute_shap_values(
    explainer: shap.KernelExplainer,
    scaled_tabular: np.ndarray,
    nsamples: int = 128,
) -> np.ndarray:
    return np.asarray(explainer.shap_values(np.asarray(scaled_tabular), nsamples=nsamples))


def proportional_oof_indices(labels: np.ndarray, n_samples: int = 40, seed: int = 42) -> np.ndarray:
    """Deterministic proportional binary sampling using largest remainders."""
    labels = np.asarray(labels, dtype=np.int64)
    if n_samples > len(labels):
        raise ValueError("Cannot sample more OOF cases than the validation fold contains")
    classes, counts = np.unique(labels, return_counts=True)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("OOF SHAP sampling requires both classes")
    exact = counts / counts.sum() * n_samples
    allocation = np.floor(exact).astype(int)
    remainder = n_samples - int(allocation.sum())
    order = np.argsort(-(exact - allocation), kind="mergesort")
    allocation[order[:remainder]] += 1
    rng = np.random.RandomState(seed)
    selected = []
    for label, size in zip(classes, allocation):
        candidates = np.where(labels == label)[0]
        selected.extend(rng.choice(candidates, size=int(size), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def plot_shap_summary(shap_values: np.ndarray, scaled_tabular: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values,
        scaled_tabular,
        feature_names=TABULAR_FEATURE_NAMES[: scaled_tabular.shape[1]],
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(
    shap_values_single: np.ndarray,
    base_value: float,
    save_path: Path,
    title: str = "Conditional metadata SHAP",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    explanation = shap.Explanation(
        values=shap_values_single,
        base_values=base_value,
        feature_names=TABULAR_FEATURE_NAMES[: len(shap_values_single)],
    )
    shap.waterfall_plot(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_gradcam(
    model: nn.Module,
    img_tensor: torch.Tensor,
    tab_tensor: torch.Tensor | None,
    target_layer: nn.Module,
) -> np.ndarray:
    model.eval()
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    forward_handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output.detach())
    )
    backward_handle = target_layer.register_full_backward_hook(
        lambda _module, _grad_inputs, grad_outputs: gradients.append(grad_outputs[0].detach())
    )
    kwargs = {"image": img_tensor}
    if tab_tensor is not None:
        kwargs["tabular"] = tab_tensor
    try:
        logit = model(**kwargs).reshape(-1)[0]
        model.zero_grad(set_to_none=True)
        logit.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()
    activation = activations[0][0]
    gradient = gradients[0][0]
    weights = gradient.mean(dim=(1, 2))
    heatmap = torch.relu((weights[:, None, None] * activation).sum(0)).cpu().numpy()
    maximum = float(heatmap.max())
    return heatmap / maximum if maximum > 0 else heatmap


def overlay_gradcam(
    pil_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    img_size: int | None = None,
) -> np.ndarray:
    size = img_size or cfg.data.image_size
    image = np.asarray(pil_img.convert("RGB").resize((size, size)))
    resized = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)


def plot_gradcam_grid(
    images: List[Image.Image],
    heatmaps: List[np.ndarray],
    titles: List[str],
    save_path: Path,
    ncols: int = 4,
) -> None:
    count = len(images)
    nrows = (count + ncols - 1) // ncols
    figure, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.axis("off")
            continue
        axis.imshow(overlay_gradcam(images[index], heatmaps[index]))
        axis.set_title(titles[index])
        axis.axis("off")
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

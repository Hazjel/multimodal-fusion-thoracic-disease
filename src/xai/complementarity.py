"""
Complementarity analysis between the two XAI methods on the S3 fusion model.

Motivation
----------
Reviewers flagged that simply *running* SHAP (tabular branch) and Grad-CAM
(image branch) side by side is a weak "novelty by combination." This module
turns the dual XAI into an actual analysis: for each test case it quantifies
*how much the fusion model relies on each modality* and whether the two
explanations agree, so the contribution becomes methodological (a joint
reading of both explanations) rather than two isolated plots.

Per-sample measures
-------------------
1. modality ablation (model-level, faithful):
     p_full   = model(image, tabular)
     p_tab    = model(zero_image, tabular)   -> tabular-only signal
     p_img    = model(image, zero_tabular)   -> image-only signal
   contribution of a modality = |p_full - p_without_that_modality|.
   We approximate "without modality" by the complementary single-modality
   prediction; the relative reliance is then
     img_reliance = d_img / (d_img + d_tab + eps)

2. SHAP magnitude (tabular)  = sum_i |phi_i|              (4 features)
3. Grad-CAM magnitude (image) = mean of the [0,1] heatmap  (spatial focus)

Agreement categories (per sample)
----------------------------------
Using median splits on img_reliance:
   - "both-strong"   : strong tabular SHAP AND strong Grad-CAM focus
   - "image-driven"  : image dominates the prediction
   - "tabular-driven": tabular dominates the prediction
   - "both-weak"     : neither explanation is pronounced

This file produces:
   * a per-sample dataframe (CSV) with all measures
   * a scatter plot (SHAP magnitude vs Grad-CAM magnitude, colored by reliance)
   * a summary of how often each modality drives the decision
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import cfg
from src.xai import compute_gradcam


EPS = 1e-8


@torch.no_grad()
def _modality_ablation(
    model: nn.Module,
    img: torch.Tensor,
    tab: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Returns (p_full, p_tab_only, p_img_only) sigmoid probabilities for one sample.
    Zeroing a modality's input isolates the other modality's contribution.
    """
    zero_img = torch.zeros_like(img)
    zero_tab = torch.zeros_like(tab)

    p_full = torch.sigmoid(model(image=img, tabular=tab).squeeze()).item()
    p_tab  = torch.sigmoid(model(image=zero_img, tabular=tab).squeeze()).item()
    p_img  = torch.sigmoid(model(image=img, tabular=zero_tab).squeeze()).item()
    return p_full, p_tab, p_img


def analyze_complementarity(
    model: nn.Module,
    test_loader,
    explainer,
    target_layer: nn.Module,
    device: torch.device,
    n_samples: int = 200,
    shap_nsamples: int = 128,
) -> Dict:
    """
    Run the joint SHAP + Grad-CAM analysis over up to n_samples test cases.

    Parameters
    ----------
    explainer : a fitted shap.KernelExplainer wrapping the tabular branch
                (from xai.make_shap_explainer).
    target_layer : denseblock4 of the image branch (for Grad-CAM).

    Returns dict with:
       "per_sample": list of dicts (one per sample)
       "summary":    dict of aggregate statistics
    """
    model.eval()
    rows: List[Dict] = []
    seen = 0

    for batch in test_loader:
        bsz = len(batch["image"])
        for i in range(bsz):
            if seen >= n_samples:
                break

            img = batch["image"][i:i+1].to(device)
            tab = batch["tabular"][i:i+1].to(device)
            label = int(batch["label_binary"][i].item())

            # 1. faithful modality ablation
            p_full, p_tab, p_img = _modality_ablation(model, img, tab, device)
            d_tab = abs(p_full - p_img)   # drop when tabular removed (img stays)
            d_img = abs(p_full - p_tab)   # drop when image removed (tab stays)
            img_reliance = d_img / (d_img + d_tab + EPS)

            # 2. SHAP magnitude on the 4 tabular features
            tab_np = tab.cpu().numpy()
            shap_v = explainer.shap_values(tab_np, nsamples=shap_nsamples)
            shap_v = np.asarray(shap_v).reshape(-1)
            shap_mag = float(np.abs(shap_v).sum())

            # 3. Grad-CAM magnitude (mean activation of normalized heatmap)
            heatmap = compute_gradcam(model, img, tab, target_layer)
            gcam_mag = float(heatmap.mean())

            rows.append({
                "label": label,
                "p_full": p_full,
                "p_tab_only": p_tab,
                "p_img_only": p_img,
                "d_tab": d_tab,
                "d_img": d_img,
                "img_reliance": img_reliance,
                "shap_mag": shap_mag,
                "gcam_mag": gcam_mag,
            })
            seen += 1
        if seen >= n_samples:
            break

    summary = _summarize(rows)
    return {"per_sample": rows, "summary": summary}


def _summarize(rows: List[Dict]) -> Dict:
    """Aggregate per-sample measures + assign agreement categories by median split."""
    if not rows:
        return {}
    shap_mag = np.array([r["shap_mag"] for r in rows])
    gcam_mag = np.array([r["gcam_mag"] for r in rows])
    img_rel  = np.array([r["img_reliance"] for r in rows])

    shap_med = float(np.median(shap_mag))
    gcam_med = float(np.median(gcam_mag))

    cats = {"both-strong": 0, "image-driven": 0, "tabular-driven": 0, "both-weak": 0}
    for r in rows:
        strong_tab = r["shap_mag"] >= shap_med
        strong_img = r["gcam_mag"] >= gcam_med
        if strong_tab and strong_img:
            cats["both-strong"] += 1
        elif strong_img and not strong_tab:
            cats["image-driven"] += 1
        elif strong_tab and not strong_img:
            cats["tabular-driven"] += 1
        else:
            cats["both-weak"] += 1
        r["category"] = _category_of(r, shap_med, gcam_med)

    # Pearson correlation between the two explanation magnitudes
    if len(rows) > 2 and shap_mag.std() > 0 and gcam_mag.std() > 0:
        corr = float(np.corrcoef(shap_mag, gcam_mag)[0, 1])
    else:
        corr = float("nan")

    return {
        "n": len(rows),
        "mean_img_reliance": float(img_rel.mean()),
        "median_img_reliance": float(np.median(img_rel)),
        "shap_mag_median": shap_med,
        "gcam_mag_median": gcam_med,
        "shap_gcam_corr": corr,
        "categories": cats,
    }


def _category_of(r: Dict, shap_med: float, gcam_med: float) -> str:
    strong_tab = r["shap_mag"] >= shap_med
    strong_img = r["gcam_mag"] >= gcam_med
    if strong_tab and strong_img:
        return "both-strong"
    if strong_img and not strong_tab:
        return "image-driven"
    if strong_tab and not strong_img:
        return "tabular-driven"
    return "both-weak"


def save_per_sample_csv(result: Dict, path: Path) -> None:
    """Write per-sample measures to CSV (no pandas dependency required)."""
    rows = result["per_sample"]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[XAI] Complementarity per-sample CSV saved -> {path}")


def plot_complementarity_scatter(result: Dict, path: Path) -> None:
    """
    Scatter of SHAP magnitude (tabular) vs Grad-CAM magnitude (image),
    each point colored by image_reliance. Median lines split the quadrants
    that define the agreement categories.
    """
    rows = result["per_sample"]
    summary = result["summary"]
    if not rows:
        return

    shap_mag = np.array([r["shap_mag"] for r in rows])
    gcam_mag = np.array([r["gcam_mag"] for r in rows])
    img_rel  = np.array([r["img_reliance"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(shap_mag, gcam_mag, c=img_rel, cmap="coolwarm",
                    vmin=0, vmax=1, s=28, edgecolor="k", linewidth=0.3)
    ax.axvline(summary["shap_mag_median"], color="gray", ls="--", lw=1)
    ax.axhline(summary["gcam_mag_median"], color="gray", ls="--", lw=1)
    ax.set_xlabel("Magnitudo SHAP tabular  (sum|phi|)")
    ax.set_ylabel("Magnitudo Grad-CAM citra  (rata-rata heatmap)")
    ax.set_title("Komplementaritas SHAP (tabular) vs Grad-CAM (citra)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Ketergantungan pada citra (image reliance)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[XAI] Complementarity scatter saved -> {path}")


def print_summary(result: Dict) -> None:
    s = result["summary"]
    if not s:
        print("[XAI] No samples analyzed.")
        return
    print("\n[XAI] COMPLEMENTARITY SUMMARY")
    print(f"  Samples analyzed       : {s['n']}")
    print(f"  Mean image reliance    : {s['mean_img_reliance']:.3f} "
          f"(1.0 = decision fully image-driven)")
    print(f"  SHAP-GradCAM magnitude correlation : {s['shap_gcam_corr']:.3f}")
    print(f"  Category counts:")
    for k, v in s["categories"].items():
        pct = 100.0 * v / s["n"]
        print(f"    {k:<15} {v:4d}  ({pct:5.1f}%)")

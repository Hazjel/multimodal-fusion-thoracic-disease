"""C7 deployment prototype for canonical multimodal scenario S3.

The app never loads legacy checkpoints. Before C7 has produced a canonical
deployment manifest, it starts in a disabled informational state.
"""
from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import cfg
from src.data.dataset import get_transforms
from src.models.architectures import MultimodalFusion
from src.protocol.contracts import file_sha256, read_json
from src.protocol.stages import load_frozen_protocol
from src.xai import compute_gradcam, make_shap_explainer, overlay_gradcam


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_NAMES = ["Age", "Gender", "View Position", "Follow-up #"]


def _discover_manifest() -> Optional[Path]:
    explicit = os.environ.get("NIH_DEPLOYMENT_MANIFEST")
    if explicit:
        return Path(explicit)
    candidates = sorted(
        (
            path
            for path in cfg.paths.canonical_dir.glob("*/deployment/deployment_manifest.json")
            if (path.parent / "_REFIT_SUCCESS").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _verify_file(path: Path, expected_checksum: str) -> None:
    actual = file_sha256(path)
    if actual != expected_checksum:
        raise RuntimeError(f"Deployment artifact checksum mismatch: {path}")


def _manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


@lru_cache(maxsize=1)
def load_deployment() -> Tuple[MultimodalFusion, Any, np.ndarray, Dict[str, Any]]:
    manifest_path = _discover_manifest()
    if manifest_path is None:
        raise RuntimeError("C7 canonical deployment artifact is not available yet")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "READY" or manifest.get("scenario") != "S3":
        raise RuntimeError("Deployment manifest is not a READY canonical S3 artifact")
    protocol_path = _manifest_path(manifest_path, manifest["protocol_path"])
    protocol = load_frozen_protocol(protocol_path.parent)
    if manifest.get("protocol_hash") != protocol.get("protocol_hash"):
        raise RuntimeError("Deployment manifest protocol hash mismatch")

    refit_index_path = _manifest_path(manifest_path, manifest["refit_index_path"])
    checkpoint_path = _manifest_path(manifest_path, manifest["checkpoint_path"])
    scaler_path = _manifest_path(manifest_path, manifest["scaler_path"])
    background_path = _manifest_path(manifest_path, manifest["shap_background_path"])
    _verify_file(refit_index_path, manifest["refit_index_checksum"])
    _verify_file(checkpoint_path, manifest["checkpoint_checksum"])
    _verify_file(scaler_path, manifest["scaler_checksum"])
    _verify_file(background_path, manifest["shap_background_checksum"])

    model = MultimodalFusion(
        backbone_name=manifest["backbone"],
        pretraining="none",
        fold=0,
        tabular_input_dim=4,
    ).to(DEVICE)
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload.get("model_state", payload))
    model.eval()
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    background = np.load(background_path).astype(np.float32)
    return model, scaler, background, manifest


def _shap_figure(values: np.ndarray) -> plt.Figure:
    colors = ["#d95f4c" if value >= 0 else "#2a9d6e" for value in values]
    figure, axis = plt.subplots(figsize=(6, 3.2))
    axis.barh(FEATURE_NAMES, values, color=colors)
    axis.axvline(0, color="#777", linewidth=0.8)
    axis.set_xlabel("Conditional SHAP value")
    axis.set_title("Kontribusi metadata bersyarat pada X-ray ini")
    figure.tight_layout()
    return figure


def predict_and_explain(image: Image.Image, age: float, gender: str, view: str, followup: float):
    if image is None:
        raise gr.Error("Unggah citra X-ray terlebih dahulu")
    model, scaler, background, manifest = load_deployment()
    image = image.convert("RGB")
    image_tensor = get_transforms(False)(image).unsqueeze(0).to(DEVICE)
    raw_tabular = np.asarray([[age, 1.0 if gender == "M" else 0.0, 1.0 if view == "PA" else 0.0, followup]], dtype=np.float32)
    scaled_tabular = scaler.transform(raw_tabular).astype(np.float32)
    tabular_tensor = torch.from_numpy(scaled_tabular).to(DEVICE)
    with torch.no_grad():
        score = float(torch.sigmoid(model(image=image_tensor, tabular=tabular_tensor)).item())
    label = "Abnormal" if score >= 0.5 else "Normal"

    heatmap = compute_gradcam(
        model,
        image_tensor,
        tabular_tensor,
        model.get_cam_target_layer(),
    )
    overlay = overlay_gradcam(image, heatmap)
    explainer = make_shap_explainer(
        model,
        background,
        DEVICE,
        fixed_image=image_tensor,
    )
    shap_values = np.asarray(explainer.shap_values(scaled_tabular, nsamples=128))[0]
    result = (
        f"### {label}\n\n"
        f"Skor model untuk kelas Abnormal: **{score:.4f}**  \n"
        "Prototype sistem multimodal S3 yang dievaluasi dalam penelitian. "
        "Skor ini bukan probabilitas klinis terkalibrasi dan bukan diagnosis."
    )
    return overlay, _shap_figure(shap_values), result


manifest_available = _discover_manifest() is not None
status_message = (
    "Artifact deployment canonical ditemukan."
    if manifest_available
    else "C7 belum menghasilkan deployment artifact; analisis dinonaktifkan."
)

with gr.Blocks(title="ChestXAI — Prototype Multimodal S3") as demo:
    gr.Markdown("# ChestXAI\nPrototype sistem multimodal S3 yang dievaluasi dalam penelitian.")
    gr.Markdown(status_message)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Chest X-ray")
            age_input = gr.Slider(0, 100, value=50, step=1, label="Age")
            gender_input = gr.Radio(["F", "M"], value="M", label="Gender")
            view_input = gr.Radio(["AP", "PA"], value="PA", label="View Position")
            followup_input = gr.Number(value=0, minimum=0, label="Follow-up #")
            submit = gr.Button("Analisis", variant="primary", interactive=manifest_available)
        with gr.Column():
            gradcam_output = gr.Image(label="Grad-CAM")
            shap_output = gr.Plot(label="Conditional metadata SHAP")
            result_output = gr.Markdown()
    submit.click(
        predict_and_explain,
        [image_input, age_input, gender_input, view_input, followup_input],
        [gradcam_output, shap_output, result_output],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, show_error=True)

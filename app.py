"""
Gradio demo: Dual XAI — SHAP (tabular) + Grad-CAM (image) pada model multimodal S3.
Jalankan: python app.py
"""
import sys
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import shap
import gradio as gr

sys.path.insert(0, r"D:\TA\nih-multimodal")
from configs.config import cfg
from src.models.architectures import MultimodalFusion

CKPT_PATH   = cfg.paths.checkpoint_dir / "model_s3_multimodal_binary.pt"
BG_PATH     = cfg.paths.xai_dir        / "shap_background.npy"
SCALER_PATH = cfg.paths.checkpoint_dir / "scaler.pkl"

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_NAMES = ["Usia Pasien", "Jenis Kelamin", "Posisi PA", "No. Follow-up"]

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading model S3...")
model = MultimodalFusion(num_classes=1).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False))
model.eval()

print("Loading scaler & background...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
background = np.load(BG_PATH)

ZERO_IMAGE   = torch.zeros(1, 3, cfg.data.image_size, cfg.data.image_size).to(DEVICE)
TARGET_LAYER = model.image_branch.features.denseblock4


@torch.no_grad()
def _predict_tabular_only(X_tab: np.ndarray) -> np.ndarray:
    results = []
    for i in range(0, len(X_tab), 64):
        t   = torch.tensor(X_tab[i:i+64], dtype=torch.float32).to(DEVICE)
        img = ZERO_IMAGE.expand(len(t), -1, -1, -1)
        results.append(torch.sigmoid(model(image=img, tabular=t).squeeze(1)).cpu().numpy())
    return np.concatenate(results)


print("Initializing SHAP KernelExplainer...")
explainer = shap.KernelExplainer(_predict_tabular_only, background)
print(f"Ready on {DEVICE}.\n")


def _gradcam(img_tensor: torch.Tensor, tab_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    acts, grads = [], []
    h1 = TARGET_LAYER.register_forward_hook(lambda _m, _i, o: acts.append(o.detach()))
    h2 = TARGET_LAYER.register_full_backward_hook(lambda _m, _gi, go: grads.append(go[0].detach()))
    logit = model(image=img_tensor, tabular=tab_tensor).squeeze()
    model.zero_grad()
    logit.backward()
    h1.remove(); h2.remove()
    w = grads[0].squeeze(0).mean(dim=(1, 2))
    heatmap = torch.relu((w[:, None, None] * acts[0].squeeze(0)).sum(0)).cpu().numpy()
    return heatmap / heatmap.max() if heatmap.max() > 0 else heatmap


def _overlay(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = np.array(pil_img.resize((cfg.data.image_size, cfg.data.image_size)))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h = cv2.resize(heatmap, (cfg.data.image_size, cfg.data.image_size))
    colored = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)


def predict_and_explain(image, age, gender, view, followup, progress=gr.Progress()):
    if image is None:
        return None, None, _result_idle()

    progress(0, desc="Mempersiapkan input...")
    pil_img = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")

    gender_enc = 1.0 if gender == "Laki-laki" else 0.0
    view_enc   = 1.0 if view == "PA (Frontal)" else 0.0
    raw_tab    = np.array([[float(age), gender_enc, view_enc, float(followup)]], dtype=np.float32)
    tab_scaled = scaler.transform(raw_tab).astype(np.float32)

    img_tensor = IMG_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    tab_tensor = torch.tensor(tab_scaled, dtype=torch.float32).to(DEVICE)

    progress(0.2, desc="Menjalankan prediksi model...")
    with torch.no_grad():
        prob = torch.sigmoid(model(image=img_tensor, tabular=tab_tensor)).item()
    pred = "Abnormal" if prob >= 0.5 else "Normal"

    progress(0.4, desc="Menghitung Grad-CAM...")
    heatmap     = _gradcam(img_tensor, tab_tensor)
    overlay_img = _overlay(pil_img, heatmap)

    progress(0.6, desc="Menghitung SHAP values (30-60 detik)...")
    shap_vals = explainer.shap_values(tab_scaled, nsamples=128)[0]

    progress(0.9, desc="Membuat visualisasi...")
    shap_fig = _make_shap_figure(shap_vals)

    progress(1.0, desc="Selesai.")
    return overlay_img, shap_fig, _result_card(pred, prob, age, gender, view, followup)


# ── Result card (right panel) ──────────────────────────────────────────────────
def _result_idle() -> str:
    return """
<div style="font-family:'Nunito',system-ui,sans-serif; padding:24px;
            background:#F9F8F5; border:1.5px dashed #E8E4DC; border-radius:12px;
            text-align:center; color:#B5B0A9; font-size:13px; line-height:1.7;">
  <div style="font-size:28px; margin-bottom:10px; opacity:0.5;">&#10697;</div>
  Upload gambar X-ray dan isi data pasien,<br>lalu klik <strong style="color:#7A746E;">Analisis</strong>
</div>"""


def _result_card(pred: str, prob: float, age, gender, view, followup) -> str:
    is_abn    = pred == "Abnormal"
    acc       = "#E05A47" if is_abn else "#2A9D6E"
    acc_bg    = "#FEF0EE" if is_abn else "#EDFAF5"
    acc_bd    = "#FBCFC9" if is_abn else "#A7F3D0"
    bar_abn   = int(prob * 100)
    bar_nor   = 100 - bar_abn
    g_disp    = "Laki-laki" if "Laki" in gender else "Perempuan"
    v_disp    = "PA (Frontal)" if "PA" in view else "AP (Terlentang)"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
.rc {{ font-family:'Nunito',system-ui,sans-serif; color:#1A1714; }}
</style>
<div class="rc">

  <!-- Diagnosis header -->
  <div style="background:{acc_bg}; border:1px solid {acc_bd}; border-radius:12px;
              padding:16px 18px; margin-bottom:10px;">
    <div style="font-size:10px; font-weight:700; color:{acc}; text-transform:uppercase;
                letter-spacing:1.5px; margin-bottom:6px; opacity:0.8;">Hasil Prediksi</div>
    <div style="display:flex; align-items:baseline; justify-content:space-between;">
      <div style="font-size:28px; font-weight:800; color:{acc}; letter-spacing:-0.5px;">{pred}</div>
      <div style="text-align:right;">
        <div style="font-size:22px; font-weight:800; color:{acc};">{prob:.1%}</div>
        <div style="font-size:10px; color:{acc}; opacity:0.7; margin-top:1px;">probabilitas abnormal</div>
      </div>
    </div>
  </div>

  <!-- Confidence bars -->
  <div style="background:#FFFFFF; border:1px solid #E8E4DC; border-radius:12px;
              padding:14px 16px; margin-bottom:10px;">
    <div style="font-size:10px; font-weight:700; color:#B5B0A9; text-transform:uppercase;
                letter-spacing:1.2px; margin-bottom:12px;">Distribusi Kepercayaan</div>
    <div style="margin-bottom:10px;">
      <div style="display:flex; justify-content:space-between; font-size:12px;
                  font-weight:600; margin-bottom:5px;">
        <span style="color:#E05A47;">Abnormal</span>
        <span style="color:#E05A47;">{bar_abn}%</span>
      </div>
      <div style="height:8px; background:#FEF0EE; border-radius:50px; overflow:hidden;">
        <div style="width:{bar_abn}%; height:100%; background:#E05A47; border-radius:50px;
                    transition:width 0.6s ease;"></div>
      </div>
    </div>
    <div>
      <div style="display:flex; justify-content:space-between; font-size:12px;
                  font-weight:600; margin-bottom:5px;">
        <span style="color:#2A9D6E;">Normal</span>
        <span style="color:#2A9D6E;">{bar_nor}%</span>
      </div>
      <div style="height:8px; background:#EDFAF5; border-radius:50px; overflow:hidden;">
        <div style="width:{bar_nor}%; height:100%; background:#2A9D6E; border-radius:50px;
                    transition:width 0.6s ease;"></div>
      </div>
    </div>
  </div>

  <!-- Patient data -->
  <div style="background:#FFFFFF; border:1px solid #E8E4DC; border-radius:12px;
              padding:14px 16px; margin-bottom:10px;">
    <div style="font-size:10px; font-weight:700; color:#B5B0A9; text-transform:uppercase;
                letter-spacing:1.2px; margin-bottom:12px;">Data Klinis Pasien</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
      <div style="background:#F9F8F5; border-radius:8px; padding:10px 12px;">
        <div style="font-size:10px; color:#B5B0A9; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.8px; margin-bottom:3px;">Usia</div>
        <div style="font-size:16px; font-weight:800; color:#1A1714;">{int(age)}
          <span style="font-size:11px; font-weight:500; color:#B5B0A9;">thn</span>
        </div>
      </div>
      <div style="background:#F9F8F5; border-radius:8px; padding:10px 12px;">
        <div style="font-size:10px; color:#B5B0A9; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.8px; margin-bottom:3px;">Kelamin</div>
        <div style="font-size:13px; font-weight:700; color:#1A1714;">{g_disp}</div>
      </div>
      <div style="background:#F9F8F5; border-radius:8px; padding:10px 12px;">
        <div style="font-size:10px; color:#B5B0A9; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.8px; margin-bottom:3px;">Posisi</div>
        <div style="font-size:12px; font-weight:700; color:#1A1714;">{v_disp}</div>
      </div>
      <div style="background:#F9F8F5; border-radius:8px; padding:10px 12px;">
        <div style="font-size:10px; color:#B5B0A9; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.8px; margin-bottom:3px;">Follow-up</div>
        <div style="font-size:16px; font-weight:800; color:#1A1714;">#{int(followup)}</div>
      </div>
    </div>
  </div>

  <!-- Disclaimer -->
  <div style="font-size:11px; color:#B5B0A9; line-height:1.6; padding:0 2px;">
    Sistem pendukung keputusan klinis. Diagnosis akhir merupakan wewenang dokter.
  </div>

</div>"""


# ── SHAP figure ────────────────────────────────────────────────────────────────
def _make_shap_figure(shap_vals: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F9F8F5")

    colors = ["#E05A47" if v > 0 else "#2A9D6E" for v in shap_vals]
    ax.barh(FEATURE_NAMES, shap_vals, color=colors, height=0.44,
            edgecolor="#FFFFFF", linewidth=0, alpha=0.9)

    max_abs = max(abs(v) for v in shap_vals) if any(shap_vals) else 1
    for i, (val, c) in enumerate(zip(shap_vals, colors)):
        offset = max_abs * 0.025 if val >= 0 else -max_abs * 0.025
        ha     = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.4f}", va="center", ha=ha,
                fontsize=9, fontweight="700", color=c)

    ax.axvline(0, color="#E8E4DC", linewidth=1.2)
    ax.set_xlabel("SHAP Value", fontsize=9, color="#7A746E", labelpad=6)
    ax.set_title("Kontribusi Fitur — SHAP KernelExplainer",
                 fontsize=10, fontweight="700", color="#1A1714", pad=10)
    ax.tick_params(colors="#7A746E", labelsize=9)
    for s in ax.spines.values():
        s.set_edgecolor("#E8E4DC")
        s.set_linewidth(0.8)
    ax.annotate("Abnormal", xy=(1, -0.28), xycoords="axes fraction",
                ha="right", fontsize=8, color="#E05A47", fontweight="600")
    ax.annotate("Normal", xy=(0, -0.28), xycoords="axes fraction",
                ha="left",  fontsize=8, color="#2A9D6E", fontweight="600")
    fig.tight_layout(pad=1.2)
    return fig


# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700;800&display=swap');

:root, .light {
    --bg:        #F3F0EB;
    --surface:   #FFFFFF;
    --surface2:  #F9F8F5;
    --border:    #E8E4DC;
    --text:      #1A1714;
    --muted:     #7A746E;
    --faint:     #B5B0A9;
    --coral:     #E05A47;
    --coral-h:   #C84E3D;
    --coral-bg:  #FEF0EE;
    --teal:      #2A9D6E;
    --shad:      0 1px 3px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.04);

    /* Gradio Base theme tokens */
    --color-accent:                          #E05A47 !important;
    --color-accent-soft:                     #FEF0EE !important;
    --button-primary-background-fill:        #E05A47 !important;
    --button-primary-background-fill-hover:  #C84E3D !important;
    --button-primary-text-color:             #FFFFFF !important;
    --button-secondary-background-fill:      transparent !important;
    --button-secondary-border-color:         #E8E4DC !important;
    --button-secondary-text-color:           #7A746E !important;
    --block-background-fill:                 #FFFFFF !important;
    --block-border-color:                    #E8E4DC !important;
    --block-radius:                          14px !important;
    --block-label-text-color:                #B5B0A9 !important;
    --block-title-text-color:                #B5B0A9 !important;
    --background-fill-primary:               #F3F0EB !important;
    --background-fill-secondary:             #F9F8F5 !important;
    --border-color-primary:                  #E8E4DC !important;
    --border-color-accent:                   #E05A47 !important;
    --input-background-fill:                 #F9F8F5 !important;
    --input-border-color:                    #E8E4DC !important;
    --input-radius:                          10px !important;
    --button-large-radius:                   50px !important;
    --button-small-radius:                   50px !important;
    --shadow-drop:                           0 1px 3px rgba(0,0,0,0.07) !important;
    --shadow-spread:                         0 0 0 1px rgba(0,0,0,0.04) !important;
    --body-text-color:                       #1A1714 !important;
    --body-text-color-subdued:               #7A746E !important;
    --font:                                  Nunito, system-ui, sans-serif !important;
    --font-mono:                             ui-monospace, monospace !important;
    --checkbox-background-color-selected:    #E05A47 !important;
    --radio-circle-color:                    #E05A47 !important;
    --slider-color:                          #E05A47 !important;
    --panel-background-fill:                 #FFFFFF !important;
    --loader-color:                          #E05A47 !important;
}

*, *::before, *::after { box-sizing: border-box; }

body {
    background: var(--bg) !important;
    font-family: 'Nunito', system-ui, sans-serif !important;
    color: var(--text) !important;
}

.gradio-container {
    background: var(--bg) !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 20px 32px !important;
}

footer { display: none !important; }

/* Cards */
.block, .gr-group, .gr-form, div.block {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: var(--shad) !important;
}
.gr-group .block, .gr-form .block {
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* Labels */
label > span, .label-wrap span {
    color: var(--faint) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'Nunito', sans-serif !important;
}

/* Inputs */
input[type="number"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}
input[type="number"]:focus {
    border-color: var(--coral) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(224,90,71,0.15) !important;
}

input[type="range"] { accent-color: #E05A47 !important; }
input[type="radio"] { accent-color: #E05A47 !important; }

.wrap label span, fieldset label span {
    color: var(--text) !important;
    font-size: 14px !important;
    font-family: 'Nunito', sans-serif !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    font-weight: 600 !important;
}

/* Radio / checkbox pill items — fix dark stone background */
fieldset .wrap label,
.gr-radio label,
.gr-checkbox label {
    background: #FFFFFF !important;
    border: 1.5px solid #E8E4DC !important;
    border-radius: 10px !important;
    color: #1A1714 !important;
    transition: border-color 0.15s, background 0.15s !important;
}
fieldset .wrap label:hover {
    border-color: #E05A47 !important;
    background: #FEF0EE !important;
}
fieldset .wrap label:has(input:checked) {
    border-color: #E05A47 !important;
    background: #FEF0EE !important;
}

/* Image */
div[data-testid="image"] .wrap {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* Plot */
.gr-plot, [data-testid="plot"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
}

/* Primary button — coral pill */
button.primary, button[data-testid="primary-btn"] {
    background: var(--coral) !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.2px !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(224,90,71,0.30) !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
}
button.primary:hover {
    background: var(--coral-h) !important;
    box-shadow: 0 6px 18px rgba(224,90,71,0.40) !important;
    transform: translateY(-1px) !important;
}
button.primary:active { transform: translateY(0) !important; }

/* Secondary button */
button.secondary {
    background: transparent !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 50px !important;
    color: var(--muted) !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
}
button.secondary:hover {
    border-color: var(--coral) !important;
    color: var(--coral) !important;
}

/* Accordion */
details summary {
    background: var(--surface2) !important;
    color: var(--muted) !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* Markdown */
.prose h3, .md h3 {
    font-family: 'Nunito', sans-serif !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    color: var(--faint) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #ECEAE4 !important;
    margin-bottom: 12px !important;
}
.prose h4, .md h4 {
    font-size: 11px !important;
    color: var(--faint) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 700 !important;
}
.prose p, .md p { color: var(--muted) !important; font-size: 13px !important; line-height:1.65 !important; }
.prose strong, .md strong { color: var(--text) !important; }
.prose li, .md li { color: var(--muted) !important; font-size:13px !important; }

/* Progress */
.progress-bar { background: var(--coral) !important; border-radius: 50px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 50px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* Section label style */
.sec-label {
    font-family: 'Nunito', sans-serif;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #B5B0A9;
    padding: 0 2px 6px;
}
"""

# ── Header ─────────────────────────────────────────────────────────────────────
HEADER_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
</style>
<div style="background:#FFFFFF; border-bottom:1px solid #E8E4DC; margin-bottom:12px;
            padding:14px 24px; display:flex; align-items:center;
            justify-content:space-between; flex-wrap:wrap; gap:12px;">

  <!-- Logo + name -->
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="width:36px; height:36px; background:#FEF0EE; border-radius:10px;
                display:flex; align-items:center; justify-content:center;
                flex-shrink:0; font-size:18px;">
      &#x2764;&#xFE0E;
    </div>
    <div>
      <div style="font-family:'Nunito',sans-serif; font-size:16px; font-weight:800;
                  color:#1A1714; line-height:1.1;">ChestXAI</div>
      <div style="font-family:'Nunito',sans-serif; font-size:11px;
                  color:#B5B0A9; margin-top:1px; font-weight:600;">
        Sistem Pendukung Keputusan Klinis
      </div>
    </div>
  </div>

  <!-- Nav pills — matching Dribbble style -->
  <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
    <div style="background:#E05A47; color:#FFFFFF; font-family:'Nunito',sans-serif;
                font-size:13px; font-weight:700; padding:7px 18px; border-radius:50px;
                user-select:none;">
      Analisis
    </div>
    <div style="color:#7A746E; font-family:'Nunito',sans-serif; font-size:13px;
                font-weight:600; padding:7px 16px; border-radius:50px;
                border:1px solid #E8E4DC; user-select:none;">
      DenseNet-121
    </div>
    <div style="color:#7A746E; font-family:'Nunito',sans-serif; font-size:13px;
                font-weight:600; padding:7px 16px; border-radius:50px;
                border:1px solid #E8E4DC; user-select:none;">
      NIH ChestX-ray14
    </div>
  </div>

  <!-- Right info -->
  <div style="font-family:'Nunito',sans-serif; font-size:12px; color:#B5B0A9;
              font-weight:600; text-align:right; line-height:2;">
    <div>S3 Binary &nbsp;&middot;&nbsp; AUC&nbsp;0.7449</div>
    <div>SHAP &amp; Grad-CAM</div>
  </div>

</div>
"""

# ── Section labels ─────────────────────────────────────────────────────────────
LBL_INPUT = """<div class="sec-label" style="font-family:'Nunito',sans-serif; font-size:11px;
    font-weight:800; text-transform:uppercase; letter-spacing:1.5px; color:#B5B0A9;
    padding:0 2px 8px;">Data Pasien</div>"""

LBL_VISUAL = """<div style="font-family:'Nunito',sans-serif; font-size:11px;
    font-weight:800; text-transform:uppercase; letter-spacing:1.5px; color:#B5B0A9;
    padding:0 2px 8px;">Visualisasi Grad-CAM
    <span style="font-size:10px; font-weight:600; color:#E8E4DC; margin-left:4px; letter-spacing:0;">
      &mdash; Area saliency pada X-ray
    </span>
    </div>"""

LBL_RESULT = """<div style="font-family:'Nunito',sans-serif; font-size:11px;
    font-weight:800; text-transform:uppercase; letter-spacing:1.5px; color:#B5B0A9;
    padding:0 2px 8px;">Hasil &amp; SHAP</div>"""

# ── Gradio Interface ───────────────────────────────────────────────────────────
with gr.Blocks(title="ChestXAI — Dual XAI", fill_width=True) as demo:

    gr.HTML(HEADER_HTML)

    # ── 3-column layout matching Dribbble ──────────────────────────────────────
    with gr.Row(equal_height=False):

        # LEFT: Patient form (like Dribbble's left diagnostic panel)
        with gr.Column(scale=1, min_width=260):
            gr.HTML(LBL_INPUT)
            img_input = gr.Image(
                type="pil", label="Gambar Chest X-Ray",
                height=200,
            )
            with gr.Group():
                age_input = gr.Slider(
                    minimum=0, maximum=100, value=50, step=1,
                    label="Usia Pasien (tahun)",
                )
                with gr.Row():
                    gender_input = gr.Radio(
                        choices=["Laki-laki", "Perempuan"],
                        value="Laki-laki", label="Jenis Kelamin",
                    )
                with gr.Row():
                    view_input = gr.Radio(
                        choices=["PA (Frontal)", "AP (Terlentang)"],
                        value="PA (Frontal)", label="Posisi Gambar",
                    )
                followup_input = gr.Number(
                    value=0, minimum=0, maximum=100,
                    label="Nomor Follow-up",
                )
            submit_btn = gr.Button("Analisis Pasien", variant="primary", size="lg")
            with gr.Accordion("Panduan", open=False):
                gr.Markdown("""
                1. **Upload** gambar chest X-ray
                2. **Isi** data klinis pasien
                3. Klik **Analisis Pasien** — tunggu 30-60 detik
                4. Baca hasil di panel kanan

                Gunakan posisi **PA (frontal)** untuk hasil terbaik.
                """)

        # CENTER: Grad-CAM visual (like Dribbble's center body diagram)
        with gr.Column(scale=1, min_width=320):
            gr.HTML(LBL_VISUAL)
            gradcam_out = gr.Image(
                label="Grad-CAM Heatmap Overlay",
                height=520,
            )

        # RIGHT: Results (like Dribbble's right diagnosis panel)
        with gr.Column(scale=1, min_width=280):
            gr.HTML(LBL_RESULT)
            result_html = gr.HTML(_result_idle())
            shap_out = gr.Plot(label="SHAP KernelExplainer")

    submit_btn.click(
        fn=predict_and_explain,
        inputs=[img_input, age_input, gender_input, view_input, followup_input],
        outputs=[gradcam_out, shap_out, result_html],
    )


if __name__ == "__main__":
    demo.launch(
        share=False, server_port=7860, show_error=True,
        theme=gr.themes.Base(
            primary_hue="orange",
            secondary_hue="stone",
            neutral_hue="stone",
        ),
        css=CSS,
    )

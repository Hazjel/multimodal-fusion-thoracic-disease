# Multimodal Deep Learning + Dual XAI untuk Klasifikasi Penyakit Dada

**Tugas Akhir — Teknik Informatika, Telkom University**  
**Dataset:** NIH ChestX-ray14 (112.120 gambar, 30.805 pasien)

## Deskripsi

Penelitian ini mengembangkan arsitektur **intermediate fusion multimodal** yang menggabungkan:
- **Tabular Branch (MLP)** — memproses metadata klinis pasien (usia, jenis kelamin, posisi pengambilan gambar, jumlah follow-up)
- **Image Branch (DenseNet-121)** — memproses gambar X-ray dada 224×224

Model dianalisis menggunakan **Dual XAI**:
- **SHAP KernelExplainer** — menjelaskan kontribusi fitur tabular klinis secara kuantitatif
- **Grad-CAM** — memvisualisasikan area gambar X-ray yang paling berpengaruh terhadap prediksi

## Research Gap

| Gap | Deskripsi |
|-----|-----------|
| Gap 1 | Sebagian besar model masih unimodal (gambar saja) |
| Gap 2 | Model multimodal + dual XAI sudah ada, namun domain osteoporosis (dataset privat) |
| Gap 3 | Model multimodal penyakit dada sudah ada, namun XAI hanya Grad-CAM — tabular tidak dijelaskan |

## Skenario Pengujian

| Skenario | Input | Output | AUC-ROC |
|----------|-------|--------|---------|
| S1 — MLP Tabular | Tabular saja | Biner | 0.6515 |
| S2 — DenseNet-121 | Gambar saja | Biner | 0.6953 |
| S3 — Multimodal Fusion | Tabular + Gambar | Biner | 0.7050 |
| S4 — Multimodal Fusion | Tabular + Gambar | Multi-label (14 penyakit) | 0.7694 (Macro) |

## Arsitektur Model

```
Tabular (4 fitur) → MLP → vektor 128
Gambar (224×224)  → DenseNet-121 → vektor 512
                            ↓
                    Concatenate → 640
                            ↓
               Fusion Head (640→256→128→output)
```

## Struktur Repository

```
nih-multimodal/
├── configs/
│   └── config.py              # Hyperparameter dan konfigurasi global
├── src/
│   ├── data/                  # Dataset loader dan preprocessing
│   ├── models/                # Arsitektur model (TabularBranch, ImageBranch, MultimodalFusion)
│   ├── training/              # Training loop dan evaluasi
│   ├── evaluation/            # Metrik evaluasi
│   └── xai/                   # SHAP dan Grad-CAM utilities
├── notebooks/
│   ├── 01_eksplorasi.ipynb    # Eksplorasi dataset NIH ChestX-ray14
│   ├── 02_preprocessing.ipynb # Preprocessing tabular dan split data
│   ├── 03_skenario1_mlp.ipynb # S1: MLP tabular only
│   ├── 04_skenario2_cnn.ipynb # S2: DenseNet-121 image only
│   ├── 05_skenario3_4_multimodal.ipynb  # S3 & S4: Multimodal fusion
│   └── 06_shap_gradcam.ipynb  # Dual XAI: SHAP + Grad-CAM
├── data/
│   └── processed/             # Split indices CSV (train/val/test)
├── results/
│   ├── figures/               # Visualisasi hasil dan arsitektur
│   ├── tables/                # CSV hasil evaluasi per skenario
│   └── xai/                   # SHAP values dan Grad-CAM outputs
└── models/
    └── scaler_tabular.pkl     # StandardScaler yang sudah di-fit
```

## Setup

### 1. Install dependencies
```bash
pip install torch torchvision
pip install shap==0.51.0
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 2. Download dataset
Dataset NIH ChestX-ray14 tersedia di:
- [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [NIH Official](https://nihcc.app.box.com/v/ChestXray-NIHCC)

Letakkan gambar di folder sesuai konfigurasi di `configs/config.py`.

### 3. Jalankan notebook secara berurutan
```
01 → 02 → 03 → 04 → 05 → 06
```

> **Catatan:** Model checkpoint (`.pt`) tidak disertakan di repo karena ukurannya besar (~30MB each). Jalankan notebook 03–05 untuk melatih ulang model.

## Hasil XAI

### SHAP — Kontribusi Fitur Tabular
| Fitur | Mean |SHAP| |
|-------|-------------|
| Patient Age | 0.0083 |
| Follow-up Number | 0.0081 |
| Gender | 0.0062 |
| View Position (PA) | 0.0060 |

### Grad-CAM
Heatmap divisualisasikan dari layer `denseblock4` DenseNet-121, menunjukkan area paru-paru yang paling berpengaruh terhadap prediksi.

## Hardware

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6.4 GB VRAM)
- Framework: PyTorch 2.x + AMP (Automatic Mixed Precision)

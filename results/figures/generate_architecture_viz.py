"""
Visualisasi arsitektur Intermediate Fusion Multimodal
untuk Tugas Akhir — NIH ChestX-ray14
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# ── Warna ──
C_INPUT   = "#2c3e50"
C_TAB     = "#2980b9"
C_IMG     = "#8e44ad"
C_FUSED   = "#27ae60"
C_FUSION  = "#16a085"
C_OUT     = "#c0392b"
C_ARROW   = "#7f8c8d"
C_VEC_TAB = "#d6eaf8"
C_VEC_IMG = "#e8daef"
C_VEC_FUS = "#d5f5e3"

def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=10, alpha=0.9, radius=0.3):
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.05,rounding_size={radius}",
                           facecolor=color, edgecolor="white",
                           linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(fancy)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            color="white", fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.25, sublabel,
                ha="center", va="center", fontsize=8,
                color="white", alpha=0.85, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=15),
                zorder=2)

def vector_bar(ax, x, y, n, color, label_top=None, label_dim=None, width=0.6, height=0.5):
    """Visualisasi vektor sebagai bar berwarna-warni."""
    cols = plt.cm.Blues(np.linspace(0.3, 0.9, n)) if color == "blue" else \
           plt.cm.Purples(np.linspace(0.3, 0.9, n)) if color == "purple" else \
           plt.cm.Greens(np.linspace(0.3, 0.9, n))
    bar_w = width / n
    for i in range(n):
        rect = plt.Rectangle((x - width/2 + i*bar_w, y - height/2),
                              bar_w, height,
                              facecolor=cols[i], edgecolor="white", linewidth=0.3, zorder=3)
        ax.add_patch(rect)
    # border
    border = plt.Rectangle((x - width/2, y - height/2), width, height,
                            fill=False, edgecolor="white", linewidth=1.5, zorder=4)
    ax.add_patch(border)
    if label_top:
        ax.text(x, y + height/2 + 0.2, label_top,
                ha="center", va="bottom", fontsize=8, color="gray", fontstyle="italic")
    if label_dim:
        ax.text(x, y - height/2 - 0.2, label_dim,
                ha="center", va="top", fontsize=9, color="gray", fontweight="bold")


# ════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════
ax.text(8, 9.5, "Arsitektur Intermediate Fusion Multimodal",
        ha="center", va="center", fontsize=14, fontweight="bold", color=C_INPUT)

# ════════════════════════════════════════════════
# BRANCH TABULAR (kiri)
# ════════════════════════════════════════════════
# Input tabular
box(ax, 2.5, 8.2, 2.8, 0.7, C_TAB, "Data Tabular Klinis",
    "Age, Gender, View, Follow-up", fontsize=9)

arrow(ax, 2.5, 7.85, 2.5, 7.25)

# MLP layer 1
box(ax, 2.5, 7.0, 2.2, 0.55, C_TAB, "Linear(4→64) + BN + ReLU", fontsize=8.5)
arrow(ax, 2.5, 6.72, 2.5, 6.22)

# MLP layer 2
box(ax, 2.5, 6.0, 2.2, 0.55, C_TAB, "Linear(64→128) + BN + ReLU", fontsize=8.5)
arrow(ax, 2.5, 5.72, 2.5, 5.22)

# MLP output
box(ax, 2.5, 5.0, 2.2, 0.55, C_TAB, "Linear(128→128) + ReLU", fontsize=8.5)
arrow(ax, 2.5, 4.72, 2.5, 4.1)

# Vektor tabular
vector_bar(ax, 2.5, 3.75, 24, "blue", label_top="Vektor Fitur Tabular", label_dim="dim = 128")

# Label branch
ax.text(2.5, 8.85, "TABULAR BRANCH (MLP)", ha="center", va="center",
        fontsize=10, color=C_TAB, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.7, 4.55), 3.6, 4.55,
             boxstyle="round,pad=0.1", fill=False,
             edgecolor=C_TAB, linewidth=1.5, linestyle="--", alpha=0.5, zorder=1))

# ════════════════════════════════════════════════
# BRANCH IMAGE (kanan)
# ════════════════════════════════════════════════
# Input image
box(ax, 13.5, 8.2, 2.8, 0.7, C_IMG, "Gambar X-ray Dada",
    "224 × 224 × 3 (RGB)", fontsize=9)

arrow(ax, 13.5, 7.85, 13.5, 7.25)

# DenseNet blocks
box(ax, 13.5, 7.0, 2.5, 0.55, C_IMG, "DenseNet-121 Blocks", "DenseBlock 1–4 + Transition", fontsize=8.5)
arrow(ax, 13.5, 6.72, 13.5, 6.22)

# GAP
box(ax, 13.5, 6.0, 2.5, 0.55, C_IMG, "GlobalAvgPool → 1024", fontsize=8.5)
arrow(ax, 13.5, 5.72, 13.5, 5.22)

# Projection
box(ax, 13.5, 5.0, 2.5, 0.55, C_IMG, "Linear(1024→512) + BN + ReLU", fontsize=8.5)
arrow(ax, 13.5, 4.72, 13.5, 4.1)

# Vektor image
vector_bar(ax, 13.5, 3.75, 24, "purple", label_top="Vektor Fitur Gambar", label_dim="dim = 512")

# Label branch
ax.text(13.5, 8.85, "IMAGE BRANCH (DenseNet-121)", ha="center", va="center",
        fontsize=10, color=C_IMG, fontweight="bold")
ax.add_patch(FancyBboxPatch((11.7, 4.55), 3.6, 4.55,
             boxstyle="round,pad=0.1", fill=False,
             edgecolor=C_IMG, linewidth=1.5, linestyle="--", alpha=0.5, zorder=1))

# ════════════════════════════════════════════════
# CONCATENATE
# ════════════════════════════════════════════════
# Arrows menuju concat
arrow(ax, 2.5, 3.5, 6.8, 2.85, color=C_TAB, lw=2)
arrow(ax, 13.5, 3.5, 9.2, 2.85, color=C_IMG, lw=2)

# Concat box
box(ax, 8.0, 2.6, 3.5, 0.65, C_FUSED,
    "CONCATENATE", "[Vektor Tabular | Vektor Gambar]", fontsize=9.5)

# Vektor gabungan
vector_bar(ax, 8.0, 1.85, 40, "green", label_top="Vektor Gabungan", label_dim="dim = 640  (128 + 512)")

arrow(ax, 8.0, 2.27, 8.0, 1.55)  # dari concat ke vektor

# ════════════════════════════════════════════════
# FUSION HEAD
# ════════════════════════════════════════════════
arrow(ax, 8.0, 1.55, 8.0, 0.85)

# Teks fusion head — di bawah vektor
ax.text(8.0, 0.65, "Fusion Head:  640 → 256 → 128 → num_classes",
        ha="center", va="center", fontsize=9, color=C_FUSION, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d0f0ea", edgecolor=C_FUSION, linewidth=1.5))

# ════════════════════════════════════════════════
# Anotasi "Intermediate Fusion"
# ════════════════════════════════════════════════
ax.annotate("Intermediate Fusion:\nkedua branch diproses\nterpisah → vektor fitur\n→ baru digabungkan",
            xy=(8.0, 2.6), xytext=(5.2, 1.2),
            fontsize=8, color=C_FUSED,
            arrowprops=dict(arrowstyle="->", color=C_FUSED, lw=1),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor=C_FUSED, alpha=0.8))

plt.tight_layout()
out_path = r"D:\TA\nih-multimodal\results\figures\architecture_intermediate_fusion.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Tersimpan: {out_path}")
plt.show()

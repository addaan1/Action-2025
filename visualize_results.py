"""
visualize_results.py
======================
Script untuk menghasilkan visualisasi hasil kompetisi ACTION 2025.
Menghasilkan file: assets/hasil_visualisasi.png

Jalankan sekali sebelum push ke GitHub:
    python visualize_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# ── Konfigurasi ─────────────────────────────────────────────────────────────
SUBMISSION_CSV      = "submission_ensemble_final.csv"
TRAINING_IMGS       = [
    "training_history_efficientnet_b0.png",
    "training_history_mobilenetv3_large_100.png",
    "training_history_resnet34.png",
]
OUTPUT_DIR          = "assets"
OUTPUT_FILE         = os.path.join(OUTPUT_DIR, "hasil_visualisasi.png")

FOOD_CLASSES = [
    'Ayam Bakar', 'Ayam Betutu', 'Ayam Goreng', 'Ayam Pop', 'Bakso',
    'Coto Makassar', 'Gado Gado', 'Gudeg', 'Nasi Goreng', 'Pempek',
    'Rawon', 'Rendang', 'Sate Madura', 'Sate Padang', 'Soto'
]

# Palet warna premium (15 warna berbeda)
PALETTE = [
    "#FF6B6B", "#FF8E53", "#FFD166", "#06D6A0", "#118AB2",
    "#073B4C", "#9B5DE5", "#F15BB5", "#00BBF9", "#00F5D4",
    "#FEE440", "#8338EC", "#FB5607", "#3A86FF", "#FFBE0B",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data submission ─────────────────────────────────────────────────────
df = pd.read_csv(SUBMISSION_CSV)
counts = df['label'].value_counts().reindex(FOOD_CLASSES, fill_value=0)
total  = counts.sum()

# ── Figure utama ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20), facecolor="#0D1117")

# Layout: baris atas (bar chart besar) + baris bawah (3 training plots)
gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[1.4, 1],
    hspace=0.40,
    wspace=0.25,
)

# ════════════════════════════════════════════════════════════════════════════
# Panel 1 – Distribusi Prediksi Label (bar chart horizontal)
# ════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :])   # full top row

y_pos  = np.arange(len(FOOD_CLASSES))
bars   = ax1.barh(
    y_pos, counts.values,
    color=PALETTE,
    edgecolor="none",
    height=0.65,
)

# Annotation nilai + persentase
for bar, val in zip(bars, counts.values):
    pct = val / total * 100
    ax1.text(
        bar.get_width() + 4, bar.get_y() + bar.get_height() / 2,
        f"{val:,}  ({pct:.1f}%)",
        va='center', ha='left',
        color="#E6EDF3", fontsize=11, fontweight='bold',
        fontfamily='monospace',
    )

ax1.set_yticks(y_pos)
ax1.set_yticklabels(FOOD_CLASSES, fontsize=12.5, color="#E6EDF3")
ax1.set_xlabel("Jumlah Prediksi", color="#8B949E", fontsize=12)
ax1.set_xlim(0, counts.max() * 1.28)
ax1.set_facecolor("#161B22")
ax1.tick_params(colors="#8B949E", labelsize=11)
ax1.spines[:].set_visible(False)
ax1.xaxis.set_tick_params(color="#8B949E")

# Grid subtle
ax1.xaxis.grid(True, color="#21262D", linewidth=0.8, linestyle='--')
ax1.set_axisbelow(True)

ax1.set_title(
    "📊  Distribusi Prediksi Label — Ensemble Final (ConvNeXt V2 + Swin Transformer)",
    color="#E6EDF3", fontsize=15, fontweight='bold', pad=16,
)

# Total annotation box
ax1.text(
    0.99, 0.03,
    f"Total: {total:,} gambar   •   15 kelas makanan",
    transform=ax1.transAxes,
    ha='right', va='bottom',
    color="#8B949E", fontsize=10,
    style='italic',
)

# ════════════════════════════════════════════════════════════════════════════
# Panel 2, 3, 4 – Training History dari 3 model (embed gambar PNG)
# ════════════════════════════════════════════════════════════════════════════
model_labels = ["EfficientNet B0", "MobileNet V3 Large", "ResNet34"]

for col_idx, (fname, mlabel) in enumerate(zip(TRAINING_IMGS, model_labels)):
    ax = fig.add_subplot(gs[1, col_idx])
    ax.set_facecolor("#161B22")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_color("#30363D")
    ax.set_title(f"🔸 {mlabel}", color="#E6EDF3", fontsize=12, fontweight='bold', pad=8)

    if os.path.exists(fname):
        img = Image.open(fname)
        ax.imshow(img, aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.text(
            0.5, 0.5,
            f"[{fname}]\ntidak ditemukan",
            ha='center', va='center',
            color="#8B949E", fontsize=10,
            transform=ax.transAxes,
        )

# ── Judul utama & footer ─────────────────────────────────────────────────────
fig.suptitle(
    "ACTION 2025 — Data Mining: Label Discovery Makanan Tradisional Indonesia",
    color="#E6EDF3", fontsize=18, fontweight='bold', y=0.98,
)
fig.text(
    0.5, 0.01,
    "Pipeline: Label Discovery (manual+CLIP) → ConvNeXt V2 + Swin Transformer → Ensemble (Logit Avg + TTA)",
    ha='center', color="#8B949E", fontsize=10, style='italic',
)

# ── Simpan ──────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches='tight',
             facecolor=fig.get_facecolor())
plt.close()
print(f"✅  Visualisasi berhasil disimpan ke: {OUTPUT_FILE}")

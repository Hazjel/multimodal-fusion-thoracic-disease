# Status Hasil Eksperimen — Peta Folder

Dokumen ini jawab: **file mana yang FINAL (masuk naskah) vs EKSPLORASI (validasi internal saja)**.

## FINAL — 3 skenario resmi (sesuai Metodologi.tex, ini yang dipakai di Bab 4)

| Lokasi | Isi |
|---|---|
| `checkpoints/model_s1_best.pt` | S1 — MLP Tabular, AUC 0.6182 |
| `checkpoints/model_s2_best.pt` | S2 — DenseNet-121 (CheXNet), AUC 0.7442 |
| `checkpoints/model_s3_best.pt` | S3 — Multimodal Concat Fusion, AUC 0.7439 |
| `checkpoints/scaler.pkl` | StandardScaler fitur tabular (4 fitur baseline) |
| `tables/predictions_s{1,2,3}.csv` | Prediksi test-set tiap skenario |
| `tables/results_s{1,2,3}.csv` | Metrik (accuracy, F1, AUC) tiap skenario |
| `tables/results_summary.csv` | Ringkasan gabungan S1-S3 |
| `tables/stats_significance.json` | Bootstrap CI + DeLong test final |
| `figures/roc_s{1,2,3}.png`, `s{1,2,3}_*.png` | Kurva ROC & training curve final |
| `xai/*` | SHAP + Grad-CAM + complementarity — semua dari model S3 final |

**Kesimpulan statistik final:** S3 vs S2 tidak signifikan (DeLong p=0.65), S2/S3 signifikan lebih baik dari S1 (p<0.0001). Lihat memory `project_proposal_status` untuk histori revisi.

## EKSPLORASI — validasi internal, TIDAK WAJIB masuk naskah

Folder `exploratory/` — semua percobaan ganti metode/fitur buat cek apakah bisa ngalahin hasil final. Boleh disebut 1 kalimat di Keterbatasan kalau mau, tapi bukan skenario resmi.

| File | Status | Hasil |
|---|---|---|
| `model_s3-gated_best.pt` + `predictions_s3-gated.csv` + `results_s3-gated.csv` | **Selesai** | AUC 0.7449, identik S3 (DeLong p=0.996) |
| `model_s3-attn_best.pt` | **Belum dievaluasi** — training di-kill 2x (I/O terlalu lambat), checkpoint parsial ada tapi belum di-eval | — |
| `model_s1-ext_best.pt` | **Belum dievaluasi** — training di-kill (6-fitur: +visit_count +pixel_spacing_x), checkpoint parsial ada tapi belum di-eval | — |
| `model_s2-attn_best.pt` + `predictions_s2attn.csv` + `results_s2attn.csv` | **Selesai** | AUC 0.7471 vs S2 baseline 0.7442 — **DeLong p=0.0017, SIGNIFIKAN LEBIH BAIK** (pakai CheXNet, sama seperti S2 baseline) |
| `model_s2-eff_best.pt` + `predictions_s2eff.csv` + `results_s2eff.csv` | **Selesai** | AUC 0.7124 vs S2 baseline 0.7442 — DeLong p<0.0001, **SIGNIFIKAN LEBIH JELEK**. Caveat: EfficientNet-B0 cuma bisa ImageNet pretrain (gak ada CheXNet), jadi bukan perbandingan arsitektur murni — kemungkinan besar kalah karena kehilangan CheXNet, bukan karena EfficientNet arsitekturnya jelek |
| `model_s2-eff-tuned_best.pt` + `predictions_s2efftuned.csv` + `results_s2efftuned.csv` | **Selesai** | AUC 0.7228 (freeze_ratio diturunkan 75%→40%, trainable 3.3M→4.5M) vs S2 baseline 0.7442 — masih signifikan lebih jelek (p<0.0001), tapi gap menyempit dari -0.0318 ke -0.0215 (~32%) dibanding S2-eff versi awal. Mengonfirmasi gap terutama dari hilangnya CheXNet pretrain, bukan arsitektur |
| `model_s2-resnet_best.pt` + `predictions_s2resnet.csv` + `results_s2resnet.csv` | **Selesai** | AUC 0.7130 (ResNet-50, ImageNet pretrain, 24.5M params — 5x lebih besar dari EfficientNet-B0) vs S2 baseline 0.7442 — DeLong p<0.0001, **SIGNIFIKAN LEBIH JELEK**, hampir identik ke EfficientNet-B0 awal (0.7124) meski arsitektur & ukuran beda jauh. Memperkuat kesimpulan: gap didominasi hilangnya CheXNet pretrain, bukan pilihan arsitektur backbone |

**Kesimpulan lintas-backbone:** dari 4 backbone dicoba (DenseNet+CheXNet baseline, DenseNet+CheXNet+attention, EfficientNet+ImageNet ×2, ResNet+ImageNet), **satu-satunya yang menang signifikan adalah S2-attn** — kombinasi CheXNet pretrain + attention module. Backbone ImageNet manapun (EfficientNet atau ResNet) kalah signifikan dan hasilnya serupa satu sama lain, menunjukkan pretrain domain-specific (CheXNet) adalah faktor dominan, bukan pilihan arsitektur.

## LEGACY — checkpoint/arsitektur lama, JANGAN DIPAKAI

Folder `legacy/` — dari sesi sebelum sinkronisasi arsitektur (tabular 3-layer, bukan 2-layer sesuai Metodologi.tex) atau scope yang sudah dihapus (S4 multi-label).

| File | Kenapa legacy |
|---|---|
| `model_s1_mlp_tabular.pt`, `model_s2_cnn_image.pt`, `model_s3_multimodal_binary.pt` | Arsitektur tabular lama (3 hidden layer), sudah digantikan checkpoint FINAL di atas |
| `model_s4_multimodal_multilabel.pt` | Skenario S4 (multi-label 14 penyakit) sudah dihapus dari scope proposal — lihat commit "Hapus skenario S4" |

## Aturan biar gak berantakan lagi

1. **Eksperimen baru** (ganti arsitektur/fitur) → checkpoint & CSV-nya **selalu** disimpan dengan suffix jelas (`-gated`, `-attn`, `-ext`) dan **langsung dipindah** ke `exploratory/` setelah selesai — jangan biarkan nyampur di `checkpoints/`/`tables/` utama.
2. **Update dokumen ini** tiap kali ada eksperimen baru selesai/dibatalkan.
3. Kalau ragu file mana yang final — cek dokumen ini dulu sebelum baca kode.

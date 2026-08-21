# Indeks Notebook

Folder ini memisahkan notebook historis dari laporan canonical. Seluruh
notebook yang sudah ada dibuat sebelum Protocol v1.0.0 dibekukan, sehingga
tidak digunakan untuk menjalankan atau menyatakan hasil canonical.

## Struktur

```text
notebooks/
├── archive/pre_protocol/  # implementasi dan output sebelum protocol freeze
├── exploratory/           # eksperimen tambahan di luar skenario canonical
└── canonical_reports/     # laporan read-only dari artifact canonical
```

## Status notebook yang dipertahankan

| Notebook | Status | Alasan |
|---|---|---|
| `01_dataset_exploration_pre_protocol.ipynb` | Archive | EDA historis dan masih mengenal official train/test lists |
| `02_row_split_preprocessing_legacy.ipynb` | Legacy | Menggunakan `train_test_split`, bukan patient-level manifest canonical |
| `03_s1_mlp_pre_protocol.ipynb` | Archive | Training S1 dan checkpoint sebelum protocol freeze |
| `04_s2_densenet_pre_protocol.ipynb` | Archive | DenseNet/CheXNet dan checkpoint sebelum C2 canonical |
| `05_s3_multimodal_pre_protocol.ipynb` | Archive | Fusion dan training policy lama |
| `06_xai_shap_gradcam_pre_protocol.ipynb` | Archive | SHAP memakai zero-image reference dan bukan OOF fold-specific protocol |
| `07_attention_experiment.ipynb` | Exploratory | Attention tidak termasuk kandidat canonical |
| `08_efficientnet_experiment.ipynb` | Exploratory | Eksperimen backbone sebelum screening yang setara |
| `09_efficientnet_tuned_experiment.ipynb` | Exploratory | Model tuning berdasarkan hasil pilot tidak termasuk protocol frozen |
| `10_resnet_experiment.ipynb` | Exploratory | Eksperimen ResNet sebelum C2 canonical |

## Aturan penggunaan

- Jangan mengambil angka notebook archive/exploratory untuk tabel utama BAB IV.
- Jangan melanjutkan training canonical dari checkpoint notebook lama.
- Jangan membuka official NIH test melalui notebook sebelum C7.
- Jangan mengubah hasil canonical karena AUC notebook lama.
- Gunakan notebook lama hanya untuk riwayat pengembangan atau konteks
  exploratory yang diberi label jelas.
- Jalankan eksperimen canonical melalui `run_experiment.py` dan registry
  protocol aktif.

Pemindahan notebook hanya mengubah lokasi dan nama file. Isi serta output
historisnya tidak dihapus.

## Notebook laporan canonical

Notebook baru dibuat secara bertahap setelah evidence terkait tersedia:

```text
C1_tabular_benchmark_report.ipynb
C2_image_screening_report.ipynb
C4_main_scenarios_report.ipynb
C5_metadata_ablation_report.ipynb
C6_statistics_xai_report.ipynb
C7_secondary_holdout_report.ipynb
```

Notebook laporan bersifat read-only terhadap evidence: hanya membaca manifest,
registry, prediction CSV, dan summary JSON dari direktori protocol hash. Ia
tidak melakukan split, fitting, checkpoint selection, atau tuning model.

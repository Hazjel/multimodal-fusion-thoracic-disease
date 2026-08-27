# Canonical Reporting Notebooks

Folder ini berisi notebook visualisasi dan penyusunan laporan yang dibuat
setelah setiap stage canonical selesai.

Setiap notebook di sini wajib:

1. mencantumkan protocol version dan full `protocol_hash`;
2. membaca artifact hanya dari direktori canonical aktif;
3. memverifikasi `_SUCCESS`, registry, schema, dan coverage sebelum membuat
   tabel atau gambar;
4. menghitung ulang metric dari prediction CSV bila relevan;
5. tidak melakukan data split, model fitting, tuning, atau checkpoint selection;
6. tidak membaca official NIH test sebelum C7;
7. memberi label yang jelas untuk primary, secondary, dan exploratory analysis.

Status notebook:

| Notebook | Status | Evidence |
|---|---|---|
| `C1_tabular_benchmark_report.ipynb` | Tersedia | C1 tabular benchmark complete |
| `C2_image_screening_report.ipynb` | Tersedia | C2 ImageNet screening + C3 model lock complete |
| `C4_main_scenarios_report.ipynb` | Tersedia | C4 S1/S2/S3, 15/15 fold complete |
| `C5_metadata_ablation_report.ipynb` | Tersedia | C5 S1/S3 feature sets A–D complete |
| `C6_statistics_xai_report.ipynb` | Tersedia | C6 bootstrap, calibration, SHAP, dan Grad-CAM complete |
| `C7_secondary_holdout_report.ipynb` | Menunggu C7 | Belum dibuat |

Dengan pola ini, CLI dan artifact registry tetap menjadi sumber kebenaran,
sedangkan notebook hanya menjadi presentation layer untuk BAB IV.

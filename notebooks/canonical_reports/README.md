# Canonical Reporting Notebooks

Folder ini disiapkan untuk notebook visualisasi dan penyusunan laporan setelah
setiap stage canonical selesai. Belum ada notebook laporan yang dibuat.

Setiap notebook di sini wajib:

1. mencantumkan protocol version dan full `protocol_hash`;
2. membaca artifact hanya dari direktori canonical aktif;
3. memverifikasi `_SUCCESS`, registry, schema, dan coverage sebelum membuat
   tabel atau gambar;
4. menghitung ulang metric dari prediction CSV bila relevan;
5. tidak melakukan data split, model fitting, tuning, atau checkpoint selection;
6. tidak membaca official NIH test sebelum C7;
7. memberi label yang jelas untuk primary, secondary, dan exploratory analysis.

Rencana nama file:

```text
C1_tabular_benchmark_report.ipynb
C2_image_screening_report.ipynb
C4_main_scenarios_report.ipynb
C5_metadata_ablation_report.ipynb
C6_statistics_xai_report.ipynb
C7_secondary_holdout_report.ipynb
```

Dengan pola ini, CLI dan artifact registry tetap menjadi sumber kebenaran,
sedangkan notebook hanya menjadi presentation layer untuk BAB IV.

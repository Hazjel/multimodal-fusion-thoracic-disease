# Matriks Literatur Canonical

Tanggal audit: 22 Agustus 2026

Status: working literature set untuk BAB I/BAB II; tidak mengubah scientific
protocol v1.0.0 yang sudah dibekukan.

## Prinsip pemilihan

Matriks inti berisi sepuluh studi empiris terbaru yang bersama-sama mencakup:

- fusion citra chest X-ray dan data klinis/metadata;
- penggunaan metadata pada NIH ChestXray14;
- perbandingan unimodal dan multimodal;
- patient-aware evaluation dan metadata ablation;
- explainability yang relevan terhadap citra dan structured data.

Jumlah sepuluh bukan syarat metodologis universal. Set ini dipilih karena setiap
paper mempunyai fungsi yang berbeda dalam menyusun gap. Paper image-only XAI,
review, dan benchmark tambahan ditempatkan sebagai literatur pendukung agar
matriks inti tidak berisi studi yang berulang.

## Matriks penelitian terdahulu

| No. | Penelitian | Dataset, Task & Evaluation Split | Input non-citra | Metode/fusion | Evaluasi & XAI | Keterbatasan dan Gap terhadap TA |
|---:|---|---|---|---|---|---|
| 1 | Hsieh et al. (2023), *MDF-Net for abnormality detection by fusing X-rays with clinical data*, Scientific Reports, [doi:10.1038/s41598-023-41463-0](https://doi.org/10.1038/s41598-023-41463-0) | MIMIC-Eye; abnormality localization lima kelas | Sepuluh clinical features | Mask R-CNN + MobileNetV3; dual 3-D/1-D fusion | Bounding-box localization dan ablation; AP meningkat 19.61% menjadi 31.69% | Bukti inti manfaat CXR + structured clinical data; task localization berbeda dari klasifikasi biner TA. |
| 2 | Khader et al. (2023), *Multimodal Deep Learning for Integrating Chest Radiographs and Clinical Parameters: A Case for Transformers*, Radiology, [doi:10.1148/radiol.230806](https://doi.org/10.1148/radiol.230806) | MIMIC dan internal ICU; hingga 25 kondisi; patient-disjoint split | Vital signs, laboratory/time-series, dan clinical parameters | Vision Transformer dan transformer/cross-attention fusion; clinical-only, image-only, multimodal | MIMIC mean AUC 0.77 multimodal vs 0.70 image-only dan 0.72 clinical-only | Dasar kuat untuk controlled S1/S2/S3 dan patient-level evaluation. |
| 3 | Shurrab et al. (2024), *Multimodal masked siamese network improves chest X-ray representation learning*, Scientific Reports, [doi:10.1038/s41598-024-74043-x](https://doi.org/10.1038/s41598-024-74043-x) | MIMIC-CXR pretraining; CheXpert dan NIH-14 downstream | Age, sex, view, patient position, ICU admission, mortality | Multimodal self-supervised pretraining; CXR + EHR concatenation | Tidak menggunakan XAI utama; patient-identifier split | Mendukung predictive value demographic/scan metadata, tetapi bukan direct image+metadata inference fusion pada NIH-14. |
| 4 | Huang et al. (2024), *Research and implementation of multi-disease diagnosis on chest X-ray based on vision transformer*, Quantitative Imaging in Medicine and Surgery, [doi:10.21037/qims-23-1280](https://doi.org/10.21037/qims-23-1280) | NIH ChestXray14; multi-label 14 penyakit; pasien dipisahkan antar-split | Age, gender, view position, serta patient-history information yang dibentuk dari pemeriksaan terdahulu | ViT-B/16 yang dimodifikasi + parallel metadata network + feature concatenation | Average AUC 0.831; sensitivity 0.863; specificity 0.821; tidak menyediakan complementary SHAP/Grad-CAM seperti TA | **Prior work sangat dekat.** Membuktikan bahwa novelty TA bukan sekadar menggabungkan NIH CXR dan metadata. Berbeda pada task, arsitektur, estimand, dan analisis metadata. |
| 5 | Tang et al. (2025), *Fusion of X-Ray Images and Clinical Data for a Multimodal Deep Learning Prediction Model of Osteoporosis*, JMIR Medical Informatics, [doi:10.2196/70738](https://doi.org/10.2196/70738) | Private single-center, 1,780 pasien; osteoporosis; 5-fold CV | Demografi, anthropometry, laboratory values, dan chest X-ray diagnosis/descriptive information | ResNet50 + wavelet/attention; clinical FC; probability fusion | SHAP untuk data klinis dan Grad-CAM untuk CXR; AUC 0.975 vs image-only 0.951 | Comparator XAI terdekat, tetapi domain, richness data klinis, dan late/probability fusion berbeda. |
| 6 | Hage Chehade et al. (2025), *Evaluating the impact of view position in X-ray imaging for the classification of lung diseases*, Physical and Engineering Sciences in Medicine, [doi:10.1007/s13246-025-01579-1](https://doi.org/10.1007/s13246-025-01579-1) | NIH ChestXray14; metadata analysis dan classification | Age, gender, view position | Hierarchical clustering; CNN + spatial attention pada kelompok view | Pneumonia weighted AUC 0.8176 dengan grouping vs 0.7941 tanpa grouping | Dasar bahwa View Position dapat menjadi predictor/confounder dan perlu dianalisis. Selisih grouping sekitar 0.0235; angka 0.0165 adalah perbandingan konfigurasi model tertentu. |
| 7 | Shimbre & Solanki (2025), *ChestXFusionNet*, EPJ Web of Conferences, [doi:10.1051/epjconf/202532801059](https://doi.org/10.1051/epjconf/202532801059) | NIH ChestXray14; abstrak/conclusion menyatakan finding vs no-finding | Age, gender, view position | Custom CNN + dense metadata branch + feature concatenation | CAM/activation map; accuracy dilaporkan 92.4% | Closest prior work, tetapi conference proceedings dan memiliki inkonsistensi binary output vs uraian enam kelas. Digunakan dengan caveat, bukan satu-satunya dasar gap. |
| 8 | Sloan et al. (2025), *Clinically-aligned Multi-modal Chest X-ray Classification (CaMCheX)*, PMLR 297/ML4H, [arXiv:2511.09581](https://arxiv.org/abs/2511.09581) | MIMIC-CXR/CXR-LT; study-level multi-view, multi-label | Clinical indications/history dan vital signs | View-specific ConvNeXt encoders + transformer fusion | Tiga seeds, ablation, mAP dan AUROC; CXR-LT 2023 dilaporkan mAP 0.576 dan AUROC 0.916 | Bukti mutakhir bahwa multimodal CXR berkembang ke study-level context. Input klinisnya jauh lebih kaya dan task berbeda dari TA. |
| 9 | Priego-Torres et al. (2025), *Multimodal Fusion of Chest X-Rays and Blood Biomarkers for Automated Silicosis Staging*, Journal of Clinical Medicine, [doi:10.3390/jcm14228074](https://doi.org/10.3390/jcm14228074) | 94 pasien/341 paired samples; silicosis staging; patient-aware 5-fold CV | Blood biomarkers | Early, late, dan hybrid fusion | AUC sekitar 0.85 untuk fusion vs 0.83 image-only dan 0.70 biomarker-only | Mendukung patient-aware multimodal comparison dan pemilihan fusion; kecil, privat, dan domain berbeda. |
| 10 | Orhan et al. (2026), *Scalable Unimodal and Multimodal Deep Learning for Multi-Label Chest Disease Detection*, Diagnostics, [doi:10.3390/diagnostics16050734](https://doi.org/10.3390/diagnostics16050734) | NIH ChestXray14, random subset dan full-scale; multi-label 14 penyakit | Gender dan image location/view position | ResNet50, EfficientNetB3, DenseNet121 dalam image-only dan image+metadata configurations | Label-wise AUROC; multimodal dilaporkan konsisten lebih tinggi; tidak melakukan per-modality OOF XAI | **Prior work paling dekat dengan C2/S2–S3.** Berbeda karena tidak memakai Age/Follow-up, binary estimand, tabular-only S1, ablation A–D, atau patient-cluster CI seperti TA. |

Baris penelitian ini ditulis terpisah dan tidak dihitung sebagai penelitian
terdahulu:

| Penelitian | Dataset dan task | Skenario | Evaluasi dan XAI | Pembeda yang diuji |
|---|---|---|---|---|
| Penelitian ini (2026) | NIH ChestXray14; binary Normal vs Abnormal; image-level prediction dengan patient-level grouping | S1 canonical MLP; S2 ResNet-50 ImageNet; S3 intermediate concatenation ResNet-50 + MLP; Age, Gender, View Position, Follow-up # | Patient-level 5-fold CV; paired patient-cluster bootstrap 95% CI; ablation A–D; fold-specific image-conditioned SHAP; paired OOF Grad-CAM; official test sebagai secondary holdout with prior exposure | Controlled incremental value `AUC(S3-D) - AUC(S2)`, uncertainty, metadata-specific contribution, dan complementary per-modality explainability. C3 locked ResNet-50 ImageNet; proposal amendment disetujui pembimbing pada 24 Agustus 2026. |

## Literatur pendukung di luar matriks inti

- Mahamud et al. (2024), Rajpoot et al. (2024), dan Aasem & Iqbal
  (2024): supporting image-XAI. SHAP pada studi tersebut tidak boleh
  dipresentasikan sebagai SHAP structured metadata.
- Ucan et al. (2025): recent image-only NIH benchmark dan konteks backbone.
- Li et al. (2024) dan Nakach et al. (2024): review/synthesis teori multimodal
  fusion, bukan bukti empiris utama.
- Alloqmani & Abushark (2026): konsep early vs intermediate fusion, tetapi
  modalitas tidak dipasangkan pada patient level; tidak digunakan sebagai core
  empirical evidence.
- Baltruschat et al. (2019), *Comparison of Deep Learning Approaches for
  Multi-Label Chest X-Ray Classification*: preseden historis yang sangat relevan
  untuk NIH + Age/Gender/View Position fusion, tetapi tidak dihitung sebagai
  sumber utama kebaruan lima tahun terakhir.

## Research gap canonical

Penelitian terdahulu telah menggabungkan chest X-ray dengan structured
clinical/non-image information, termasuk pada NIH ChestXray14. Karena itu,
novelty penelitian ini **bukan** klaim pertama menggabungkan citra dan metadata,
pertama memakai SHAP/Grad-CAM, atau asumsi bahwa multimodal pasti lebih baik.

Gap yang diuji adalah masih terbatasnya evaluasi pada NIH ChestXray14 yang
secara bersamaan:

1. membandingkan metadata-only, image-only, dan intermediate-fusion secara
   terkontrol;
2. mengukur incremental discrimination S3 terhadap S2 dengan paired
   patient-cluster confidence interval;
3. menguji kontribusi konfigurasi metadata, termasuk Follow-up Number, melalui
   ablation A–D;
4. menjaga split dan explainability pada patient-level/OOF; dan
5. menggunakan SHAP untuk structured branch serta Grad-CAM untuk image branch
   tanpa mengklaim satu joint cross-modal attribution.

Formulasi ini adalah sintesis atas working literature set, bukan klaim bahwa
setiap unsur tersebut belum pernah diteliti secara terpisah.

## Catatan penggunaan

- Seluruh sepuluh paper inti berada pada 2023–2026 dan dapat digunakan untuk
  tren kebaruan lima tahun terakhir.
- Venue dan indexing perlu diverifikasi kembali pada waktu finalisasi naskah;
  status indexing venue tidak menggantikan audit kualitas metodologi paper.
- Hasil C1/C2 dan model lock tidak boleh digunakan untuk mengubah research gap
  secara post hoc. Literatur menjelaskan pertanyaan yang sudah dibekukan;
  eksperimen menjawab pertanyaan tersebut.

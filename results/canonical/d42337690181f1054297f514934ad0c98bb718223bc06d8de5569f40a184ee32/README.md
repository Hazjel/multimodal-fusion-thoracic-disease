# Canonical Evidence — Protocol v1.0.0

Direktori ini menyimpan bukti canonical untuk scientific protocol hash:

```text
d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32
```

Status saat dokumentasi ini dibuat: **C0 PASS, C1 selesai, C2 selesai dan
diaudit, C3 selesai; C4 menunggu eksekusi**.

## Ringkasan C1 — Tabular Benchmark

Semua nilai berikut berasal dari pooled out-of-fold predictions pada 86.524
citra dari 28.008 pasien. C1 adalah characterization benchmark; hasilnya tidak
mengganti MLP sebagai encoder canonical S1/S3.

| Model | Pooled ROC-AUC | Average Precision | Brier score |
|---|---:|---:|---:|
| Canonical MLP | 0.621382 | 0.519234 | 0.240854 |
| RealMLP | 0.623753 | 0.521023 | 0.231982 |
| TabM | 0.622107 | 0.518755 | 0.232309 |

## Ringkasan C2 — ImageNet Screening

| Backbone | Pooled ROC-AUC | Mean fold ROC-AUC ± SD | Average Precision |
|---|---:|---:|---:|
| ResNet-50 | **0.752390** | **0.753542 ± 0.002201** | **0.672896** |
| EfficientNet-B0 | 0.750460 | 0.751193 ± 0.001919 | 0.670582 |
| DenseNet-121 | 0.744798 | 0.745843 ± 0.002406 | 0.664484 |

Paired patient-cluster bootstrap menggunakan 2.000 deterministic replicates:

| Perbandingan | Δ ROC-AUC | 95% percentile CI |
|---|---:|---:|
| ResNet-50 − DenseNet-121 | +0.007591 | [+0.005609, +0.009606] |
| ResNet-50 − EfficientNet-B0 | +0.001930 | [+0.000009, +0.003881] |

Berdasarkan aturan seleksi yang dibekukan, evidence C2 mengarah ke **ResNet-50
ImageNet** dan candidate set hanya berisi ResNet-50. Conditional CheXNet
comparison tidak dijalankan karena kandidat terpilih bukan DenseNet-121.

## C3 — Model Lock dan Amendment Proposal

C3 menghasilkan `model_lock.json` berstatus `LOCKED` dengan keputusan:

- backbone canonical: **ResNet-50**;
- pretraining canonical: **ImageNet**;
- heuristic candidate set: hanya ResNet-50;
- scientific protocol dan protocol hash: tidak berubah.

Proposal awal menyebut DenseNet-121/CheXNet. Pembimbing menyetujui penggunaan
ResNet-50 setelah melihat evidence ROC-AUC C2, sebagaimana dilaporkan mahasiswa
pada 24 Agustus 2026. Keputusan dicatat dalam `proposal_amendment.json` dan
diikat ke checksum `model_lock.json`. Amendment ini memperbarui metode pada
proposal, bukan scientific protocol yang sudah dibekukan.

## Integritas evidence

- C1: 15/15 run `done` (3 model × 5 fold).
- C2: 15/15 run `done` (3 backbone × 5 fold).
- Setiap model memiliki tepat 86.524 OOF predictions dari 28.008 pasien.
- Seluruh fold cocok dengan `folds.csv`; tidak ada patient overlap.
- Protocol hash, environment hash, dan implementation commit konsisten dalam
  masing-masing stage.
- Metric yang diregenerasi dari prediction CSV cocok dengan summary artifact.
- Marker `_SUCCESS` tersedia untuk C1 dan C2.
- `model_lock.json` dan `proposal_amendment.json` tersedia serta gate C4 telah
  divalidasi terhadap protocol hash dan checksum model lock.

## Struktur direktori

```text
./
├── protocol.json                 # scientific specification + provenance
├── folds.csv                     # immutable primary-CV manifest
├── deployment_split.csv          # immutable 90/10 deployment manifest
├── experiment_registry.csv       # indeks seluruh canonical run
├── checksums.json                # checksum artifact freeze
├── environment.json              # environment pada protocol freeze
├── pip-freeze.txt                # dependency snapshot
├── model_lock.json               # keputusan immutable C3
├── proposal_amendment.json       # persetujuan perubahan metode proposal
├── runs/                         # artifact per run/fold
├── screening/tabular/            # pooled OOF dan summary C1
├── screening/image/              # pooled OOF dan summary C2
├── main/                         # C4; belum dijalankan
├── ablation/                     # C5; belum dijalankan
├── xai/ dan statistics/          # C6; belum dijalankan
└── secondary_holdout/            # C7; belum dijalankan
```

## Provenance dan penggunaan

`protocol_hash` mengidentifikasi keputusan ilmiah, sedangkan exact Git commit,
environment, dan resolved run configuration tercatat terpisah melalui
provenance serta semantic config hash. Jangan memindahkan atau mengganti nama
artefak run secara manual.

OOF CSV berisi pengenal NIH yang diperlukan untuk patient-cluster bootstrap dan
audit XAI. Checkpoint biner tidak disertakan dalam publikasi GitHub; evidence
yang dipublikasikan berfokus pada prediction, metrics, registry, dan provenance.

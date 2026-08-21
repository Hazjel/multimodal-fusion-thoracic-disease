# NIH ChestX-ray14 Multimodal Classification

Repository Tugas Akhir untuk klasifikasi biner **Normal vs Abnormal** pada NIH
ChestX-ray14 menggunakan citra X-ray dan empat metadata: Age, Gender, View
Position, dan Follow-up #.

## Status penelitian

| Tahap | Status | Ringkasan |
|---|---|---|
| C0 | **PASS** | Protocol v1.0.0 sudah dibekukan dan guardrail implementasi lulus |
| C1 | **Selesai** | MLP, RealMLP, dan TabM; masing-masing 5 fold |
| C2 | **Selesai dan diaudit** | DenseNet-121, ResNet-50, dan EfficientNet-B0; masing-masing 5 fold |
| C3 | **Menunggu** | Aturan frozen mengarah ke ResNet-50 ImageNet; `model_lock.json` belum dibuat |
| C4–C7 | **Belum dimulai** | Menunggu C3 dan konsultasi amendment proposal |

- Scientific protocol hash:
  `d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32`.
- Protocol: **v1.0.0 — FROZEN**.
- Primary evidence: patient-level 5-fold `StratifiedGroupKFold`.
- Official NIH test hanya digunakan pada C7 sebagai *secondary holdout with
  prior exposure*.
- Hasil sebelum protocol freeze tetap berstatus pilot/exploratory atau legacy.

Indeks hasil dan aturan penamaan tersedia di
[results/canonical/README.md](results/canonical/README.md). Ringkasan lengkap
C1/C2 tersedia di direktori protocol aktif:
[canonical evidence README](results/canonical/d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32/README.md).

## Skenario utama

| Skenario | Input | Arsitektur |
|---|---|---|
| S1 | Empat metadata | MLP `4 → 64 → 128 → 1` |
| S2 | Citra | CNN terpilih `→ GAP → 512 → 1` |
| S3 | Citra + metadata | Concatenation `640 → 256 → 128 → 1` |

C1 adalah benchmark karakterisasi metadata. MLP tetap menjadi encoder canonical
S1/S3, terlepas dari perbandingan terhadap RealMLP dan TabM. C2 memilih image
backbone mengikuti aturan yang sudah dibekukan sebelum hasil AUC diamati.

## Perintah eksekusi

Jalankan dari root repository. `--protocol-dir` tidak diperlukan apabila
direktori canonical aktif menggunakan struktur standar.

```powershell
python run_experiment.py status
python run_experiment.py c0

# Sudah selesai pada protocol aktif
python run_experiment.py benchmark-tabular --model all --device cuda
python run_experiment.py screen-image --backbone all --pretraining imagenet

# Tahap berikutnya
python run_experiment.py select
python run_experiment.py main --scenario all
python run_experiment.py ablate --scenario both --feature-set all
```

`main` dan `ablate` akan *hard-fail* sebelum `model_lock.json` tersedia.
Apabila C3 memilih metode berbeda dari proposal, persetujuan pembimbing dicatat
sebagai proposal amendment tanpa mengubah scientific protocol.

## Struktur hasil

```text
results/
├── canonical/       # bukti penelitian berdasarkan protocol hash
├── exploratory/     # eksperimen tambahan yang bukan bukti utama
├── legacy/          # artefak dari desain lama
└── STATUS.md        # status dan batas penggunaan artefak
```

BAB IV hanya boleh mengambil angka dari direktori protocol canonical yang
memiliki registry valid, checksum, dan penanda `_SUCCESS` pada tahap terkait.
Checkpoint biner tetap disimpan lokal dan tidak dipublikasikan ke GitHub.

## Reproducibility dan guardrail

- Scientific hash, runtime config, manifest split, dan environment diverifikasi
  sebelum canonical run.
- C1–C6 hanya membaca official training pool; official test diblokir sebelum C7.
- Split dan bootstrap menggunakan Patient ID sebagai grouping unit.
- Resume memulihkan model, optimizer, scheduler, scaler AMP, epoch, serta state
  RNG.
- Seluruh kandidat C2 harus menggunakan environment hash yang sama.
- CheXNet hanya boleh dijalankan setelah audit provenance berstatus `APPROVED`;
  kegagalan load tidak boleh melakukan fallback.
- Commit dan tag menggunakan identitas Git peneliti tanpa AI author/co-author.

Spesifikasi ilmiah lengkap tersedia di
[Canonical Execution Protocol v1.0.0-rc2](docs/CANONICAL_PROTOCOL_v1.0.0-rc2.md).

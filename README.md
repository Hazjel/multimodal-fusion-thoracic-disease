# NIH ChestX-ray14 Multimodal Classification

Tugas Akhir tentang klasifikasi biner **Normal vs Abnormal** menggunakan citra
X-ray dan empat metadata NIH ChestX-ray14: Age, Gender, View Position, dan
Follow-up #.

Status saat ini:

- Desain penelitian: **FINAL**.
- Protocol: **v1.0.0 — FROZEN**.
- Scientific protocol hash: `d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32`.
- C0 awal telah lulus; implementation patch wajib menjalankan C0 ulang tanpa
  mengubah scientific protocol hash.
- Hasil commit terdahulu tetap disimpan sebagai legacy/exploratory dan bukan
  bukti canonical.

## Skenario utama

| Skenario | Input | Arsitektur |
|---|---|---|
| S1 | Empat metadata | MLP `4 → 64 → 128 → 1` |
| S2 | Citra | CNN terpilih `→ GAP → 512 → 1` |
| S3 | Citra + metadata | Concatenation `640 → 256 → 128 → 1` |

Primary evidence menggunakan patient-level five-fold
`StratifiedGroupKFold`. Official NIH test hanya digunakan pada C7 sebagai
secondary holdout dengan disclosure bahwa hasilnya pernah terlihat pada
eksperimen lama.

## Status eksekusi

- C0 dan protocol freeze: **selesai**.
- C1–C7 canonical: **belum menghasilkan hasil final**.
- Semua hasil sebelum freeze adalah pilot/exploratory dan bukan primary
  evidence BAB IV.

## Menjalankan acceptance test

```powershell
python -m pip install -r requirements-c0.txt
python run_experiment.py c0
```

C0 menguji hashing, split, preprocessing, structural fine-tuning, frozen
BatchNorm, initialization equality S2/S3, PyTabKit runtime contract,
checkpoint/resume, metrics, prediction schema, dan image-conditioned SHAP.
C0 bukan eksperimen performa.

Untuk setup repository baru, freeze hanya boleh dilakukan setelah report C0
menyatakan PASS pada implementation commit yang bersih:

```powershell
python run_experiment.py freeze
```

Perintah tersebut menghasilkan `folds.csv`, `deployment_split.csv`,
`protocol.json`, `environment.json`, `pip-freeze.txt`, registry, checksums, dan
`_SUCCESS` di `results/canonical/<protocol_hash>/`.

Detail lengkap: [Canonical Protocol rc2](docs/CANONICAL_PROTOCOL_v1.0.0-rc2.md).

## Antarmuka eksekusi canonical

Generic `cv` sengaja dihapus dari CLI agar stage tidak dapat dilewati.

```powershell
python run_experiment.py status
python run_experiment.py benchmark-tabular --model all
python run_experiment.py screen-image --backbone all --pretraining imagenet
python run_experiment.py select
# Hanya jika select menghasilkan kandidat DenseNet:
Copy-Item docs/chexnet_provenance_declaration.example.json docs/chexnet_provenance_declaration.json
# Isi declaration berdasarkan bukti yang dapat diverifikasi, lalu:
python run_experiment.py audit-chexnet --declaration docs/chexnet_provenance_declaration.json
# Jalankan CheXNet hanya bila audit berstatus APPROVED, kemudian select ulang.
python run_experiment.py screen-image --backbone densenet121 --pretraining chexnet
python run_experiment.py select
python run_experiment.py main --scenario all
python run_experiment.py ablate --scenario both --feature-set all
```

`main` dan `ablate` hard-fail sebelum `model_lock.json` tersedia. Conditional
CheXNet hanya dapat dijalankan setelah C3 memilih kandidat DenseNet-121 dan
`chexnet_provenance_audit.json` berstatus `APPROVED`. Deklarasi yang tidak dapat
membuktikan sumber, split, preprocessing, label mapping, dan non-use official
NIH test menghasilkan status `EXCLUDED`; C3 kemudian mengunci ImageNet tanpa
menjalankan CheXNet canonical.

## Guardrail

- Canonical factory hanya menerima S1, S2, dan S3.
- Full CV ditolak sebelum protocol berstatus `FROZEN`.
- Scientific hash, runtime config, dan kedua manifest diverifikasi sebelum
  setiap canonical run; mismatch menghasilkan hard error.
- C1–C6 hanya membaca `train_val_list.txt`, bukan `test_list.txt`.
- Resume menyimpan RNG global dan state generator DataLoader; worker tidak
  dibuat persistent agar restart dapat direproduksi.
- Run pertama setiap stage mengunci `environment_hash`. Seluruh fold/kandidat
  C2 harus memakai environment yang sama; C3 menolak registry yang berbeda.
- Entry point menetapkan `CUBLAS_WORKSPACE_CONFIG=:4096:8` sebelum CUDA dipakai,
  mencatatnya pada environment hash, dan memblokir nilai yang berbeda.
- Official test diblokir sebelum C7.
- CheXNet yang gagal dimuat tidak pernah fallback diam-diam.
- Aplikasi C7 harus menampilkan **skor model**, bukan probabilitas terkalibrasi.
- Commit/tag menggunakan identitas Git peneliti tanpa AI author/co-author.

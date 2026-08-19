# NIH ChestX-ray14 Multimodal Classification

Tugas Akhir tentang klasifikasi biner **Normal vs Abnormal** menggunakan citra
X-ray dan empat metadata NIH ChestX-ray14: Age, Gender, View Position, dan
Follow-up #.

Status saat ini:

- Desain penelitian: **FINAL**.
- Protocol: **v1.0.0-rc2 — Final Freeze Candidate**.
- `v1.0.0 — FROZEN` hanya setelah seluruh acceptance test C0 lulus.
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

## Menjalankan C0

```powershell
python -m pip install -r requirements-c0.txt
python run_experiment.py c0
```

C0 menguji hashing, split, preprocessing, structural fine-tuning, frozen
BatchNorm, initialization equality S2/S3, PyTabKit runtime contract,
checkpoint/resume, metrics, prediction schema, dan image-conditioned SHAP.
C0 bukan eksperimen performa.

Commit implementasi C0 terlebih dahulu, lalu jalankan C0 pada commit bersih.
Setelah report menyatakan PASS untuk commit tersebut:

```powershell
python run_experiment.py freeze
```

Perintah tersebut menghasilkan `folds.csv`, `deployment_split.csv`,
`protocol.json`, `environment.json`, `pip-freeze.txt`, registry, checksums, dan
`_SUCCESS` di `results/canonical/<protocol_hash>/`.

Detail lengkap: [Canonical Protocol rc2](docs/CANONICAL_PROTOCOL_v1.0.0-rc2.md).

## Guardrail

- Canonical factory hanya menerima S1, S2, dan S3.
- Full CV ditolak sebelum protocol berstatus `FROZEN`.
- Official test diblokir sebelum C7.
- CheXNet yang gagal dimuat tidak pernah fallback diam-diam.
- Aplikasi C7 harus menampilkan **skor model**, bukan probabilitas terkalibrasi.
- Commit/tag menggunakan identitas Git peneliti tanpa AI author/co-author.

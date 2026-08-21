# Indeks Hasil Canonical

Setiap subdirektori di sini mewakili satu scientific protocol hash. Nama hash
dipertahankan agar hasil dapat ditelusuri langsung ke `protocol.json`, manifest
split, dan aturan ilmiah yang menghasilkan artefak tersebut.

## Protocol aktif

| Versi | Protocol hash | Status |
|---|---|---|
| v1.0.0 | [`d4233769…a184ee32`](d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32/README.md) | FROZEN; C1/C2 selesai, C3 menunggu |

## Konvensi penamaan run

Nama run mengikuti pola berikut:

```text
<stage>-<scenario/model>-<backbone>-<pretraining>-<feature_set>-fold<k>-<semantic_hash_prefix>
```

Komponen yang tidak berlaku ditulis eksplisit sebagai `not_applicable` atau
dihilangkan hanya bila runner canonical memang menetapkan pola model-specific.
Contoh yang sudah dihasilkan:

```text
C1-S1-canonical_mlp-not_applicable-D-fold0-<hash>
C1-realmlp-D-fold0-<hash>
C1-tabm-D-fold0-<hash>
C2-S2-resnet50-imagenet-D-fold0-<hash>
```

Arti komponen:

- `C1`, `C2`, ...: tahap protocol.
- `S1`, `S2`, `S3`: skenario utama bila berlaku.
- nama model/backbone: implementasi yang diuji.
- `imagenet`/`chexnet`: sumber pretraining.
- `A`–`D`: feature set metadata; `D` berarti seluruh empat metadata.
- `fold0`–`fold4`: validation fold primary CV.
- suffix hash: identitas semantic configuration run.

Nama file dan direktori yang sudah diregistrasikan tidak boleh diganti manual.
Perubahan nama akan memutus hubungan dengan registry, checksum, resume state,
dan audit provenance. Dokumentasi ini menjadi lapisan nama yang mudah dibaca
manusia tanpa mengubah identitas mesin.

## Kebijakan publikasi

- CSV/JSON dan marker `_SUCCESS` dapat dipublikasikan sebagai evidence bundle.
- Checkpoint `.pt` dan scaler/model binary `.pkl` tetap lokal dan diabaikan Git.
- OOF prediction memuat pengenal NIH (`patient_id` dan `image_index`), label,
  serta skor model. Perlakukan sebagai artefak penelitian dan jangan mengubah
  schema atau isinya secara manual.
- Selalu publikasikan README, registry, summary, dan provenance bersama hasil
  agar angka tidak terlepas dari konteks protocol.

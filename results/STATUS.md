# Status Artefak Penelitian

Dokumen ini menentukan status bukti penelitian dan mencegah pencampuran hasil
canonical dengan eksperimen lama.

## Canonical

- Protocol: **v1.0.0 — FROZEN**.
- Scientific protocol hash:
  `d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32`.
- C0: **PASS**.
- C1 tabular benchmark: **selesai**, 15/15 run.
- C2 ImageNet screening: **selesai dan diaudit**, 15/15 run.
- C3 model lock: **belum dibuat**.
- C4–C7: **belum dimulai**.

Hasil C1/C2 berada di
[`canonical/d423.../`](canonical/d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32/README.md).
Aturan frozen C2 mengarah ke ResNet-50 ImageNet, tetapi keputusan operasional
baru dianggap terkunci setelah C3 menghasilkan `model_lock.json` dan proses
proposal amendment diselesaikan bersama pembimbing.

BAB IV hanya boleh memakai angka dari direktori protocol aktif apabila tahap
terkait memiliki `_SUCCESS`, schema prediksi valid, registry lengkap, serta
checksum yang dapat diverifikasi.

## Exploratory

Artefak exploratory boleh digunakan untuk menjelaskan riwayat pengembangan,
tetapi:

- bukan primary comparative evidence;
- tidak boleh dicampur ke tabel hasil canonical;
- tidak boleh digunakan untuk mengubah kandidat atau hyperparameter frozen;
- hasil official test lama harus disertai disclosure *prior exposure*.

Kategori ini mencakup attention/gated fusion, eksperimen backbone lama,
complementarity analysis lama, serta analisis tambahan di luar protocol.

## Legacy

Artefak berikut tidak kompatibel dengan protocol canonical:

- row-level atau non-patient-grouped split;
- arsitektur metadata lama;
- S4 multi-label;
- checkpoint dengan preprocessing, initialization, atau test policy lama;
- silent fallback CheXNet atau artifact tanpa provenance yang dapat diaudit.

## Aturan tahap

1. C1 dan C2 hanya berjalan setelah protocol berstatus `FROZEN`.
2. C4 dan C5 hanya berjalan setelah `model_lock.json` tersedia.
3. Proposal amendment diperlukan bila hasil frozen selection rule berbeda dari
   metode pada proposal, tetapi tidak mengubah `protocol_hash`.
4. Scientific protocol amendment menghasilkan versi dan hash baru; hasil
   antarversi tidak dicampur.
5. Implementation bug-fix mengubah implementation commit dan semantic config
   hash, bukan scientific protocol hash; run terdampak wajib diulang.
6. C1–C6 tidak boleh membaca official `test_list.txt`.

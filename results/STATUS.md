# Status Artefak Penelitian

Dokumen ini adalah sumber status untuk membedakan bukti canonical dari hasil
pilot dan legacy.

## CANONICAL

- Canonical Execution Protocol **v1.0.0 — FROZEN**.
- Scientific protocol hash:
  `d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32`.
- Manifest primary CV dan deployment split sudah immutable.
- **Belum ada hasil canonical C1–C7.**
- BAB IV hanya boleh mengambil angka dari direktori protocol hash di
  `results/canonical/` setelah stage terkait memiliki `_SUCCESS` dan artifact
  checksum/registry yang valid.

## PILOT / EXPLORATORY

Seluruh checkpoint, tabel, statistik, dan XAI S1/S2/S3 yang dibuat sebelum
freeze diperlakukan sebagai pilot/exploratory. Artefak tersebut tetap berguna
sebagai riwayat pengembangan, tetapi:

- tidak menjadi primary comparative evidence;
- tidak boleh disebut hasil final BAB IV;
- tidak boleh dipakai untuk mengubah kandidat atau hyperparameter frozen;
- official-test metrics lama harus disertai disclosure prior exposure jika
  dibahas sebagai konteks sekunder.

Ini termasuk AUC lama S1/S2/S3, DeLong lama, attention/gated fusion,
EfficientNet/ResNet lama, serta complementarity analysis lama.

## LEGACY

Artefak berikut tidak kompatibel dengan protocol canonical dan tidak boleh
digunakan sebagai bukti penelitian:

- row-level atau non-patient-grouped split;
- arsitektur metadata tiga hidden layer yang sudah diganti;
- S4 multi-label;
- checkpoint dengan preprocessing, initialization, atau test policy lama;
- silent fallback CheXNet atau artifact tanpa provenance yang dapat diaudit.

## Aturan status

1. C1 dan C2 hanya boleh dimulai setelah protocol berstatus `FROZEN`.
2. C4/C5 hanya boleh berjalan setelah `model_lock.json` terbentuk dan proposal
   amendment disetujui bila diwajibkan.
3. C1–C6 tidak membaca official `test_list.txt`.
4. Implementation bug-fix mengubah `implementation_commit` dan semantic hash,
   bukan scientific protocol hash.
5. Hasil canonical dan exploratory tidak boleh dicampur dalam tabel utama.

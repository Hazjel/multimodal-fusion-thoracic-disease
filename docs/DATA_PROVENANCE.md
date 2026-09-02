# NIH ChestX-ray14 Data Provenance

## Public sources

NIH ChestX-ray14 is available from two public endpoints:

- Canonical NIH Clinical Center distribution:
  <https://nihcc.app.box.com/v/ChestXray-NIHCC>
- NIH Chest X-rays mirror on Kaggle:
  <https://www.kaggle.com/datasets/nih-chest-xrays/data>

The files in the local study snapshot match the standard ChestX-ray14 package.
The original download endpoint was not retained, so this repository does not
assert whether the local copy was acquired directly from NIH or from the Kaggle
mirror. Reproducibility is tied to the byte-level identifiers below instead of
an unverified acquisition claim.

## Local snapshot used by the study

| Item | Value |
|---|---|
| PNG radiographs | 112,120 |
| Metadata file | `Data_Entry_2017.csv` |
| Official training-pool list | `train_val_list.txt` |
| Official test list | `test_list.txt` |
| Image directories | `images_001/images` through `images_012/images` |

SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `Data_Entry_2017.csv` | `88f75094e25ccc0c6f1f9cdfd4b2f94f9379a0ae07d5ff4dcf94242707b07462` |
| `train_val_list.txt` | `61fbe896321c1c1c8b75f3e4f3a08e4fef6486d95ef8a667c31d4d60dca6cb81` |
| `test_list.txt` | `38ca5ef7f756092946f57c1a59faca882ed589a1ab1f72590b45dc06c6d5e1cc` |
| `README_CHESTXRAY.pdf` | `93d1614fa2ec27da98a8cef803ea8d39cc6132ad2ea1b57460e17b9887c2085d` |

## Study use

- `train_val_list.txt` defines the official NIH training pool used for C1 to
  C6 and the patient-level five-fold primary evaluation.
- `test_list.txt` defines the secondary C7 holdout. It was not read by C1 to C6.
- The prediction unit is an image/examination. Patient ID is the grouping unit
  for splitting and clustered uncertainty estimation.
- No raw radiographs are committed to Git.
- The official-test row-level manifest and predictions remain local. Aggregate
  C7 metrics, calibration summaries, access provenance, and checksums are
  published.

## Local setup

Set the dataset root before running the pipeline when it is not located next to
the repository:

```powershell
$env:NIH_DATASET_ROOT = "D:\path\to\nih-chest-xrays"
```

The runtime rejects missing metadata, missing split lists, unexpected category
values, and patient overlap across canonical folds.

# C7 Secondary Holdout Results

The official NIH ChestX-ray14 test partition was evaluated once under access
event `1fd6844c-a989-49c5-84d6-5e44a4127b9f`. These results are secondary
holdout evidence with prior-exposure disclosure. Primary comparative evidence
remains the patient-level five-fold out-of-fold evaluation on the official
training pool.

## Aggregate results

| Scenario | ROC-AUC | Average Precision | Brier score | Accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 metadata-only | 0.617793 | 0.685066 | 0.226724 | 0.651117 | 0.866857 | 0.306865 | 0.753383 |
| S2 image-only | 0.710535 | 0.764908 | 0.205502 | 0.700930 | 0.837941 | 0.482304 | 0.775018 |
| S3 multimodal | 0.712527 | 0.765530 | 0.202772 | 0.704837 | 0.838894 | 0.490924 | 0.777500 |

The secondary-holdout ROC-AUC difference `S3 - S2` is `0.001992`. This small
positive difference does not replace the primary OOF conclusion. The primary
patient-cluster bootstrap interval crossed zero, so the available evidence did
not clearly separate S3 and S2.

## Published and local-only evidence

Git tracks aggregate metrics, calibration tables, summaries, access receipt,
prior-exposure disclosure, and SHA-256 provenance. Row-level prediction files,
the official-test image/patient manifest, case-level SHAP background records,
model checkpoints, and scalers remain local and are excluded from publication.

The complete local scientific artifact manifest contains 17 immutable evidence
files, all of which passed checksum and byte-size verification after C7.

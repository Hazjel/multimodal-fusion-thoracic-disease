# Canonical Evidence - Protocol v1.0.0

Scientific protocol hash:

```text
d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32
```

Status: **C0-C7 complete**. Primary evidence comes from patient-grouped
five-fold out-of-fold predictions. The NIH official test partition is reported
only as a secondary holdout with prior-exposure disclosure.

## C1 - Tabular characterization benchmark

All values use pooled OOF predictions from 86,524 images belonging to 28,008
patients. This benchmark characterizes the four structured variables; it does
not replace the canonical MLP used by S1 and S3.

| Model | ROC-AUC | Average Precision | Brier score |
|---|---:|---:|---:|
| Canonical MLP | 0.621382 | 0.519234 | 0.240854 |
| RealMLP | 0.623753 | 0.521023 | 0.231982 |
| TabM | 0.622107 | 0.518755 | 0.232309 |

## C2-C3 - Image screening and model lock

| Backbone | Pooled ROC-AUC | Fold ROC-AUC, mean +/- SD | Average Precision |
|---|---:|---:|---:|
| ResNet-50 | **0.752390** | **0.753542 +/- 0.002201** | **0.672896** |
| EfficientNet-B0 | 0.750460 | 0.751193 +/- 0.001919 | 0.670582 |
| DenseNet-121 | 0.744798 | 0.745843 +/- 0.002406 | 0.664484 |

The paired 2,000-replicate patient-cluster bootstrap produced:

| Comparison | Delta ROC-AUC | 95% percentile CI |
|---|---:|---:|
| ResNet-50 - DenseNet-121 | +0.007591 | [+0.005609, +0.009606] |
| ResNet-50 - EfficientNet-B0 | +0.001930 | [+0.000009, +0.003881] |

The frozen selection rule locked **ResNet-50 with ImageNet initialization**.
The conditional CheXNet comparison was not run because DenseNet-121 was not
selected. The proposal-method update is recorded separately from the unchanged
scientific protocol.

## C4 - Primary comparative evidence

| Scenario | Input | ROC-AUC | Average Precision | Brier score |
|---|---|---:|---:|---:|
| S1-D | Four metadata variables | 0.621382 | 0.519234 | 0.240854 |
| S2-D | Chest X-ray | 0.752390 | 0.672896 | 0.200192 |
| S3-D | Chest X-ray plus metadata | **0.753367** | 0.671663 | 0.200396 |

Primary estimand:

```text
Delta ROC-AUC (S3-D - S2) = +0.000977
95% patient-cluster bootstrap CI = [-0.000375, +0.002317]
```

The interval includes zero. Under the prespecified analysis, the available OOF
evidence did not establish a clear improvement in discrimination from adding
all four metadata variables to the image-only model.

## C5 - Metadata ablation

| S3 feature set | Included metadata | ROC-AUC | Delta vs S2 | 95% CI |
|---|---|---:|---:|---:|
| A | Age, Gender | 0.753244 | +0.000854 | [-0.000188, +0.001888] |
| B | A plus View Position | 0.753603 | +0.001214 | [-0.000173, +0.002528] |
| C | A plus Follow-up # | 0.753471 | +0.001081 | [-0.000413, +0.002463] |
| D | All four variables | 0.753367 | +0.000977 | [-0.000375, +0.002317] |

Descriptive differences relative to S3-A were +0.000359 for B, +0.000227 for
C, and +0.000123 for D. These are incremental predictive contributions under
the tested feature configurations, not causal effects.

## C6 - Statistics and explainability

- All paired confidence intervals use 2,000 deterministic bootstrap replicates
  clustered by `patient_id` and are conditional on the fitted CV models.
- Calibration diagnostics use Brier score and 10-bin reliability summaries;
  sigmoid outputs are not claimed to be calibrated probabilities.
- Fold-aware, image-conditioned KernelSHAP used 200 OOF cases. Mean absolute
  conditional attribution was highest for Follow-up # (0.008760), followed by
  View Position (0.007145), Age (0.005614), and Gender (0.002839).
- Paired OOF Grad-CAM compared S2 and S3 on the same eight selected cases. The
  maps are treated as a behavioral audit rather than lesion localization.

## C7 - Secondary official holdout

| Scenario | ROC-AUC | Average Precision | Brier score |
|---|---:|---:|---:|
| S1 | 0.617793 | 0.685066 | 0.226724 |
| S2 | 0.710535 | 0.764908 | 0.205502 |
| S3 | **0.712527** | **0.765530** | **0.202772** |

The C7 S3-S2 ROC-AUC difference was +0.001992. These figures are secondary
evidence because the official partition had influenced earlier exploratory
work; primary inference remains based on pooled OOF predictions.

## Integrity and publication policy

- C1: 15/15 registered runs complete.
- C2: 15/15 registered runs complete.
- C4: 15/15 registered runs complete.
- C5: 30/30 registered runs complete.
- C6 statistics, SHAP, and Grad-CAM have completion markers.
- C7 records one authorized official-test access event.
- Fold manifests have no patient overlap, and metrics can be regenerated from
  the published prediction artifacts.
- Binary checkpoints, fitted scalers, and row-level C7 predictions remain local
  and are ignored by Git. Aggregate metrics, OOF evidence, registry entries,
  protocol manifests, and provenance are published.

## Directory map

```text
protocol.json                 frozen scientific specification and provenance
folds.csv                     immutable primary-CV manifest
deployment_split.csv          immutable 90/10 deployment manifest
experiment_registry.csv       registered canonical runs
checksums.json                freeze artifact checksums
environment.json              freeze environment manifest
pip-freeze.txt                dependency snapshot
model_lock.json               immutable C3 decision
proposal_amendment.json       proposal-method record
screening/                    C1 and C2 summaries plus OOF predictions
main/                         C4 summaries plus OOF predictions
ablation/                     C5 summaries plus OOF predictions
statistics/ and xai/          C6 inference, calibration, SHAP, and Grad-CAM
secondary_holdout/            C7 aggregate public evidence
```

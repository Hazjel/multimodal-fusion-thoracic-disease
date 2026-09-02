# Incremental Value of Structured Metadata in Chest X-ray Classification

This repository contains the reproducible research pipeline and canonical
evidence for a final-year thesis on binary Normal-versus-Abnormal
classification using NIH ChestX-ray14. The study compares a metadata-only MLP
(S1), an image-only ResNet-50 (S2), and intermediate multimodal fusion (S3).

The central question is not whether a multimodal model can obtain a high score,
but whether four public NIH metadata fields add predictive discrimination after
image features are already available.

## Research status

All protocol stages are complete.

| Stage | Status | Purpose |
|---|---|---|
| C0 | PASS | Freeze protocol, manifests, environment, and guardrails |
| C1 | Complete | MLP, RealMLP, and TabM tabular characterization |
| C2 | Complete | DenseNet-121, ResNet-50, and EfficientNet-B0 screening |
| C3 | LOCKED | Select ResNet-50 with ImageNet initialization |
| C4 | Complete | Paired five-fold OOF evaluation of S1, S2, and S3 |
| C5 | Complete | Metadata ablation for feature sets A to D |
| C6 | Complete | Cluster bootstrap, calibration, SHAP, and Grad-CAM |
| C7 | Complete | Secondary official-test evaluation with prior-exposure disclosure |

Scientific protocol hash:

```text
d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32
```

The protocol is frozen as v1.0.0. The official NIH test partition is secondary
evidence because it influenced exploratory work before protocol freeze. Primary
evidence remains patient-level five-fold out-of-fold evaluation on the official
training pool.

## Main result

| Scenario | Pooled ROC-AUC | Average Precision | Brier score |
|---|---:|---:|---:|
| S1, metadata only | 0.621382 | 0.519234 | 0.240854 |
| S2, image only | 0.752390 | 0.672896 | 0.200192 |
| S3, multimodal fusion | 0.753367 | 0.671663 | 0.200396 |

The paired primary difference was:

```text
ROC-AUC(S3 - S2) = 0.000977
95% patient-cluster bootstrap CI = [-0.000375, 0.002317]
```

Under the prespecified comparison, the available evidence did not clearly
separate S3 from S2. This is not a formal equivalence claim and does not imply
that structured clinical data are universally uninformative.

## Dataset

NIH ChestX-ray14 is publicly available from:

- [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC), the
  canonical distribution;
- [NIH Chest X-rays on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data),
  a public mirror.

Raw radiographs are not redistributed in this repository. Set
`NIH_DATASET_ROOT` to a local extracted copy containing:

```text
Data_Entry_2017.csv
train_val_list.txt
test_list.txt
images_001/images/*.png
...
images_012/images/*.png
```

The exact local data snapshot used for this study is documented in
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).

## Scenarios

| Scenario | Input | Canonical architecture |
|---|---|---|
| S1 | Age, sex, view position, follow-up number | MLP `4 -> 64 -> 128 -> 1` |
| S2 | Chest radiograph | ResNet-50 `-> GAP -> 512 -> 1` |
| S3 | Image and metadata | Concatenation `640 -> 256 -> 128 -> 1` |

C1 is a characterization benchmark. RealMLP obtained the highest tabular point
estimate, but the canonical MLP remains S1 and the S3 metadata encoder because
that architecture was prespecified as the end-to-end fusion representation.

## Installation

Python 3.12 and a CUDA-capable PyTorch environment were used for the canonical
runs. Install the frozen C0 dependencies:

```powershell
python -m pip install -r requirements-c0.txt
```

Set dataset and project paths when the defaults do not match your checkout:

```powershell
$env:NIH_DATASET_ROOT = "D:\path\to\nih-chest-xrays"
$env:NIH_PROJECT_ROOT = "$PWD"
```

On Linux or macOS:

```bash
export NIH_DATASET_ROOT=/path/to/nih-chest-xrays
export NIH_PROJECT_ROOT="$PWD"
```

## Entry points

Inspect the frozen study state:

```powershell
python run_experiment.py status
python run_experiment.py c0
```

Canonical stages are executed through `run_experiment.py`, not through the
archived training notebooks:

```powershell
python run_experiment.py benchmark-tabular --model all --device cuda
python run_experiment.py screen-image --backbone all --pretraining imagenet
python run_experiment.py main --scenario all
python run_experiment.py ablate --scenario both --feature-set all
python run_experiment.py c6 --component all --device cuda
```

C7 has already been completed under its recorded access event. The test guard
prevents treating the official partition as a fresh holdout or silently opening
it during C1 to C6.

## Repository structure

```text
configs/                    Frozen runtime configuration
docs/                       Protocol, literature matrix, and provenance notes
notebooks/
  archive/pre_protocol/     Historical notebooks, not canonical evidence
  exploratory/              Non-canonical model experiments
  canonical_reports/        Read-only reports generated from canonical outputs
results/
  canonical/<hash>/         Canonical manifests, OOF evidence, and summaries
  exploratory/              Explicitly non-canonical outputs
  legacy/                   Outputs from superseded designs
src/                        Data, model, training, evaluation, and XAI modules
tests/                      C0 and protocol regression tests
run_experiment.py           Canonical command-line entry point
```

Start with the [canonical evidence index](results/canonical/README.md) and
[artifact status policy](results/STATUS.md). The complete scientific contract is
in [Canonical Execution Protocol v1.0.0-rc2](docs/CANONICAL_PROTOCOL_v1.0.0-rc2.md).

## Reproducibility and publication policy

- Splits and bootstrap resampling group by Patient ID.
- S2 and S3 use matched image-branch initialization within each fold.
- Checkpoint selection and early stopping use validation ROC-AUC.
- The primary uncertainty interval uses 2,000 paired patient-cluster bootstrap
  replicates on pooled OOF predictions.
- Checkpoints, scalers, raw images, per-fold C5 state, and official-test
  row-level predictions remain local.
- Published evidence includes protocol files, split manifests, aggregate OOF
  predictions, metrics, calibration outputs, XAI summaries, and checksums.
- Archived and exploratory notebooks must not be used as sources for canonical
  manuscript results.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). When using the
code or evidence bundle, cite the associated manuscript and the original NIH
ChestX-ray14 publication.

## License

No software reuse license has been selected yet. Public visibility alone does
not grant permission to copy, modify, or redistribute the code. A license should
be chosen by the repository owner before encouraging third-party reuse.

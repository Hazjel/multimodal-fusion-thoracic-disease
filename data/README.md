# Data directory

Raw NIH ChestX-ray14 images and metadata are not redistributed in this
repository. Configure `NIH_DATASET_ROOT` to point to a local copy as described
in the project README and `docs/DATA_PROVENANCE.md`.

The canonical C1-C7 pipeline does not consume pre-generated row-level train and
test CSV files from `data/processed/`. It derives fold-specific inputs from the
source metadata using the frozen patient-level manifests in
`results/canonical/<protocol_hash>/`. The former `X_tabular_*`, `img_*`, and
`y_*` CSV files were outputs of an archived pre-protocol row-split notebook and
were removed from the publication snapshot to avoid confusion with canonical
evidence. Their historical provenance remains available in Git history and in
`notebooks/archive/pre_protocol/02_row_split_preprocessing_legacy.ipynb`.

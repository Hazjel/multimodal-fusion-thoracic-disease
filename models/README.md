# Local model artifacts

This directory is reserved for local legacy or non-canonical model assets.
Files such as `model.pth.tar` and the historical `scaler_tabular.pkl` are not
part of the publication evidence bundle and must not be used to reproduce the
reported C1-C7 results.

Canonical metadata preprocessing fits a separate `StandardScaler` on the
training portion of each patient-level fold. Canonical model identities,
predictions, metrics, and reproducibility records are indexed under
`results/canonical/`.

Large checkpoints and fitted scalers remain local and are ignored by Git.

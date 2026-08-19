# Canonical Execution Protocol v1.0.0-rc2

Status: **FINAL research design / freeze candidate**. The protocol becomes
`v1.0.0 — FROZEN` only after C0 passes and immutable artifacts are generated.

## Canonical evidence

- Task: NIH ChestX-ray14, binary Normal versus Abnormal.
- Unit of prediction: image/examination; grouping unit: Patient ID.
- S1: `4 → 64 → 128 → 1` metadata MLP.
- S2: selected CNN, GAP, `native → 512 → 1`.
- S3: image 512 plus metadata 128, `640 → 256 → 128 → 1`.
- Primary evidence: five-fold `StratifiedGroupKFold`, seed 42, official training pool only.
- Official test: secondary holdout with prior-exposure disclosure; blocked until C7.

All scientific values, preprocessing, selection rules, estimands, XAI rules,
and immutable manifest hashes are serialized in the generated `protocol.json`.
Only its `scientific_spec` contributes to `protocol_hash`; Git/environment
provenance contributes to each run's `semantic_config_hash`.

## C0 and freeze

```powershell
# commit implementation C0 so the Git SHA is stable
python run_experiment.py c0
python run_experiment.py freeze
```

Freeze refuses a failed report, dirty worktree, missing PyTabKit, or a C0
report produced from a different commit. Smoke-test AUC is never used to alter
the design.

## Subsequent execution

```powershell
python run_experiment.py cv --protocol-dir results/canonical/<protocol_hash> --scenario S1
python run_experiment.py cv --protocol-dir results/canonical/<protocol_hash> --scenario S2 --backbone densenet121
```

Full CV is rejected unless `protocol.json` has status `FROZEN` and the fold
manifest checksum still matches. Legacy and exploratory results are not read
by the canonical runner.

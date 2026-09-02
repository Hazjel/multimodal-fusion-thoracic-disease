# Research Artifact Status

This document separates canonical evidence from exploratory and legacy results.

## Canonical protocol

- Protocol: **v1.0.0 - FROZEN**.
- Scientific protocol hash:
  `d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32`.
- C0 implementation acceptance: **PASS**.
- C1 tabular benchmark: **complete**, 15/15 runs.
- C2 ImageNet backbone screening: **complete**, 15/15 runs.
- C3 model lock: **complete**, ResNet-50 with ImageNet initialization.
- C4 primary S1/S2/S3 experiments: **complete**, 15/15 runs.
- C5 metadata ablation: **complete**, 30/30 runs.
- C6 statistics and OOF XAI: **complete**.
- C7 secondary official holdout: **complete**, with prior-exposure disclosure.

The canonical evidence bundle is documented in
[`canonical/d423.../`](canonical/d42337690181f1054297f514934ad0c98bb718223bc06d8de5569f40a184ee32/README.md).
Only artifacts with the expected `_SUCCESS` markers, valid schemas, complete
registry entries, and matching provenance are canonical evidence.

The primary paired comparison yielded a pooled OOF ROC-AUC difference of
`S3-D - S2 = 0.000977`, with a patient-cluster bootstrap 95% confidence
interval of `[-0.000375, 0.002317]`. The interval includes zero, so the
available evidence did not establish a clear discrimination improvement from
adding all four metadata variables to the image model.

## Exploratory

Exploratory artifacts may document development history, but they:

- are not primary comparative evidence;
- must not be mixed into canonical result tables;
- must not change the frozen candidate set or hyperparameters; and
- require disclosure when an older experiment used the official test set.

This category includes older attention or gated-fusion experiments, earlier
backbone comparisons, and the legacy complementarity analysis.

## Legacy

The following artifacts are incompatible with the canonical protocol:

- row-level or non-patient-grouped splits;
- earlier metadata architectures;
- the former multilabel S4 scenario;
- checkpoints with incompatible preprocessing, initialization, or test policy;
- silent CheXNet fallback or weights without auditable provenance.

## Governance rules

1. C1 and C2 require a frozen scientific protocol.
2. C4 and C5 require an immutable `model_lock.json`.
3. A proposal amendment may document a frozen selection outcome without
   changing the scientific protocol hash.
4. A scientific protocol amendment requires a new version and hash; evidence
   from different versions is not combined as canonical.
5. An implementation bug fix changes the implementation commit and semantic
   configuration hash. Affected runs must be invalidated and repeated.
6. C1-C6 cannot read the official test partition. C7 is the single authorized
   secondary-holdout access recorded by the protocol.

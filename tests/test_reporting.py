from __future__ import annotations

import unittest
from pathlib import Path

from src.reporting.canonical import (
    load_canonical_context,
    load_oof_predictions,
    metric_table,
)


class CanonicalReportingTests(unittest.TestCase):
    def test_completed_canonical_oof_artifacts_are_auditable(self):
        context = load_canonical_context(Path(__file__).resolve())
        main_dir = context.protocol_dir / "main"
        models = {
            "S1": load_oof_predictions(
                context,
                main_dir / "S1-canonical_mlp-not_applicable-D-oof.csv",
                expected_stage_directory="main",
            ),
            "S2": load_oof_predictions(
                context,
                main_dir / "S2-resnet50-imagenet-D-oof.csv",
                expected_stage_directory="main",
            ),
            "S3": load_oof_predictions(
                context,
                main_dir / "S3-resnet50-imagenet-D-oof.csv",
                expected_stage_directory="main",
            ),
        }
        table = metric_table(models)
        self.assertEqual(len(table), 3)
        self.assertEqual(set(table["n_images"]), {len(context.folds)})
        self.assertGreater(table.loc["S3", "roc_auc_pooled"], 0.5)


if __name__ == "__main__":
    unittest.main()

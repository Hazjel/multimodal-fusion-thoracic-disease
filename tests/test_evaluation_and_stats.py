from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation import calibration_table, compute_metrics, validate_prediction_frame
from src.evaluation.stats import paired_patient_cluster_bootstrap, select_cnn_candidate


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_uniform_calibration(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = compute_metrics(probabilities, labels)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        table = calibration_table(probabilities, labels, n_bins=10)
        self.assertEqual(len(table), 10)
        self.assertTrue(table.loc[table["count"] == 0, "mean_score"].isna().all())

    def test_prediction_schema_rejects_duplicates(self):
        columns = [
            "image_index", "patient_id", "fold", "split", "true_label",
            "probability", "prediction_0_5", "model", "scenario", "feature_set",
            "pretraining", "seed", "checkpoint_epoch", "protocol_hash",
            "semantic_config_hash", "run_id",
        ]
        row = ["a.png", 1, 0, "validation", 0, 0.2, 0, "m", "S2", "D", "imagenet", 42, 1, "p", "s", "r"]
        with self.assertRaises(ValueError):
            validate_prediction_frame(pd.DataFrame([row, row], columns=columns))

    def test_patient_cluster_bootstrap_is_deterministic(self):
        rows = []
        for patient in range(20):
            label = patient % 2
            rows.append({
                "patient_id": patient,
                "true_label": label,
                "a": 0.8 if label else 0.2,
                "b": 0.7 if label else 0.3,
            })
        frame = pd.DataFrame(rows)
        first = paired_patient_cluster_bootstrap(
            frame, probability_a="a", probability_b="b", n_boot=100, seed=42
        )
        second = paired_patient_cluster_bootstrap(
            frame, probability_a="a", probability_b="b", n_boot=100, seed=42
        )
        self.assertEqual(first, second)
        self.assertTrue(first["conditional_on_fitted_cv_models"])

    def test_selection_considers_every_candidate(self):
        predictions = {}
        for offset, name in enumerate(("dense", "resnet", "efficient")):
            rows = []
            for patient in range(30):
                label = patient % 2
                base = 0.65 if label else 0.35
                rows.append({
                    "image_index": f"{patient}.png",
                    "patient_id": patient,
                    "true_label": label,
                    "probability": base + offset * 1e-4,
                })
            predictions[name] = pd.DataFrame(rows)
        result = select_cnn_candidate(
            predictions,
            fold_aucs={name: [0.7] * 5 for name in predictions},
            trainable_parameters={"dense": 3, "resnet": 2, "efficient": 1},
            median_training_seconds={"dense": 3.0, "resnet": 2.0, "efficient": 1.0},
            n_boot=50,
        )
        self.assertEqual(set(result["heuristic_candidate_set"]), set(predictions))
        self.assertEqual(result["selected"], "efficient")


if __name__ == "__main__":
    unittest.main()

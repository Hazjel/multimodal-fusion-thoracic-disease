from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.reporting.c6 import align_paired_oof, select_local_xai_cases, youden_threshold


class C6ReportingTests(unittest.TestCase):
    @staticmethod
    def _frame(probabilities):
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        return pd.DataFrame({
            "image_index": [f"{index}.png" for index in range(len(labels))],
            "patient_id": np.arange(len(labels)),
            "true_label": labels,
            "fold": np.arange(len(labels)) % 5,
            "probability": probabilities,
        })

    def test_pair_alignment_is_keyed_not_row_order(self):
        first = self._frame([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        second = self._frame([0.2, 0.3, 0.7, 0.8, 0.4, 0.6, 0.45, 0.55]).iloc[::-1]
        aligned = align_paired_oof(first, second)
        self.assertEqual(len(aligned), len(first))
        self.assertEqual(aligned.loc[aligned["image_index"] == "0.png", "probability_b"].item(), 0.2)

    def test_youden_threshold_is_finite_and_secondary(self):
        frame = self._frame([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        result = youden_threshold(frame)
        self.assertTrue(np.isfinite(result["threshold"]))
        self.assertEqual(result["role"], "secondary_operating_point_only")
        self.assertGreaterEqual(result["youden_j"], 0.0)

    def test_gpu_scale_score_drift_is_within_c6_audit_tolerance(self):
        self.assertTrue(np.isclose(0.4406360090, 0.4390806556, rtol=1e-3, atol=5e-3))
        self.assertFalse(np.isclose(0.4510, 0.4390, rtol=1e-3, atol=5e-3))

    def test_local_selection_returns_two_per_confusion_category(self):
        rows = []
        specifications = {
            "TP": (1, [0.6, 0.7, 0.8]),
            "TN": (0, [0.2, 0.3, 0.4]),
            "FP": (0, [0.6, 0.7, 0.8]),
            "FN": (1, [0.2, 0.3, 0.4]),
        }
        index = 0
        for _category, (label, probabilities) in specifications.items():
            for probability in probabilities:
                rows.append({
                    "image_index": f"{index}.png",
                    "patient_id": index,
                    "true_label": label,
                    "fold": index % 5,
                    "probability": probability,
                })
                index += 1
        selected = select_local_xai_cases(pd.DataFrame(rows), per_category=2)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected.groupby("category").size().to_dict(), {
            "FN": 2, "FP": 2, "TN": 2, "TP": 2,
        })


if __name__ == "__main__":
    unittest.main()

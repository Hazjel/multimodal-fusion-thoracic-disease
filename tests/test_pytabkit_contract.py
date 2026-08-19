from __future__ import annotations

import unittest
import pickle

import numpy as np
import pandas as pd

from src.training.tabular_benchmark import (
    REALMLP_FROZEN_PARAMS,
    TABM_FROZEN_PARAMS,
    build_pytabkit_estimator,
)


class PyTabKitContractTests(unittest.TestCase):
    def test_frozen_estimators_expose_required_parameters_and_explicit_validation(self):
        try:
            from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
        except ImportError as error:
            self.fail(f"pytabkit==1.7.3 is required for C0: {error}")

        real = build_pytabkit_estimator("realmlp", device="cpu", verbosity=0)
        tabm = build_pytabkit_estimator("tabm", device="cpu", verbosity=0)
        for estimator, expected in (
            (real, REALMLP_FROZEN_PARAMS),
            (tabm, TABM_FROZEN_PARAMS),
        ):
            params = estimator.get_params(deep=False)
            for key, value in expected.items():
                self.assertEqual(params[key], value)

        # API smoke only: one epoch, tiny data, explicit X_val/y_val. Its score
        # is deliberately ignored and never enters a design decision.
        rng = np.random.RandomState(42)
        frame = pd.DataFrame({
            "Age": rng.normal(50, 10, 48),
            "Gender": pd.Categorical(["F", "M"] * 24),
            "View": pd.Categorical(["AP", "PA", "AP"] * 16),
            "Follow": np.arange(48) % 5,
        })
        labels = np.array([0, 1] * 24)
        smoke = RealMLP_TD_Classifier(
            device="cpu",
            n_cv=1,
            n_refit=0,
            n_epochs=1,
            batch_size=16,
            hidden_sizes=[8],
            val_metric_name="cross_entropy",
            use_ls=False,
            calibration_method=None,
            random_state=42,
            verbosity=0,
        )
        smoke.fit(
            frame.iloc[:32],
            labels[:32],
            X_val=frame.iloc[32:],
            y_val=labels[32:],
            cat_col_names=["Gender", "View"],
        )
        probabilities = smoke.predict_proba(frame.iloc[32:])
        self.assertEqual(probabilities.shape, (16, 2))
        self.assertGreater(len(pickle.dumps(smoke)), 0)

        # Tiny TabM runtime fit verifies explicit validation is accepted by
        # the installed version. Smoke-test scores are never research data.
        tabm_smoke = TabM_D_Classifier(
            device="cpu",
            n_cv=1,
            n_refit=0,
            arch_type="tabm-mini",
            tabm_k=2,
            num_emb_type="none",
            batch_size=16,
            lr=2e-3,
            n_epochs=1,
            patience=1,
            val_metric_name="cross_entropy",
            random_state=42,
            verbosity=0,
        )
        tabm_smoke.fit(
            frame.iloc[:32],
            labels[:32],
            X_val=frame.iloc[32:],
            y_val=labels[32:],
            cat_col_names=["Gender", "View"],
        )
        tabm_probabilities = tabm_smoke.predict_proba(frame.iloc[32:])
        self.assertEqual(tabm_probabilities.shape, (16, 2))
        self.assertGreater(len(pickle.dumps(tabm_smoke)), 0)


if __name__ == "__main__":
    unittest.main()

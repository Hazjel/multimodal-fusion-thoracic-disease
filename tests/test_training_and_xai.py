from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.models.architectures import TabularMLP
from src.training import capture_rng_state, restore_rng_state, train
from src.xai import build_shap_background, make_shap_explainer, proportional_oof_indices


class TinyTabularDataset(Dataset):
    def __init__(self):
        raw = np.arange(256, dtype=np.float32).reshape(64, 4)
        self.scaler = StandardScaler().fit(raw[:48])
        self.values = self.scaler.transform(raw).astype(np.float32)
        self.labels = (np.arange(64) % 2).astype(np.float32)
        self.modalities = frozenset({"tabular"})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "tabular": torch.from_numpy(self.values[index]),
            "label_binary": torch.tensor(self.labels[index]),
            "image_index": f"{index}.png",
            "patient_id": index,
        }


class RecordingFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_images = []

    def forward(self, image, tabular):
        self.seen_images.append(image.detach().cpu().clone())
        return tabular.sum(dim=1, keepdim=True) + image.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)


class TrainingAndXAITests(unittest.TestCase):
    def test_checkpoint_contains_full_resume_state_and_best_auc(self):
        dataset = TinyTabularDataset()
        train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        validation_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        model = TabularMLP()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            train(
                model,
                train_loader,
                validation_loader,
                torch.tensor([1.0]),
                "S1",
                torch.device("cpu"),
                run_dir=run_dir,
                run_metadata={"run_id": "c0-smoke"},
                resume=False,
            )
            best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            required = {
                "model_state", "optimizer_state", "scheduler_state", "grad_scaler_state",
                "rng_state", "best_validation_auc", "best_epoch", "patience_count",
                "history", "run_metadata",
            }
            self.assertTrue(required.issubset(last))
            self.assertAlmostEqual(
                best["best_validation_auc"],
                max(row["validation_auc"] for row in last["history"]),
            )

    def test_rng_state_round_trip(self):
        torch.manual_seed(42)
        state = capture_rng_state()
        expected = torch.rand(3)
        restore_rng_state(state)
        actual = torch.rand(3)
        self.assertTrue(torch.equal(expected, actual))

    def test_shap_background_is_scaled_and_image_conditioned(self):
        dataset = TinyTabularDataset()
        background = build_shap_background(dataset, n_samples=8)
        self.assertEqual(background.shape, (8, 4))
        model = RecordingFusion()
        fixed_image = torch.ones(1, 3, 8, 8)
        explainer = make_shap_explainer(
            model,
            background,
            torch.device("cpu"),
            fixed_image=fixed_image,
        )
        explainer.model.f(background[:2])
        self.assertTrue(model.seen_images)
        self.assertTrue(all(torch.allclose(batch, torch.ones_like(batch)) for batch in model.seen_images))

    def test_proportional_oof_sampling(self):
        labels = np.array([0] * 80 + [1] * 20)
        selected = proportional_oof_indices(labels, n_samples=40, seed=42)
        self.assertEqual(int((labels[selected] == 0).sum()), 32)
        self.assertEqual(int((labels[selected] == 1).sum()), 8)


if __name__ == "__main__":
    unittest.main()

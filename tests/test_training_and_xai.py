from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from configs.config import cfg as base_cfg
from src.models.architectures import TabularMLP
from src.training import (
    capture_dataloader_state,
    capture_rng_state,
    restore_dataloader_state,
    restore_rng_state,
    train,
)
from src.xai import build_shap_background, make_shap_explainer, proportional_oof_indices


class TinyTabularDataset(Dataset):
    def __init__(self, *, augment=False):
        raw = np.arange(256, dtype=np.float32).reshape(64, 4)
        self.scaler = StandardScaler().fit(raw[:48])
        self.values = self.scaler.transform(raw).astype(np.float32)
        self.labels = (np.arange(64) % 2).astype(np.float32)
        self.modalities = frozenset({"tabular"})
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        value = torch.from_numpy(self.values[index]).clone()
        if self.augment:
            value = value + 0.01 * torch.rand_like(value)
        return {
            "tabular": value,
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
    @staticmethod
    def _loaders(seed=42, num_workers=0):
        train_dataset = TinyTabularDataset(augment=True)
        validation_dataset = TinyTabularDataset(augment=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=num_workers,
            persistent_workers=False,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=16,
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 1_000_000),
            num_workers=num_workers,
            persistent_workers=False,
        )
        return train_loader, validation_loader

    def test_checkpoint_contains_full_resume_state_and_best_auc(self):
        train_loader, validation_loader = self._loaders()
        model = TabularMLP()
        test_cfg = replace(
            base_cfg,
            train=replace(base_cfg.train, num_epochs=2, use_amp=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with patch("src.training.cfg", test_cfg):
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
                "rng_state", "dataloader_state", "best_validation_auc", "best_epoch",
                "patience_count", "history", "timing_seconds", "run_metadata",
            }
            self.assertTrue(required.issubset(last))
            self.assertAlmostEqual(
                best["best_validation_auc"],
                max(row["validation_auc"] for row in last["history"]),
            )

    def test_interrupted_resume_matches_uninterrupted_training(self):
        two_epoch_cfg = replace(
            base_cfg,
            train=replace(base_cfg.train, num_epochs=2, use_amp=False),
        )
        one_epoch_cfg = replace(
            base_cfg,
            train=replace(base_cfg.train, num_epochs=1, use_amp=False),
        )
        torch.manual_seed(2026)
        initial_model = TabularMLP()
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in initial_model.state_dict().items()
        }

        def fresh_model():
            model = TabularMLP()
            model.load_state_dict(initial_state)
            return model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted_dir = root / "uninterrupted"
            resumed_dir = root / "resumed"
            train_loader, validation_loader = self._loaders(seed=71)
            with patch("src.training.cfg", two_epoch_cfg):
                train(
                    fresh_model(), train_loader, validation_loader,
                    torch.tensor([1.0]), "S1", torch.device("cpu"),
                    run_dir=uninterrupted_dir,
                    run_metadata={"run_id": "resume-equivalence"},
                    resume=False,
                )

            first_loader, first_validation = self._loaders(seed=71)
            with patch("src.training.cfg", one_epoch_cfg):
                train(
                    fresh_model(), first_loader, first_validation,
                    torch.tensor([1.0]), "S1", torch.device("cpu"),
                    run_dir=resumed_dir,
                    run_metadata={"run_id": "resume-equivalence"},
                    resume=False,
                )
            restart_loader, restart_validation = self._loaders(seed=71)
            with patch("src.training.cfg", two_epoch_cfg):
                train(
                    fresh_model(), restart_loader, restart_validation,
                    torch.tensor([1.0]), "S1", torch.device("cpu"),
                    run_dir=resumed_dir,
                    run_metadata={"run_id": "resume-equivalence"},
                    resume=True,
                )

            uninterrupted = torch.load(
                uninterrupted_dir / "last.pt", map_location="cpu", weights_only=False
            )
            resumed = torch.load(
                resumed_dir / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(uninterrupted["history"], resumed["history"])
            for name, expected in uninterrupted["model_state"].items():
                self.assertTrue(torch.equal(expected, resumed["model_state"][name]), name)

    def test_rng_state_round_trip(self):
        torch.manual_seed(42)
        state = capture_rng_state()
        expected = torch.rand(3)
        restore_rng_state(state)
        actual = torch.rand(3)
        self.assertTrue(torch.equal(expected, actual))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_rng_and_dataloader_resume_accept_cuda_mapped_checkpoint_states(self):
        """Regression: map_location='cuda' must not break CPU RNG restoration."""
        rng_state = capture_rng_state()
        cuda_mapped_rng = dict(rng_state)
        cuda_mapped_rng["torch_cpu"] = rng_state["torch_cpu"].cuda()
        cuda_mapped_rng["torch_cuda"] = [
            value.cuda() for value in rng_state["torch_cuda"]
        ]
        restore_rng_state(cuda_mapped_rng)

        train_loader, validation_loader = self._loaders()
        loader_state = capture_dataloader_state(train_loader, validation_loader)
        cuda_mapped_loaders = {
            name: value.cuda() for name, value in loader_state.items()
        }
        restore_dataloader_state(
            cuda_mapped_loaders, train_loader, validation_loader
        )

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

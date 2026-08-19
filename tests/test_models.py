from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.models.architectures import (
    BACKBONE_NATIVE_DIMS,
    ImageCNN,
    ImageEncoder,
    MultimodalFusion,
    TabularMLP,
    build_s2_s3_pair,
    image_initial_hashes,
)
from src.protocol.contracts import state_dict_sha256


def _bn_buffer_snapshot(modules):
    return [
        (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
        for module in modules
    ]


class CanonicalModelTests(unittest.TestCase):
    def test_s1_architecture(self):
        model = TabularMLP()
        model.eval()
        self.assertEqual(tuple(model(torch.randn(3, 4)).shape), (3, 1))

    def test_s2_s3_initial_image_state_is_equal_for_all_backbones(self):
        for backbone in BACKBONE_NATIVE_DIMS:
            with self.subTest(backbone=backbone):
                s2, s3 = build_s2_s3_pair(
                    backbone_name=backbone,
                    fold=0,
                    pretraining="none",
                )
                self.assertEqual(image_initial_hashes(s2), image_initial_hashes(s3))
                del s2, s3
                gc.collect()

    def test_structural_freeze_and_batchnorm_buffers(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for backbone in BACKBONE_NATIVE_DIMS:
            with self.subTest(backbone=backbone):
                model = ImageCNN(backbone_name=backbone, pretraining="none").to(device)
                branch = model.branch
                self.assertTrue(all(not p.requires_grad for m in branch.frozen_stage_modules() for p in m.parameters()))
                self.assertTrue(all(p.requires_grad for m in branch.trainable_stage_modules() for p in m.parameters()))
                frozen_before = _bn_buffer_snapshot(branch.frozen_batchnorm_modules())
                trainable_before = _bn_buffer_snapshot(branch.trainable_batchnorm_modules())
                model.train()
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                output = model(image=torch.randn(2, 3, 64, 64, device=device))
                output.mean().backward()
                optimizer.step()
                frozen_after = _bn_buffer_snapshot(branch.frozen_batchnorm_modules())
                trainable_after = _bn_buffer_snapshot(branch.trainable_batchnorm_modules())
                self.assertTrue(all(not module.training for module in branch.frozen_batchnorm_modules()))
                self.assertTrue(all(torch.equal(a, b) for before, after in zip(frozen_before, frozen_after) for a, b in zip(before, after)))
                self.assertTrue(any(not torch.equal(a[2], b[2]) for a, b in zip(trainable_before, trainable_after)))
                del model, optimizer, output
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    def test_fusion_output_shape(self):
        model = MultimodalFusion(backbone_name="densenet121", pretraining="none")
        model.eval()
        with torch.no_grad():
            output = model(image=torch.randn(2, 3, 64, 64), tabular=torch.randn(2, 4))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_chexnet_missing_file_is_hard_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                ImageEncoder(
                    "densenet121",
                    pretraining="chexnet",
                    chexnet_path=Path(directory) / "missing.pt",
                )

    def test_real_chexnet_checkpoint_matches_frozen_checksum(self):
        model = ImageEncoder("densenet121", pretraining="chexnet")
        self.assertEqual(len(model.weight_file_checksum), 64)

    def test_imagenet_weight_identifiers_and_forward(self):
        for backbone in BACKBONE_NATIVE_DIMS:
            with self.subTest(backbone=backbone):
                model = ImageEncoder(backbone, pretraining="imagenet")
                self.assertNotEqual(model.weight_identifier, "NONE")
                self.assertEqual(len(model.pretrained_state_checksum), 64)
                model.eval()
                with torch.no_grad():
                    output = model(torch.randn(1, 3, 64, 64))
                self.assertEqual(tuple(output.shape), (1, 512))
                del model
                gc.collect()


if __name__ == "__main__":
    unittest.main()

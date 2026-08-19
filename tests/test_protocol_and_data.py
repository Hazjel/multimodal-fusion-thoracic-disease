from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import InterpolationMode

from configs.config import cfg as base_cfg
from src.data.dataset import (
    get_transforms,
    load_and_prepare_metadata,
    load_official_training_pool,
    make_fold_dataloaders,
)
from src.protocol.contracts import (
    atomic_write_json,
    file_sha256,
    protocol_hash,
    semantic_config_hash,
)
from src.protocol.environment import environment_hash
from src.protocol.execution_environment import (
    EnvironmentConsistencyError,
    assert_registered_runs_match_environment,
    ensure_stage_environment,
)
from src.protocol.chexnet import (
    SAFE_OFFICIAL_TEST_VALUE,
    evaluate_provenance_declaration,
)
from src.protocol.guards import OfficialTestAccessError, assert_official_test_access
from src.protocol.manifests import (
    audit_manifests,
    create_deployment_manifest,
    create_primary_fold_manifest,
)
from src.protocol.spec import build_scientific_spec
from src.protocol.stages import (
    StageGateError,
    load_frozen_protocol,
    validate_cv_request,
)
from src.protocol.registry import upsert_registry


class ProtocolIdentityTests(unittest.TestCase):
    def test_protocol_hash_excludes_implementation(self):
        scientific = {"task": "binary", "folds": 5}
        first = protocol_hash(scientific)
        second = protocol_hash(dict(scientific))
        self.assertEqual(first, second)
        a = semantic_config_hash(
            protocol_hash_value=first,
            selected_architecture="densenet121",
            weight_checksum="weight",
            fold=0,
            feature_set="D",
            resolved_runtime_config={"batch": 16},
            environment_hash="env",
            implementation_commit="commit-a",
        )
        b = semantic_config_hash(
            protocol_hash_value=first,
            selected_architecture="densenet121",
            weight_checksum="weight",
            fold=0,
            feature_set="D",
            resolved_runtime_config={"batch": 16},
            environment_hash="env",
            implementation_commit="commit-b",
        )
        self.assertNotEqual(a, b)

    def test_environment_hash_excludes_implementation_commit(self):
        first = environment_hash({"implementation_commit": "a", "python": "3.12"})
        second = environment_hash({"implementation_commit": "b", "python": "3.12"})
        self.assertEqual(first, second)

    def _frozen_fixture(self, root: Path):
        folds = root / "folds.csv"
        deployment = root / "deployment_split.csv"
        folds.write_text("image_index\n", encoding="utf-8")
        deployment.write_text("image_index\n", encoding="utf-8")
        scientific = build_scientific_spec(
            fold_manifest_hash=file_sha256(folds),
            deployment_split_hash=file_sha256(deployment),
        )
        protocol = {
            "status": "FROZEN",
            "protocol_hash": protocol_hash(scientific),
            "scientific_spec": scientific,
            "provenance": {},
        }
        atomic_write_json(root / "protocol.json", protocol)
        return protocol

    def test_frozen_runtime_and_scientific_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._frozen_fixture(root)
            loaded = load_frozen_protocol(root)
            self.assertEqual(loaded["protocol_hash"], protocol["protocol_hash"])

            mismatched_cfg = replace(
                base_cfg,
                train=replace(base_cfg.train, lr_backbone=5e-4),
            )
            with patch("src.protocol.stages.cfg", mismatched_cfg):
                with self.assertRaises(RuntimeError):
                    load_frozen_protocol(root)

            tampered = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
            tampered["scientific_spec"]["evaluation"]["threshold"] = 0.4
            atomic_write_json(root / "protocol.json", tampered)
            with self.assertRaises(StageGateError):
                load_frozen_protocol(root)

    def test_stage_gate_blocks_main_before_model_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._frozen_fixture(root)
            validate_cv_request(
                stage="C2",
                scenario="S2",
                backbone="densenet121",
                pretraining="imagenet",
                feature_set="D",
                protocol_dir=root,
            )
            candidate_path = root / "screening" / "image" / "model_lock_candidate.json"
            atomic_write_json(candidate_path, {
                "status": "CHEXNET_AUDIT_REQUIRED",
                "protocol_hash": protocol["protocol_hash"],
                "selected_backbone": "densenet121",
            })
            with self.assertRaises(StageGateError):
                validate_cv_request(
                    stage="C2",
                    scenario="S2",
                    backbone="densenet121",
                    pretraining="chexnet",
                    feature_set="D",
                    protocol_dir=root,
                )
            with self.assertRaises(StageGateError):
                validate_cv_request(
                    stage="C4",
                    scenario="S3",
                    backbone="densenet121",
                    pretraining="imagenet",
                    feature_set="D",
                    protocol_dir=root,
                )

    def test_stage_environment_lock_rejects_mixed_c2_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._frozen_fixture(root)
            environment = {"gpu": "test-gpu", "packages": {"torch": "test"}}
            ensure_stage_environment(
                protocol_dir=root,
                stage="C2",
                protocol_hash=protocol["protocol_hash"],
                environment_hash="environment-a",
                environment=environment,
                implementation_commit="commit-a",
            )
            ensure_stage_environment(
                protocol_dir=root,
                stage="C2",
                protocol_hash=protocol["protocol_hash"],
                environment_hash="environment-a",
                environment=environment,
                implementation_commit="commit-b",
            )
            with self.assertRaises(EnvironmentConsistencyError):
                ensure_stage_environment(
                    protocol_dir=root,
                    stage="C2",
                    protocol_hash=protocol["protocol_hash"],
                    environment_hash="environment-b",
                    environment=environment,
                    implementation_commit="commit-c",
                )

            upsert_registry(root / "experiment_registry.csv", {
                "run_id": "good-run",
                "phase": "C2",
                "environment_hash": "environment-a",
            })
            self.assertEqual(
                assert_registered_runs_match_environment(
                    protocol_dir=root, stage="C2", run_ids=["good-run"]
                ),
                "environment-a",
            )
            upsert_registry(root / "experiment_registry.csv", {
                "run_id": "bad-run",
                "phase": "C2",
                "environment_hash": "environment-b",
            })
            with self.assertRaises(EnvironmentConsistencyError):
                assert_registered_runs_match_environment(
                    protocol_dir=root, stage="C2", run_ids=["good-run", "bad-run"]
                )

    def test_chexnet_provenance_decision_is_computed_not_trusted(self):
        declaration = {
            "source_url": "https://example.org/repository",
            "source_commit": "a" * 40,
            "training_dataset": "NIH ChestX-ray14",
            "training_split_provenance": "Patient-disjoint source train and validation manifests.",
            "preprocessing": {"resize": 224, "normalization": "documented"},
            "label_mapping": list(base_cfg.data.label_names),
            "official_nih_test_usage": SAFE_OFFICIAL_TEST_VALUE,
            "evidence_urls": ["https://example.org/evidence"],
            "reviewed_by": "Supervisor",
        }
        status, reasons = evaluate_provenance_declaration(declaration)
        self.assertEqual(status, "APPROVED")
        self.assertEqual(reasons, [])
        declaration["official_nih_test_usage"] = "UNKNOWN"
        status, reasons = evaluate_provenance_declaration(declaration)
        self.assertEqual(status, "EXCLUDED")
        self.assertTrue(any("test" in reason.lower() for reason in reasons))


class DataContractTests(unittest.TestCase):
    def _metadata(self):
        return pd.DataFrame({
            "Image Index": ["a.png", "b.png"],
            "Finding Labels": ["No Finding", "Mass"],
            "Patient ID": [1, 2],
            "Patient Age": [32, 414],
            "Patient Gender": ["F", "M"],
            "View Position": ["AP", "PA"],
            "Follow-up #": [0, 2],
        })

    def test_metadata_mapping_and_age_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            self._metadata().to_csv(path, index=False)
            frame = load_and_prepare_metadata(path)
        self.assertEqual(frame["Patient Age"].tolist(), [32.0, 100.0])
        self.assertEqual(frame["gender_encoded"].tolist(), [0.0, 1.0])
        self.assertEqual(frame["view_PA"].tolist(), [0.0, 1.0])
        self.assertEqual(frame["binary_label"].tolist(), [0, 1])

    def test_invalid_category_is_hard_error(self):
        frame = self._metadata()
        frame.loc[0, "Patient Gender"] = "UNKNOWN"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_and_prepare_metadata(path)

    def test_training_pool_loader_does_not_require_test_list(self):
        frame = self._metadata()
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train_val_list.txt"
            train_path.write_text("a.png\nb.png\n", encoding="utf-8")
            prepared_path = Path(directory) / "metadata.csv"
            frame.to_csv(prepared_path, index=False)
            prepared = load_and_prepare_metadata(prepared_path)
            training = load_official_training_pool(prepared, train_path)
        self.assertEqual(set(training["Image Index"]), {"a.png", "b.png"})

    def test_canonical_loaders_have_restorable_generators_and_no_persistent_workers(self):
        frame = self._metadata()
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "metadata.csv"
            frame.to_csv(metadata_path, index=False)
            prepared = load_and_prepare_metadata(metadata_path)
            train_loader, validation_loader, _, _ = make_fold_dataloaders(
                prepared,
                prepared,
                {},
                modalities=("tabular",),
                feature_set="D",
                seed=42,
            )
        self.assertIsNotNone(train_loader.generator)
        self.assertIsNotNone(validation_loader.generator)
        self.assertFalse(train_loader.persistent_workers)
        self.assertFalse(validation_loader.persistent_workers)

    def test_transforms_are_explicit_and_shape_is_canonical(self):
        training = get_transforms(True)
        evaluation = get_transforms(False)
        self.assertEqual(training.transforms[0].interpolation, InterpolationMode.BILINEAR)
        self.assertTrue(training.transforms[0].antialias)
        self.assertEqual(training.transforms[3].interpolation, InterpolationMode.BILINEAR)
        self.assertEqual(evaluation.transforms[0].interpolation, InterpolationMode.BILINEAR)
        self.assertTrue(evaluation.transforms[0].antialias)
        image = Image.fromarray(np.full((300, 300, 3), 127, dtype=np.uint8), mode="RGB")
        self.assertEqual(tuple(training(image).shape), (3, 224, 224))
        self.assertEqual(tuple(evaluation(image).shape), (3, 224, 224))

    def test_patient_grouped_manifests(self):
        rows = []
        for patient in range(30):
            for image_no in range(2):
                rows.append({
                    "Image Index": f"p{patient}-{image_no}.png",
                    "Patient ID": patient,
                    "binary_label": (patient + image_no) % 2,
                })
        training = pd.DataFrame(rows)
        folds = create_primary_fold_manifest(training)
        deployment = create_deployment_manifest(training)
        official_test = pd.DataFrame({"Patient ID": [100, 101]})
        audit = audit_manifests(folds, deployment, official_test)
        self.assertEqual(folds.groupby("patient_id")["fold"].nunique().max(), 1)
        self.assertEqual(deployment.groupby("patient_id")["split"].nunique().max(), 1)
        self.assertEqual(audit["patient_overlap"], 0)

    def test_official_test_is_blocked_before_c7(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path = Path(directory) / "protocol.json"
            protocol_path.write_text(
                json.dumps({"status": "FROZEN", "protocol_hash": "abc"}),
                encoding="utf-8",
            )
            with self.assertRaises(OfficialTestAccessError):
                assert_official_test_access(stage="C2", protocol_path=protocol_path)
            assert_official_test_access(stage="C7", protocol_path=protocol_path)


if __name__ == "__main__":
    unittest.main()

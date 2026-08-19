from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import InterpolationMode

from src.data.dataset import get_transforms, load_and_prepare_metadata
from src.protocol.contracts import protocol_hash, semantic_config_hash
from src.protocol.environment import environment_hash
from src.protocol.guards import OfficialTestAccessError, assert_official_test_access
from src.protocol.manifests import (
    audit_manifests,
    create_deployment_manifest,
    create_primary_fold_manifest,
)


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

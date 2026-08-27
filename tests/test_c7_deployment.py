from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.protocol.contracts import atomic_write_json, file_sha256, protocol_hash
from src.protocol.guards import OfficialTestAccessError, assert_official_test_access
from src.protocol.spec import build_scientific_spec
from src.protocol.stages import validate_c7_prerequisites
from src.training.deployment import (
    OFFICIAL_TEST_CONFIRMATION,
    _secondary_holdout_artifacts,
    claim_official_test_access,
)


class C7GateTests(unittest.TestCase):
    def _complete_fixture(self, root: Path) -> None:
        folds = root / "folds.csv"
        deployment = root / "deployment_split.csv"
        folds.write_text("image_index,patient_id,true_label,fold\n", encoding="utf-8")
        deployment.write_text("image_index,patient_id,true_label,split\n", encoding="utf-8")
        scientific = build_scientific_spec(
            fold_manifest_hash=file_sha256(folds),
            deployment_split_hash=file_sha256(deployment),
        )
        protocol_hash_value = protocol_hash(scientific)
        atomic_write_json(root / "protocol.json", {
            "status": "FROZEN",
            "protocol_hash": protocol_hash_value,
            "scientific_spec": scientific,
            "provenance": {},
        })
        for marker in (
            "screening/tabular/_SUCCESS",
            "screening/image/_SUCCESS",
            "main/_SUCCESS",
            "ablation/_SUCCESS",
            "statistics/_SUCCESS",
            "xai/_SUCCESS",
        ):
            path = root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok\n", encoding="utf-8")
        proof = root / "statistics" / "c6-proof.txt"
        proof.write_text("proof\n", encoding="utf-8")
        atomic_write_json(root / "statistics" / "artifact_manifest.json", {
            "status": "COMPLETE",
            "protocol_hash": protocol_hash_value,
            "artifacts": {
                "statistics/c6-proof.txt": {
                    "sha256": file_sha256(proof),
                    "bytes": proof.stat().st_size,
                }
            },
        })
        atomic_write_json(root / "model_lock.json", {
            "status": "LOCKED",
            "protocol_hash": protocol_hash_value,
            "selected_backbone": "resnet50",
            "selected_pretraining": "imagenet",
            "proposal_amendment_required": False,
        })

        scenarios = {}
        for scenario in ("S1", "S2", "S3"):
            run_dir = root / "deployment" / "runs" / scenario
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "best.pt"
            checkpoint.write_bytes(f"checkpoint-{scenario}".encode("ascii"))
            entry = {
                "status": "COMPLETE",
                "scenario": scenario,
                "checkpoint_path": str(checkpoint.relative_to(root)).replace("\\", "/"),
                "checkpoint_sha256": file_sha256(checkpoint),
                "scaler_path": None,
                "scaler_sha256": None,
            }
            if scenario in {"S1", "S3"}:
                scaler = run_dir / "scaler.pkl"
                scaler.write_bytes(f"scaler-{scenario}".encode("ascii"))
                entry["scaler_path"] = str(scaler.relative_to(root)).replace("\\", "/")
                entry["scaler_sha256"] = file_sha256(scaler)
            scenarios[scenario] = entry
        index = {
            "status": "READY_FOR_SECONDARY_HOLDOUT",
            "protocol_hash": protocol_hash_value,
            "model_lock_hash": file_sha256(root / "model_lock.json"),
            "deployment_split_hash": file_sha256(deployment),
            "scenarios": scenarios,
        }
        atomic_write_json(root / "deployment" / "refit_index.json", index)
        (root / "deployment" / "_REFIT_SUCCESS").write_text("ok\n", encoding="utf-8")

    def test_c7_preflight_requires_complete_dag_and_all_refits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            state = validate_c7_prerequisites(root, require_refit=True)
            self.assertEqual(set(state["refit_index"]["scenarios"]), {"S1", "S2", "S3"})
            (root / "statistics" / "_SUCCESS").unlink()
            with self.assertRaises(Exception):
                validate_c7_prerequisites(root, require_refit=True)

    def test_official_test_requires_claimed_matching_access_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            with self.assertRaises(OfficialTestAccessError):
                assert_official_test_access(
                    stage="C7", protocol_path=root / "protocol.json"
                )
            receipt = claim_official_test_access(
                root,
                confirmation=OFFICIAL_TEST_CONFIRMATION,
                implementation_commit="commit-a",
                environment_hash_value="environment-a",
            )
            assert_official_test_access(
                stage="C7",
                protocol_path=root / "protocol.json",
                access_event_id=receipt["access_event_id"],
            )
            with self.assertRaises(OfficialTestAccessError):
                assert_official_test_access(
                    stage="C7",
                    protocol_path=root / "protocol.json",
                    access_event_id="wrong-event",
                )

    def test_access_claim_is_resumable_but_new_access_after_success_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            first = claim_official_test_access(
                root,
                confirmation=OFFICIAL_TEST_CONFIRMATION,
                implementation_commit="commit-a",
                environment_hash_value="environment-a",
            )
            resumed = claim_official_test_access(
                root,
                confirmation=OFFICIAL_TEST_CONFIRMATION,
                implementation_commit="commit-a",
                environment_hash_value="environment-a",
            )
            self.assertEqual(first["access_event_id"], resumed["access_event_id"])
            with self.assertRaises(OfficialTestAccessError):
                claim_official_test_access(
                    root,
                    confirmation=OFFICIAL_TEST_CONFIRMATION,
                    implementation_commit="commit-b",
                    environment_hash_value="environment-a",
                )
            (root / "secondary_holdout" / "_SUCCESS").write_text("done\n", encoding="utf-8")
            with self.assertRaises(OfficialTestAccessError):
                claim_official_test_access(
                    root,
                    confirmation=OFFICIAL_TEST_CONFIRMATION,
                    implementation_commit="commit-c",
                    environment_hash_value="environment-a",
                )

    def test_wrong_confirmation_cannot_create_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            with self.assertRaises(OfficialTestAccessError):
                claim_official_test_access(
                    root,
                    confirmation="yes",
                    implementation_commit="commit-a",
                    environment_hash_value="environment-a",
                )
            self.assertFalse(
                (root / "secondary_holdout" / "official_test_access_receipt.json").exists()
            )

    def test_secondary_manifest_excludes_process_owned_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "secondary_holdout"
            evidence = output / "S1" / "predictions.csv"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text("image_index,probability\ncase.png,0.5\n", encoding="utf-8")
            (output / "c7_holdout_stdout.log").write_text("still writable\n", encoding="utf-8")
            (output / "c7_holdout_stderr.log").write_text("still writable\n", encoding="utf-8")
            atomic_write_json(output / "artifact_manifest.json", {"status": "old"})
            (output / "_SUCCESS").write_text("ok\n", encoding="utf-8")

            artifacts = _secondary_holdout_artifacts(root, output)

            self.assertIn("secondary_holdout/S1/predictions.csv", artifacts)
            self.assertNotIn("secondary_holdout/c7_holdout_stdout.log", artifacts)
            self.assertNotIn("secondary_holdout/c7_holdout_stderr.log", artifacts)
            self.assertNotIn("secondary_holdout/artifact_manifest.json", artifacts)
            self.assertNotIn("secondary_holdout/_SUCCESS", artifacts)


if __name__ == "__main__":
    unittest.main()

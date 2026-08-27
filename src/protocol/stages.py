"""Frozen-protocol validation and executable stage gates."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from configs.config import cfg
from src.protocol.contracts import (
    assert_runtime_matches,
    file_sha256,
    protocol_hash,
    read_json,
)
from src.protocol.manifests import validate_manifest_hash


class StageGateError(RuntimeError):
    pass


def load_frozen_protocol(protocol_dir: Path) -> Dict[str, Any]:
    """Validate scientific hash, runtime contract, and immutable manifests."""
    protocol_dir = Path(protocol_dir)
    protocol = read_json(protocol_dir / "protocol.json")
    if protocol.get("status") != "FROZEN":
        raise StageGateError("Canonical execution requires a FROZEN protocol")
    scientific = protocol.get("scientific_spec")
    if not isinstance(scientific, dict):
        raise StageGateError("Frozen protocol has no scientific_spec")
    calculated = protocol_hash(scientific)
    if calculated != protocol.get("protocol_hash"):
        raise StageGateError(
            "Scientific protocol hash mismatch; the frozen specification changed"
        )
    assert_runtime_matches(
        scientific["runtime_contract"], cfg.scientific_runtime_values()
    )
    validate_manifest_hash(
        protocol_dir / "folds.csv",
        scientific["splits"]["primary_cv"]["fold_manifest_hash"],
    )
    validate_manifest_hash(
        protocol_dir / "deployment_split.csv",
        scientific["splits"]["deployment"]["deployment_split_hash"],
    )
    return protocol


def load_model_lock(protocol_dir: Path) -> Dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    protocol = load_frozen_protocol(protocol_dir)
    path = protocol_dir / "model_lock.json"
    if not path.exists():
        raise StageGateError("C4/C5 are blocked until C3 creates model_lock.json")
    lock = read_json(path)
    if lock.get("status") != "LOCKED":
        raise StageGateError("model_lock.json is not in LOCKED status")
    if lock.get("protocol_hash") != protocol["protocol_hash"]:
        raise StageGateError("Model lock belongs to a different protocol")
    if lock.get("selected_backbone") not in cfg.model.image_candidates:
        raise StageGateError("Model lock contains an unsupported backbone")
    if lock.get("selected_pretraining") not in {"imagenet", "chexnet"}:
        raise StageGateError("Model lock contains unsupported pretraining")
    return lock


def _artifact_path(protocol_dir: Path, value: str) -> Path:
    """Resolve a recorded artifact path without allowing escape from the protocol."""
    root = Path(protocol_dir).resolve()
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StageGateError(f"Artifact path escapes canonical protocol directory: {value}") from exc
    return resolved


def _validate_checksum_manifest(protocol_dir: Path, manifest_path: Path) -> None:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise StageGateError(f"Artifact manifest is not COMPLETE: {manifest_path}")
    protocol = read_json(Path(protocol_dir) / "protocol.json")
    if manifest.get("protocol_hash") != protocol.get("protocol_hash"):
        raise StageGateError(f"Artifact manifest belongs to another protocol: {manifest_path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise StageGateError(f"Artifact manifest has no checksummed artifacts: {manifest_path}")
    for relative, expected in artifacts.items():
        artifact = _artifact_path(protocol_dir, str(relative))
        if not artifact.is_file():
            raise StageGateError(f"Checksummed artifact is missing: {relative}")
        if file_sha256(artifact) != expected.get("sha256"):
            raise StageGateError(f"Artifact checksum mismatch: {relative}")
        if artifact.stat().st_size != int(expected.get("bytes", -1)):
            raise StageGateError(f"Artifact byte-size mismatch: {relative}")


def _require_proposal_amendment(protocol_dir: Path, lock: Dict[str, Any]) -> None:
    if not lock.get("proposal_amendment_required", False):
        return
    amendment_path = Path(protocol_dir) / "proposal_amendment.json"
    if not amendment_path.exists():
        raise StageGateError(
            "C4 is blocked pending an approved proposal_amendment.json"
        )
    amendment = read_json(amendment_path)
    if amendment.get("status") != "APPROVED":
        raise StageGateError("Proposal amendment exists but is not APPROVED")
    if amendment.get("protocol_hash") != lock.get("protocol_hash"):
        raise StageGateError("Proposal amendment belongs to a different protocol")
    if amendment.get("model_lock_hash") != file_sha256(
        Path(protocol_dir) / "model_lock.json"
    ):
        raise StageGateError("Proposal amendment does not approve the current model lock")


def validate_c7_prerequisites(
    protocol_dir: Path,
    *,
    require_refit: bool = False,
) -> Dict[str, Any]:
    """Validate the immutable C1-C6 DAG and, optionally, all deployment refits."""
    root = Path(protocol_dir)
    protocol = load_frozen_protocol(root)
    lock = load_model_lock(root)
    _require_proposal_amendment(root, lock)
    required_markers = (
        root / "screening" / "tabular" / "_SUCCESS",
        root / "screening" / "image" / "_SUCCESS",
        root / "main" / "_SUCCESS",
        root / "ablation" / "_SUCCESS",
        root / "statistics" / "_SUCCESS",
        root / "xai" / "_SUCCESS",
    )
    missing = [str(path.relative_to(root)) for path in required_markers if not path.is_file()]
    if missing:
        raise StageGateError(f"C7 is blocked; incomplete prerequisite markers: {missing}")
    c6_manifest = root / "statistics" / "artifact_manifest.json"
    if not c6_manifest.is_file():
        raise StageGateError("C7 requires the checksummed C6 artifact manifest")
    _validate_checksum_manifest(root, c6_manifest)

    result: Dict[str, Any] = {"protocol": protocol, "model_lock": lock}
    if not require_refit:
        return result

    marker = root / "deployment" / "_REFIT_SUCCESS"
    index_path = root / "deployment" / "refit_index.json"
    if not marker.is_file() or not index_path.is_file():
        raise StageGateError("Official test is blocked until all C7 deployment refits finish")
    index = read_json(index_path)
    if index.get("status") != "READY_FOR_SECONDARY_HOLDOUT":
        raise StageGateError("Deployment refit index is not ready for secondary holdout")
    if index.get("protocol_hash") != protocol["protocol_hash"]:
        raise StageGateError("Deployment refit index belongs to another protocol")
    if index.get("model_lock_hash") != file_sha256(root / "model_lock.json"):
        raise StageGateError("Deployment refit index does not match the current model lock")
    if index.get("deployment_split_hash") != file_sha256(root / "deployment_split.csv"):
        raise StageGateError("Deployment refit index does not match the immutable split")
    scenarios = index.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != {"S1", "S2", "S3"}:
        raise StageGateError("Deployment refit index must contain exactly S1, S2, and S3")
    for scenario, entry in scenarios.items():
        if entry.get("scenario") != scenario or entry.get("status") != "COMPLETE":
            raise StageGateError(f"Invalid deployment refit entry for {scenario}")
        checkpoint = _artifact_path(root, str(entry.get("checkpoint_path", "")))
        if not checkpoint.is_file():
            raise StageGateError(f"Missing deployment checkpoint for {scenario}: {checkpoint}")
        if file_sha256(checkpoint) != entry.get("checkpoint_sha256"):
            raise StageGateError(f"Deployment checkpoint checksum mismatch for {scenario}")
        scaler_value: Optional[str] = entry.get("scaler_path")
        if scenario in {"S1", "S3"}:
            if not scaler_value:
                raise StageGateError(f"Missing deployment scaler entry for {scenario}")
            scaler = _artifact_path(root, scaler_value)
            if not scaler.is_file() or file_sha256(scaler) != entry.get("scaler_sha256"):
                raise StageGateError(f"Deployment scaler checksum mismatch for {scenario}")
    result["refit_index"] = index
    result["refit_index_path"] = index_path
    return result


def validate_cv_request(
    *,
    stage: str,
    scenario: str,
    backbone: str,
    pretraining: str,
    feature_set: str,
    protocol_dir: Path,
) -> None:
    """Reject valid-looking runs that violate the frozen C1-C5 DAG."""
    load_frozen_protocol(protocol_dir)
    if stage == "C1":
        if (scenario, backbone, pretraining, feature_set) != (
            "S1", "canonical_mlp", "not_applicable", "D"
        ):
            raise StageGateError("C1 MLP benchmark contract mismatch")
        return
    if stage == "C2":
        if scenario != "S2" or feature_set != "D":
            raise StageGateError("C2 only allows image-only S2 with feature set D")
        if backbone not in cfg.model.image_candidates:
            raise StageGateError("C2 backbone is outside the frozen candidate set")
        if pretraining == "imagenet":
            return
        if pretraining == "chexnet" and backbone == "densenet121":
            candidate_path = Path(protocol_dir) / "screening" / "image" / "model_lock_candidate.json"
            if not candidate_path.exists():
                raise StageGateError(
                    "CheXNet C2 comparison is conditional on a DenseNet model-lock candidate"
                )
            candidate = read_json(candidate_path)
            if candidate.get("selected_backbone") != "densenet121":
                raise StageGateError("CheXNet comparison is not authorized for this candidate")
            from src.protocol.chexnet import require_approved_chexnet_audit
            require_approved_chexnet_audit(protocol_dir)
            return
        raise StageGateError("C2 pretraining request violates the frozen policy")
    if stage in {"C4", "C5"}:
        lock = load_model_lock(protocol_dir)
        _require_proposal_amendment(protocol_dir, lock)
        if stage == "C4" and feature_set != "D":
            raise StageGateError("C4 main scenarios require feature set D")
        if stage == "C5" and scenario not in {"S1", "S3"}:
            raise StageGateError("C5 metadata ablation only allows S1 or S3")
        if stage == "C5" and feature_set == "D":
            raise StageGateError("C5-D reuses the corresponding C4 main artifact")
        if scenario == "S1":
            if backbone != "canonical_mlp" or pretraining != "not_applicable":
                raise StageGateError("S1 must use the canonical metadata MLP")
        elif scenario in {"S2", "S3"}:
            if backbone != lock["selected_backbone"] or pretraining != lock["selected_pretraining"]:
                raise StageGateError("Run does not match immutable model_lock.json")
        else:
            raise StageGateError("Only S1/S2/S3 are canonical")
        return
    raise StageGateError(f"Unsupported canonical stage: {stage}")


def oof_path_for(
    protocol_dir: Path,
    *,
    stage: str,
    scenario: str,
    model: str,
    pretraining: str,
    feature_set: str,
) -> Path:
    protocol_dir = Path(protocol_dir)
    if stage == "C1":
        return protocol_dir / "screening" / "tabular" / f"{model}-{feature_set}-oof.csv"
    if stage == "C2":
        return protocol_dir / "screening" / "image" / f"{model}-{pretraining}-oof.csv"
    if stage == "C4":
        return protocol_dir / "main" / f"{scenario}-{model}-{pretraining}-{feature_set}-oof.csv"
    if stage == "C5":
        return protocol_dir / "ablation" / f"{scenario}-{model}-{pretraining}-{feature_set}-oof.csv"
    raise StageGateError(f"No OOF artifact policy for stage {stage}")


def finalize_cv_stage_if_complete(
    protocol_dir: Path,
    *,
    stage: str,
    backbone: str,
    pretraining: str,
) -> bool:
    """Create a stage marker only after every canonical OOF artifact exists."""
    root = Path(protocol_dir)
    if stage == "C4":
        expected = (
            oof_path_for(
                root,
                stage="C4",
                scenario="S1",
                model="canonical_mlp",
                pretraining="not_applicable",
                feature_set="D",
            ),
            oof_path_for(
                root,
                stage="C4",
                scenario="S2",
                model=backbone,
                pretraining=pretraining,
                feature_set="D",
            ),
            oof_path_for(
                root,
                stage="C4",
                scenario="S3",
                model=backbone,
                pretraining=pretraining,
                feature_set="D",
            ),
        )
        marker = root / "main" / "_SUCCESS"
    elif stage == "C5":
        expected = tuple(
            oof_path_for(
                root,
                stage="C5",
                scenario=scenario,
                model="canonical_mlp" if scenario == "S1" else backbone,
                pretraining="not_applicable" if scenario == "S1" else pretraining,
                feature_set=feature_set,
            )
            for scenario in ("S1", "S3")
            for feature_set in ("A", "B", "C")
        )
        marker = root / "ablation" / "_SUCCESS"
    else:
        raise StageGateError(f"No completion policy for stage {stage}")

    if not all(path.is_file() for path in expected):
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{stage} canonical artifacts complete\n", encoding="utf-8")
    return True


def stage_status(protocol_dir: Path) -> Dict[str, Any]:
    protocol = load_frozen_protocol(protocol_dir)
    root = Path(protocol_dir)
    image_dir = root / "screening" / "image"
    tabular_dir = root / "screening" / "tabular"
    chexnet_audit = image_dir / "chexnet_provenance_audit.json"
    return {
        "protocol_hash": protocol["protocol_hash"],
        "status": protocol["status"],
        "C1_tabular_complete": (tabular_dir / "_SUCCESS").exists(),
        "C2_imagenet_complete": all(
            (image_dir / f"{name}-imagenet-oof.csv").exists()
            for name in cfg.model.image_candidates
        ),
        "C2_environment_locked": (image_dir / "environment_lock.json").exists(),
        "CheXNet_provenance_status": (
            read_json(chexnet_audit).get("status") if chexnet_audit.exists() else "NOT_AUDITED"
        ),
        "C3_model_locked": (root / "model_lock.json").exists(),
        "C4_main_complete": (root / "main" / "_SUCCESS").exists(),
        "C5_ablation_complete": (root / "ablation" / "_SUCCESS").exists(),
        "C6_xai_statistics_complete": (root / "statistics" / "_SUCCESS").exists(),
        "C7_deployment_refit_complete": (root / "deployment" / "_REFIT_SUCCESS").exists(),
        "C7_official_test_access_claimed": (
            root / "secondary_holdout" / "official_test_access_receipt.json"
        ).exists(),
        "C7_secondary_holdout_complete": (root / "secondary_holdout" / "_SUCCESS").exists(),
    }

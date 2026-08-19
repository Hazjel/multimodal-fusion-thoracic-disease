"""Executable provenance audit for conditional CheXNet initialization."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from configs.config import cfg
from src.models.architectures import ImageEncoder
from src.protocol.contracts import (
    atomic_write_json,
    file_sha256,
    git_commit,
    read_json,
)
from src.protocol.stages import StageGateError, load_frozen_protocol, oof_path_for


AUDIT_FILENAME = "chexnet_provenance_audit.json"
SAFE_OFFICIAL_TEST_VALUE = "NOT_USED_FOR_TRAINING_OR_MODEL_SELECTION"


def chexnet_audit_path(protocol_dir: Path) -> Path:
    return Path(protocol_dir) / "screening" / "image" / AUDIT_FILENAME


def evaluate_provenance_declaration(
    declaration: Optional[Mapping[str, Any]],
) -> Tuple[str, List[str]]:
    """Return APPROVED only when every frozen provenance question is answered."""
    if not declaration:
        return "EXCLUDED", ["No provenance declaration was supplied"]
    reasons: List[str] = []
    source_url = str(declaration.get("source_url", "")).strip()
    if not source_url.startswith(("https://", "http://")):
        reasons.append("source_url must identify a reviewable HTTP(S) source")
    source_commit = str(declaration.get("source_commit", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        reasons.append("source_commit must be an exact 40-character Git SHA")
    if str(declaration.get("training_dataset", "")).strip() != "NIH ChestX-ray14":
        reasons.append("training_dataset must be explicitly identified as NIH ChestX-ray14")
    if len(str(declaration.get("training_split_provenance", "")).strip()) < 20:
        reasons.append("training_split_provenance is missing or insufficient")
    if not declaration.get("preprocessing"):
        reasons.append("source preprocessing is undocumented")
    labels = declaration.get("label_mapping")
    if not isinstance(labels, list) or set(map(str, labels)) != set(cfg.data.label_names):
        reasons.append("label_mapping must enumerate the 14 canonical NIH pathologies")
    if declaration.get("official_nih_test_usage") != SAFE_OFFICIAL_TEST_VALUE:
        reasons.append("official NIH test non-use is not established")
    evidence = declaration.get("evidence_urls")
    if not isinstance(evidence, list) or not evidence or not all(
        str(item).startswith(("https://", "http://")) for item in evidence
    ):
        reasons.append("at least one reviewable evidence URL is required")
    if not str(declaration.get("reviewed_by", "")).strip():
        reasons.append("reviewed_by is required")
    return ("APPROVED" if not reasons else "EXCLUDED"), reasons


def _technical_checkpoint_audit() -> Tuple[Dict[str, Any], List[str]]:
    path = Path(cfg.model.chexnet_weights_path)
    technical: Dict[str, Any] = {
        "checkpoint_filename": path.name,
        "configured_path": str(path),
        "expected_sha256": cfg.model.chexnet_expected_sha256,
    }
    reasons: List[str] = []
    if not path.is_file():
        technical["exists"] = False
        return technical, ["configured CheXNet checkpoint does not exist"]
    technical["exists"] = True
    technical["size_bytes"] = path.stat().st_size
    actual = file_sha256(path)
    technical["checkpoint_sha256"] = actual
    technical["checksum_matches"] = actual == cfg.model.chexnet_expected_sha256
    if not technical["checksum_matches"]:
        reasons.append("checkpoint SHA-256 does not match the frozen identifier")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        technical["top_level_keys"] = sorted(raw) if isinstance(raw, dict) else []
        technical["declared_arch"] = raw.get("arch") if isinstance(raw, dict) else None
        technical["source_epoch"] = raw.get("epoch") if isinstance(raw, dict) else None
        encoder = ImageEncoder(
            "densenet121", pretraining="chexnet", chexnet_path=path, fold=0
        )
        technical["feature_mapping_valid"] = True
        technical["feature_state_checksum"] = encoder.pretrained_state_checksum
    except Exception as error:  # technical exclusion is recorded, never silently bypassed
        technical["feature_mapping_valid"] = False
        technical["mapping_error"] = f"{type(error).__name__}: {error}"
        reasons.append("checkpoint cannot be mapped to the canonical DenseNet feature encoder")
    return technical, reasons


def write_chexnet_provenance_audit(
    *,
    protocol_dir: Path,
    declaration_path: Optional[Path] = None,
) -> Dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    protocol = load_frozen_protocol(protocol_dir)
    candidate_path = protocol_dir / "screening" / "image" / "model_lock_candidate.json"
    if not candidate_path.exists():
        raise StageGateError("CheXNet audit is conditional on a C3 DenseNet candidate")
    candidate = read_json(candidate_path)
    if candidate.get("selected_backbone") != "densenet121":
        raise StageGateError("CheXNet audit is outside scope because DenseNet was not selected")
    if (protocol_dir / "model_lock.json").exists():
        raise StageGateError("CheXNet provenance cannot change after model lock")
    chexnet_oof = oof_path_for(
        protocol_dir,
        stage="C2",
        scenario="S2",
        model="densenet121",
        pretraining="chexnet",
        feature_set="D",
    )
    if chexnet_oof.exists():
        raise StageGateError("CheXNet provenance cannot change after canonical predictions exist")

    declaration = read_json(declaration_path) if declaration_path is not None else None
    declaration_status, declaration_reasons = evaluate_provenance_declaration(declaration)
    technical, technical_reasons = _technical_checkpoint_audit()
    reasons = technical_reasons + declaration_reasons
    status = "APPROVED" if not reasons and declaration_status == "APPROVED" else "EXCLUDED"
    audit = {
        "schema_version": 1,
        "status": status,
        "protocol_hash": protocol["protocol_hash"],
        "checkpoint": technical,
        "provenance_declaration": declaration,
        "exclusion_reasons": reasons,
        "implementation_commit": git_commit(cfg.paths.project_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(chexnet_audit_path(protocol_dir), audit)
    return audit


def require_approved_chexnet_audit(protocol_dir: Path) -> Dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    path = chexnet_audit_path(protocol_dir)
    if not path.exists():
        raise StageGateError("CheXNet is blocked until chexnet_provenance_audit.json exists")
    audit = read_json(path)
    protocol = load_frozen_protocol(protocol_dir)
    declaration_status, reasons = evaluate_provenance_declaration(
        audit.get("provenance_declaration")
    )
    checkpoint_path = Path(cfg.model.chexnet_weights_path)
    if not checkpoint_path.is_file():
        raise StageGateError("Approved CheXNet checkpoint is no longer available")
    actual_checksum = file_sha256(checkpoint_path)
    recorded = audit.get("checkpoint", {})
    valid = all([
        audit.get("status") == "APPROVED",
        audit.get("protocol_hash") == protocol["protocol_hash"],
        declaration_status == "APPROVED",
        not reasons,
        recorded.get("checkpoint_sha256") == actual_checksum,
        actual_checksum == cfg.model.chexnet_expected_sha256,
        recorded.get("feature_mapping_valid") is True,
    ])
    if not valid:
        raise StageGateError(
            "CheXNet provenance is not APPROVED for this protocol/checkpoint"
        )
    return audit

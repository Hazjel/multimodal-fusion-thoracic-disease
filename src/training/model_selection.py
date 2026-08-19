"""C3 model-lock creation from completed frozen C2 evidence."""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import roc_auc_score

from configs.config import cfg
from src.evaluation.stats import (
    paired_patient_cluster_bootstrap,
    select_cnn_candidate,
)
from src.models.architectures import build_model
from src.protocol.contracts import atomic_write_json, git_commit, read_json
from src.protocol.stages import load_frozen_protocol, oof_path_for


def _oof_path(protocol_dir: Path, backbone: str, pretraining: str) -> Path:
    return oof_path_for(
        protocol_dir,
        stage="C2",
        scenario="S2",
        model=backbone,
        pretraining=pretraining,
        feature_set="D",
    )


def _fold_aucs(frame: pd.DataFrame) -> List[float]:
    return [
        float(roc_auc_score(part["true_label"], part["probability"]))
        for _, part in frame.groupby("fold", sort=True)
    ]


def _training_seconds(protocol_dir: Path, frame: pd.DataFrame) -> List[float]:
    values = []
    for run_id in sorted(frame["run_id"].unique()):
        history_path = protocol_dir / "runs" / str(run_id) / "history.json"
        if not history_path.exists():
            raise RuntimeError(f"Missing training timing artifact: {history_path}")
        timing = read_json(history_path).get("timing_seconds")
        if not timing:
            raise RuntimeError(f"Missing timing_seconds in {history_path}")
        values.append(float(sum(timing)))
    return values


def _trainable_parameter_count(backbone: str, pretraining: str = "imagenet") -> int:
    model = build_model(
        "S2", backbone_name=backbone, pretraining=pretraining, fold=0
    )
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _write_immutable(path: Path, payload: Dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise FileExistsError(f"Immutable stage artifact already exists: {path}")
        return
    atomic_write_json(path, payload)


def create_model_lock(protocol_dir: Path) -> Dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    protocol = load_frozen_protocol(protocol_dir)
    model_lock_path = protocol_dir / "model_lock.json"
    if model_lock_path.exists():
        return read_json(model_lock_path)

    predictions: Dict[str, pd.DataFrame] = {}
    fold_aucs: Dict[str, List[float]] = {}
    trainable_parameters: Dict[str, int] = {}
    median_training_seconds: Dict[str, float] = {}
    for backbone in cfg.model.image_candidates:
        path = _oof_path(protocol_dir, backbone, "imagenet")
        if not path.exists():
            raise RuntimeError(
                f"C3 requires completed ImageNet OOF evidence for {backbone}: {path}"
            )
        frame = pd.read_csv(path)
        predictions[backbone] = frame
        fold_aucs[backbone] = _fold_aucs(frame)
        trainable_parameters[backbone] = _trainable_parameter_count(backbone)
        median_training_seconds[backbone] = float(
            statistics.median(_training_seconds(protocol_dir, frame))
        )

    architecture_selection = select_cnn_candidate(
        predictions,
        fold_aucs,
        trainable_parameters,
        median_training_seconds,
        n_boot=cfg.evaluation.bootstrap_replicates,
        seed=cfg.train.seed,
    )
    backbone = architecture_selection["selected"]
    image_screening_dir = protocol_dir / "screening" / "image"
    image_screening_dir.mkdir(parents=True, exist_ok=True)
    if backbone == "densenet121":
        chexnet_path = _oof_path(protocol_dir, backbone, "chexnet")
        if not chexnet_path.exists():
            candidate = {
                "status": "CHEXNET_AUDIT_REQUIRED",
                "protocol_hash": protocol["protocol_hash"],
                "selected_backbone": backbone,
                "architecture_evidence": architecture_selection,
                "implementation_commit": git_commit(cfg.paths.project_root),
            }
            atomic_write_json(image_screening_dir / "model_lock_candidate.json", candidate)
            return candidate

        imagenet = predictions[backbone][
            ["image_index", "patient_id", "true_label", "probability"]
        ].rename(columns={"probability": "probability_imagenet"})
        chexnet_frame = pd.read_csv(chexnet_path)
        chexnet = chexnet_frame[
            ["image_index", "patient_id", "true_label", "probability"]
        ].rename(columns={"probability": "probability_chexnet"})
        merged = imagenet.merge(
            chexnet,
            on=["image_index", "patient_id", "true_label"],
            validate="one_to_one",
        )
        if len(merged) != len(imagenet) or len(merged) != len(chexnet):
            raise RuntimeError("DenseNet ImageNet/CheXNet OOF predictions are not aligned")
        comparison = paired_patient_cluster_bootstrap(
            merged,
            probability_a="probability_imagenet",
            probability_b="probability_chexnet",
            n_boot=cfg.evaluation.bootstrap_replicates,
            seed=cfg.train.seed,
        )
        clearly_separated = not (
            comparison["ci_low"] <= 0.0 <= comparison["ci_high"]
        )
        if clearly_separated:
            pretraining = "imagenet" if comparison["delta_auc"] > 0 else "chexnet"
        else:
            pretraining = "imagenet"
        pretraining_evidence: Dict[str, Any] = {
            "comparison": comparison,
            "tie_policy": "ImageNet when the OOF evidence does not clearly separate initializations",
            "selected": pretraining,
            "chexnet_expected_sha256": cfg.model.chexnet_expected_sha256,
        }
    else:
        pretraining = "imagenet"
        pretraining_evidence = {"selected": "imagenet", "chexnet_comparison": "not_applicable"}

    lock = {
        "status": "LOCKED",
        "protocol_hash": protocol["protocol_hash"],
        "selected_backbone": backbone,
        "selected_pretraining": pretraining,
        "architecture_evidence": architecture_selection,
        "pretraining_evidence": pretraining_evidence,
        "proposal_amendment_required": (backbone, pretraining)
        != ("densenet121", "chexnet"),
        "selection_rule": "frozen_protocol_v1.0.0",
        "implementation_commit": git_commit(cfg.paths.project_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_immutable(model_lock_path, lock)
    return lock

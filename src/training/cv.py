"""Manifest-driven canonical patient-level cross-validation runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from configs.config import cfg
from src.data.dataset import (
    TABULAR_FEATURE_SETS,
    build_image_index,
    load_and_prepare_metadata,
    load_official_training_pool,
    make_fold_dataloaders,
)
from src.evaluation import (
    collect_prediction_frame,
    compute_metrics,
    validate_oof_coverage,
    write_prediction_frame,
)
from src.models.architectures import build_model, image_initial_hashes
from src.protocol.contracts import (
    atomic_write_json,
    git_commit,
    read_json,
    semantic_config_hash,
)
from src.protocol.environment import collect_environment, environment_hash
from src.protocol.registry import upsert_registry
from src.protocol.stages import (
    load_frozen_protocol,
    oof_path_for,
    validate_cv_request,
)
from src.training import save_scaler, train


def _modalities(scenario: str):
    return {
        "S1": ("tabular",),
        "S2": ("image",),
        "S3": ("image", "tabular"),
    }[scenario]


def run_cross_validation(
    scenario: str,
    *,
    stage: str,
    protocol_dir: Path,
    backbone_name: str = "densenet121",
    pretraining: str = "imagenet",
    feature_set: str = "D",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if scenario not in {"S1", "S2", "S3"}:
        raise ValueError("Canonical CV only allows S1, S2, or S3")
    if feature_set not in TABULAR_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    protocol_dir = Path(protocol_dir)
    validate_cv_request(
        stage=stage,
        scenario=scenario,
        backbone=backbone_name,
        pretraining=pretraining,
        feature_set=feature_set,
        protocol_dir=protocol_dir,
    )
    protocol = load_frozen_protocol(protocol_dir)
    protocol_hash_value = protocol["protocol_hash"]
    implementation = git_commit(cfg.paths.project_root)
    environment = collect_environment(implementation)
    environment_hash_value = environment_hash(environment)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_index = build_image_index(cfg.paths.image_dirs)
    metadata = load_and_prepare_metadata(cfg.paths.csv_path, image_index)
    training_pool = load_official_training_pool(
        metadata, cfg.paths.train_list_path
    )
    manifest = pd.read_csv(protocol_dir / "folds.csv")
    manifest_lookup = manifest.set_index("image_index")["fold"]
    training_pool = training_pool.copy()
    training_pool["fold"] = training_pool["Image Index"].map(manifest_lookup)
    if training_pool["fold"].isna().any():
        raise RuntimeError("Training pool contains images absent from immutable folds.csv")

    fold_predictions: List[pd.DataFrame] = []
    fold_metrics = []
    for fold in range(cfg.data.cv_splits):
        train_frame = training_pool[training_pool["fold"] != fold].reset_index(drop=True)
        validation_frame = training_pool[training_pool["fold"] == fold].reset_index(drop=True)
        train_loader, validation_loader, scaler, pos_weight = make_fold_dataloaders(
            train_frame,
            validation_frame,
            image_index,
            modalities=_modalities(scenario),
            feature_set=feature_set,
            seed=cfg.train.seed + fold,
        )
        input_dim = len(TABULAR_FEATURE_SETS[feature_set])
        model = build_model(
            scenario,
            backbone_name=backbone_name,
            pretraining=pretraining,
            fold=fold,
            tabular_input_dim=input_dim,
        ).to(device)
        branch = None
        if scenario == "S2":
            branch = model.branch
        elif scenario == "S3":
            branch = model.image_branch
        weight_checksum = branch.pretrained_state_checksum if branch is not None else "not_applicable"
        resolved = {
            "scenario": scenario,
            "backbone": backbone_name if scenario != "S1" else "not_applicable",
            "pretraining": pretraining if scenario != "S1" else "not_applicable",
            "feature_set": feature_set,
            "fold": fold,
            "runtime": cfg.scientific_runtime_values(),
        }
        semantic_hash = semantic_config_hash(
            protocol_hash_value=protocol_hash_value,
            selected_architecture=backbone_name if scenario != "S1" else "canonical_mlp",
            weight_checksum=weight_checksum,
            fold=fold,
            feature_set=feature_set,
            resolved_runtime_config=resolved,
            environment_hash=environment_hash_value,
            implementation_commit=implementation,
        )
        run_id = f"{stage}-{scenario}-{backbone_name}-{pretraining}-{feature_set}-fold{fold}-{semantic_hash[:12]}"
        run_dir = protocol_dir / "runs" / run_id
        success = run_dir / "_SUCCESS"
        prediction_path = run_dir / "predictions.csv"
        metrics_path = run_dir / "metrics.json"
        if success.exists() and prediction_path.exists() and metrics_path.exists():
            fold_predictions.append(pd.read_csv(prediction_path))
            fold_metrics.append(read_json(metrics_path))
            continue
        initial_hashes = image_initial_hashes(model) if scenario in {"S2", "S3"} else None
        metadata_run = {
            "run_id": run_id,
            "protocol_hash": protocol_hash_value,
            "semantic_config_hash": semantic_hash,
            "implementation_commit": implementation,
            "environment_hash": environment_hash_value,
            "stage": stage,
            "scenario": scenario,
            "fold": fold,
            "feature_set": feature_set,
            "backbone": backbone_name,
            "pretraining": pretraining,
            "initial_image_hashes": initial_hashes,
        }
        model = train(
            model,
            train_loader,
            validation_loader,
            pos_weight,
            scenario,
            device,
            run_dir=run_dir,
            run_metadata=metadata_run,
            resume=True,
        )
        if scaler is not None:
            save_scaler(scaler, run_dir / "scaler.pkl")
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        prediction_frame = collect_prediction_frame(
            model,
            validation_loader,
            device,
            fold=fold,
            split="validation",
            model_name=backbone_name if scenario != "S1" else "canonical_mlp",
            scenario=scenario,
            feature_set=feature_set,
            pretraining=pretraining if scenario != "S1" else "not_applicable",
            checkpoint_epoch=int(best["best_epoch"]),
            protocol_hash=protocol_hash_value,
            semantic_config_hash=semantic_hash,
            run_id=run_id,
        )
        write_prediction_frame(prediction_frame, prediction_path)
        metrics = compute_metrics(
            prediction_frame["probability"].to_numpy(),
            prediction_frame["true_label"].to_numpy(),
        )
        atomic_write_json(metrics_path, metrics)
        upsert_registry(protocol_dir / "experiment_registry.csv", {
            "run_id": run_id,
            "phase": stage,
            "scenario": scenario,
            "model": backbone_name if scenario != "S1" else "canonical_mlp",
            "fold": fold,
            "feature_set": feature_set,
            "pretraining": pretraining,
            "protocol_hash": protocol_hash_value,
            "semantic_config_hash": semantic_hash,
            "implementation_commit": implementation,
            "environment_hash": environment_hash_value,
            "pos_weight": float(pos_weight.item()),
            "best_epoch": int(best["best_epoch"]),
            "best_validation_auc": float(best["best_validation_auc"]),
            "status": "done",
            "artifact_path": str(run_dir),
        })
        success.write_text("ok\n", encoding="utf-8")
        fold_predictions.append(prediction_frame)
        fold_metrics.append(metrics)

    oof = pd.concat(fold_predictions, ignore_index=True)
    validate_oof_coverage(oof, manifest["image_index"])
    model_name = backbone_name if scenario != "S1" else "canonical_mlp"
    oof_path = oof_path_for(
        protocol_dir,
        stage=stage,
        scenario=scenario,
        model=model_name,
        pretraining=pretraining if scenario != "S1" else "not_applicable",
        feature_set=feature_set,
    )
    write_prediction_frame(oof, oof_path)
    summary = {
        "scenario": scenario,
        "stage": stage,
        "backbone": backbone_name,
        "pretraining": pretraining,
        "feature_set": feature_set,
        "pooled_metrics": compute_metrics(oof["probability"].to_numpy(), oof["true_label"].to_numpy()),
        "fold_metrics": fold_metrics,
        "oof_path": str(oof_path),
    }
    atomic_write_json(oof_path.with_suffix(".summary.json"), summary)
    return summary

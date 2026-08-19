"""Immutable patient-grouped fold and deployment manifest generation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import cfg
from src.data.dataset import load_official_partitions
from src.protocol.contracts import file_sha256


MANIFEST_COLUMNS = ["image_index", "patient_id", "true_label", "fold"]
DEPLOYMENT_COLUMNS = ["image_index", "patient_id", "true_label", "split"]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _base_training_pool(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, official_test = load_official_partitions(
        frame, cfg.paths.train_list_path, cfg.paths.test_list_path
    )
    train = train.sort_values("Image Index", kind="mergesort").reset_index(drop=True)
    official_test = official_test.sort_values("Image Index", kind="mergesort").reset_index(drop=True)
    if train.empty or official_test.empty:
        raise ValueError("Official NIH partitions must both be non-empty")
    return train, official_test


def create_primary_fold_manifest(training_pool: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(
        n_splits=cfg.data.cv_splits,
        shuffle=True,
        random_state=cfg.train.seed,
    )
    assignments = np.full(len(training_pool), -1, dtype=np.int64)
    y = training_pool["binary_label"].to_numpy(dtype=np.int64)
    groups = training_pool["Patient ID"].to_numpy(dtype=np.int64)
    placeholder = np.zeros((len(training_pool), 1), dtype=np.float32)
    for fold, (_, validation_indices) in enumerate(splitter.split(placeholder, y, groups)):
        if (assignments[validation_indices] != -1).any():
            raise AssertionError("A row was assigned to more than one validation fold")
        assignments[validation_indices] = fold
    if (assignments < 0).any():
        raise AssertionError("Some training rows were not assigned to a fold")
    return pd.DataFrame({
        "image_index": training_pool["Image Index"].astype(str),
        "patient_id": groups,
        "true_label": y,
        "fold": assignments,
    })


def create_deployment_manifest(training_pool: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(
        n_splits=cfg.data.deployment_splits,
        shuffle=True,
        random_state=cfg.train.seed,
    )
    y = training_pool["binary_label"].to_numpy(dtype=np.int64)
    groups = training_pool["Patient ID"].to_numpy(dtype=np.int64)
    placeholder = np.zeros((len(training_pool), 1), dtype=np.float32)
    validation_indices = None
    for fold, (_, candidate_validation) in enumerate(splitter.split(placeholder, y, groups)):
        if fold == cfg.data.deployment_validation_fold:
            validation_indices = candidate_validation
            break
    if validation_indices is None:
        raise AssertionError("Configured deployment validation fold does not exist")
    split = np.full(len(training_pool), "train", dtype=object)
    split[validation_indices] = "validation"
    return pd.DataFrame({
        "image_index": training_pool["Image Index"].astype(str),
        "patient_id": groups,
        "true_label": y,
        "split": split,
    })


def audit_manifests(
    folds: pd.DataFrame,
    deployment: pd.DataFrame,
    official_test: pd.DataFrame,
) -> Dict[str, Any]:
    if list(folds.columns) != MANIFEST_COLUMNS:
        raise ValueError(f"Unexpected folds.csv schema: {list(folds.columns)}")
    if list(deployment.columns) != DEPLOYMENT_COLUMNS:
        raise ValueError(f"Unexpected deployment_split.csv schema: {list(deployment.columns)}")
    if folds["image_index"].duplicated().any() or deployment["image_index"].duplicated().any():
        raise ValueError("Manifest Image Index must be unique")
    if set(folds["image_index"]) != set(deployment["image_index"]):
        raise ValueError("Primary and deployment manifests must cover the same training pool")
    if sorted(folds["fold"].unique().tolist()) != list(range(cfg.data.cv_splits)):
        raise ValueError("Primary manifest does not contain exactly folds 0-4")

    patient_fold_counts = folds.groupby("patient_id")["fold"].nunique()
    if int(patient_fold_counts.max()) != 1:
        raise ValueError("Patient leakage detected across CV folds")
    patient_split_counts = deployment.groupby("patient_id")["split"].nunique()
    if int(patient_split_counts.max()) != 1:
        raise ValueError("Patient leakage detected in deployment split")
    training_patients = set(folds["patient_id"])
    official_test_patients = set(official_test["Patient ID"])
    overlap = training_patients & official_test_patients
    if overlap:
        raise ValueError(f"Official train/test patient overlap: {len(overlap)}")

    fold_summary = []
    for fold, part in folds.groupby("fold", sort=True):
        fold_summary.append({
            "fold": int(fold),
            "images": int(len(part)),
            "patients": int(part["patient_id"].nunique()),
            "abnormal_prevalence": float(part["true_label"].mean()),
        })
    deployment_summary = []
    for split, part in deployment.groupby("split", sort=True):
        deployment_summary.append({
            "split": str(split),
            "images": int(len(part)),
            "patients": int(part["patient_id"].nunique()),
            "abnormal_prevalence": float(part["true_label"].mean()),
        })
    return {
        "training_images": int(len(folds)),
        "training_patients": int(folds["patient_id"].nunique()),
        "official_test_images": int(len(official_test)),
        "official_test_patients": int(official_test["Patient ID"].nunique()),
        "patient_overlap": 0,
        "folds": fold_summary,
        "deployment": deployment_summary,
    }


def generate_manifests(frame: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    training_pool, official_test = _base_training_pool(frame)
    folds = create_primary_fold_manifest(training_pool)
    deployment = create_deployment_manifest(training_pool)
    audit = audit_manifests(folds, deployment, official_test)
    folds_path = output_dir / "folds.csv"
    deployment_path = output_dir / "deployment_split.csv"
    _atomic_csv(folds, folds_path)
    _atomic_csv(deployment, deployment_path)
    return {
        "folds_path": folds_path,
        "deployment_path": deployment_path,
        "fold_manifest_hash": file_sha256(folds_path),
        "deployment_split_hash": file_sha256(deployment_path),
        "audit": audit,
    }


def validate_manifest_hash(path: Path, expected_hash: str) -> None:
    actual = file_sha256(path)
    if actual != expected_hash:
        raise RuntimeError(f"Immutable manifest changed: {path}; expected={expected_hash}, actual={actual}")

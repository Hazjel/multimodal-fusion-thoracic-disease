"""Validated, read-only access to canonical OOF reporting artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


REQUIRED_PREDICTION_COLUMNS = {
    "image_index",
    "patient_id",
    "fold",
    "split",
    "true_label",
    "probability",
    "prediction_0_5",
    "model",
    "scenario",
    "feature_set",
    "pretraining",
    "protocol_hash",
    "semantic_config_hash",
    "run_id",
}


@dataclass(frozen=True)
class CanonicalContext:
    project_root: Path
    protocol_dir: Path
    protocol: dict
    folds: pd.DataFrame

    @property
    def protocol_hash(self) -> str:
        return str(self.protocol["protocol_hash"])


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "run_experiment.py").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Project root containing run_experiment.py was not found")


def load_canonical_context(start: Path | str | None = None) -> CanonicalContext:
    """Locate the sole frozen protocol and validate its immutable fold manifest."""
    project_root = _find_project_root(Path(start or Path.cwd()))
    candidates = sorted(
        path.parent
        for path in (project_root / "results" / "canonical").glob("*/protocol.json")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one canonical protocol, found {len(candidates)}"
        )
    protocol_dir = candidates[0]
    protocol = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    protocol_hash = str(protocol.get("protocol_hash", ""))
    if protocol.get("status") != "FROZEN":
        raise RuntimeError("Canonical protocol is not FROZEN")
    if protocol_dir.name != protocol_hash:
        raise RuntimeError("Protocol directory name does not match protocol_hash")

    folds = pd.read_csv(protocol_dir / "folds.csv")
    required = {"image_index", "patient_id", "true_label", "fold"}
    missing = sorted(required.difference(folds.columns))
    if missing:
        raise RuntimeError(f"folds.csv is missing columns: {missing}")
    if folds["image_index"].duplicated().any():
        raise RuntimeError("folds.csv contains duplicate image_index values")
    if sorted(folds["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("folds.csv must contain folds 0-4")
    if folds.groupby("patient_id")["fold"].nunique().max() != 1:
        raise RuntimeError("Patient overlap detected across primary CV folds")
    return CanonicalContext(project_root, protocol_dir, protocol, folds)


def require_stage_success(context: CanonicalContext, relative_directory: str) -> Path:
    stage_dir = context.protocol_dir / relative_directory
    if not (stage_dir / "_SUCCESS").is_file():
        raise RuntimeError(f"Stage is not complete: {relative_directory}")
    return stage_dir


def load_oof_predictions(
    context: CanonicalContext,
    path: Path | str,
    *,
    expected_stage_directory: str,
) -> pd.DataFrame:
    """Read an OOF CSV and hard-fail on schema, provenance, or coverage drift."""
    stage_dir = require_stage_success(context, expected_stage_directory)
    resolved = Path(path).resolve()
    if stage_dir.resolve() not in resolved.parents:
        raise RuntimeError(f"OOF artifact is outside {expected_stage_directory}: {resolved}")
    frame = pd.read_csv(resolved)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction artifact is missing columns: {missing}")
    if frame.empty:
        raise RuntimeError(f"Prediction artifact is empty: {resolved.name}")
    if set(frame["protocol_hash"].astype(str)) != {context.protocol_hash}:
        raise RuntimeError(f"protocol_hash mismatch in {resolved.name}")
    if set(frame["split"].astype(str)) != {"validation"}:
        raise RuntimeError(f"Non-OOF split found in {resolved.name}")
    if frame["image_index"].duplicated().any():
        raise RuntimeError(f"Duplicate OOF image_index found in {resolved.name}")
    if sorted(frame["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Incomplete fold coverage in {resolved.name}")

    expected = context.folds.rename(
        columns={
            "patient_id": "manifest_patient_id",
            "true_label": "manifest_true_label",
            "fold": "manifest_fold",
        }
    )
    audited = frame.merge(expected, on="image_index", how="outer", indicator=True)
    if not (audited["_merge"] == "both").all():
        raise RuntimeError(f"OOF coverage differs from folds.csv in {resolved.name}")
    comparisons = (
        ("patient_id", "manifest_patient_id"),
        ("true_label", "manifest_true_label"),
        ("fold", "manifest_fold"),
    )
    for observed, manifest in comparisons:
        if not np.array_equal(
            audited[observed].to_numpy(), audited[manifest].to_numpy()
        ):
            raise RuntimeError(f"{observed} differs from folds.csv in {resolved.name}")
    return frame.sort_values("image_index").reset_index(drop=True)


def pooled_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = frame["true_label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fold_metric_frame(models: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in models.items():
        for fold, subset in frame.groupby("fold", sort=True):
            metrics = pooled_metrics(subset)
            rows.append({"model": name, "fold": int(fold), **metrics})
    return pd.DataFrame(rows)


def metric_table(models: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    folds = fold_metric_frame(models)
    rows = []
    for name, frame in models.items():
        pooled = pooled_metrics(frame)
        model_folds = folds[folds["model"] == name]
        rows.append(
            {
                "model": name,
                "roc_auc_pooled": pooled["roc_auc"],
                "roc_auc_mean": model_folds["roc_auc"].mean(),
                "roc_auc_sd": model_folds["roc_auc"].std(ddof=1),
                "ap_pooled": pooled["average_precision"],
                "ap_mean": model_folds["average_precision"].mean(),
                "ap_sd": model_folds["average_precision"].std(ddof=1),
                "brier_score": pooled["brier_score"],
                "accuracy_0.5": pooled["accuracy"],
                "sensitivity_0.5": pooled["sensitivity"],
                "specificity_0.5": pooled["specificity"],
                "f1_0.5": pooled["f1"],
                "n_images": len(frame),
                "n_patients": frame["patient_id"].nunique(),
            }
        )
    return pd.DataFrame(rows).set_index("model")

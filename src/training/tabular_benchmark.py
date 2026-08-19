"""C1 tabular characterization benchmark under the frozen outer folds."""
from __future__ import annotations

import os
import pickle
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from configs.config import cfg
from src.data.dataset import (
    build_image_index,
    load_and_prepare_metadata,
    load_official_training_pool,
    raw_semantic_tabular_frame,
)
from src.evaluation import (
    compute_metrics,
    prediction_frame_from_arrays,
    validate_oof_coverage,
    write_prediction_frame,
)
from src.protocol.contracts import (
    atomic_write_json,
    git_commit,
    read_json,
    semantic_config_hash,
)
from src.protocol.environment import collect_environment, environment_hash
from src.protocol.registry import upsert_registry
from src.protocol.stages import load_frozen_protocol, oof_path_for
from src.training.cv import run_cross_validation


REALMLP_FROZEN_PARAMS: Dict[str, Any] = {
    "n_cv": 1,
    "n_refit": 0,
    "n_epochs": 256,
    "batch_size": 256,
    "hidden_sizes": [256, 256, 256],
    "lr": 0.04,
    "val_metric_name": "cross_entropy",
    "use_ls": False,
    "calibration_method": None,
    "random_state": 42,
}

TABM_FROZEN_PARAMS: Dict[str, Any] = {
    "n_cv": 1,
    "n_refit": 0,
    "arch_type": "tabm-mini",
    "tabm_k": 32,
    "num_emb_type": "none",
    "batch_size": 256,
    "lr": 2e-3,
    "n_epochs": 100,
    "patience": 16,
    "val_metric_name": "cross_entropy",
    "random_state": 42,
}

TABULAR_MODELS = ("canonical_mlp", "realmlp", "tabm")


def build_pytabkit_estimator(model_name: str, *, device: str, verbosity: int = 1):
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier

    if model_name == "realmlp":
        return RealMLP_TD_Classifier(
            device=device, verbosity=verbosity, **REALMLP_FROZEN_PARAMS
        )
    if model_name == "tabm":
        return TabM_D_Classifier(
            device=device, verbosity=verbosity, **TABM_FROZEN_PARAMS
        )
    raise ValueError(f"Unsupported PyTabKit benchmark model: {model_name}")


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _run_pytabkit_model(
    model_name: str,
    *,
    protocol_dir: Path,
    device: str,
) -> Dict[str, Any]:
    protocol = load_frozen_protocol(protocol_dir)
    protocol_hash_value = protocol["protocol_hash"]
    implementation = git_commit(cfg.paths.project_root)
    environment = collect_environment(implementation)
    environment_hash_value = environment_hash(environment)
    pytabkit_version = version("pytabkit")

    image_index = build_image_index(cfg.paths.image_dirs)
    metadata = load_and_prepare_metadata(cfg.paths.csv_path, image_index)
    training_pool = load_official_training_pool(metadata, cfg.paths.train_list_path)
    manifest = pd.read_csv(Path(protocol_dir) / "folds.csv")
    manifest_lookup = manifest.set_index("image_index")["fold"]
    training_pool = training_pool.copy()
    training_pool["fold"] = training_pool["Image Index"].map(manifest_lookup)
    if training_pool["fold"].isna().any():
        raise RuntimeError("Training pool contains images absent from folds.csv")

    fold_predictions: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []
    frozen_params = (
        REALMLP_FROZEN_PARAMS if model_name == "realmlp" else TABM_FROZEN_PARAMS
    )
    resolved = {
        "stage": "C1",
        "model": model_name,
        "feature_set": "D",
        "device": device,
        "pytabkit_version": pytabkit_version,
        "estimator_params": frozen_params,
        "runtime": cfg.scientific_runtime_values(),
    }
    categorical = ["Patient Gender", "View Position"]

    for fold in range(cfg.data.cv_splits):
        semantic_hash = semantic_config_hash(
            protocol_hash_value=protocol_hash_value,
            selected_architecture=model_name,
            weight_checksum=f"pytabkit-{pytabkit_version}-{model_name}",
            fold=fold,
            feature_set="D",
            resolved_runtime_config=resolved,
            environment_hash=environment_hash_value,
            implementation_commit=implementation,
        )
        run_id = f"C1-{model_name}-D-fold{fold}-{semantic_hash[:12]}"
        run_dir = Path(protocol_dir) / "runs" / run_id
        prediction_path = run_dir / "predictions.csv"
        metrics_path = run_dir / "metrics.json"
        success_path = run_dir / "_SUCCESS"
        if success_path.exists() and prediction_path.exists() and metrics_path.exists():
            fold_predictions.append(pd.read_csv(prediction_path))
            fold_metrics.append(read_json(metrics_path))
            continue

        train_frame = training_pool[training_pool["fold"] != fold].reset_index(drop=True)
        validation_frame = training_pool[training_pool["fold"] == fold].reset_index(drop=True)
        x_train = raw_semantic_tabular_frame(train_frame, "D")
        x_validation = raw_semantic_tabular_frame(validation_frame, "D")
        y_train = train_frame["binary_label"].to_numpy(dtype=np.int64)
        y_validation = validation_frame["binary_label"].to_numpy(dtype=np.int64)
        estimator = build_pytabkit_estimator(model_name, device=device)
        started = time.perf_counter()
        estimator.fit(
            x_train,
            y_train,
            X_val=x_validation,
            y_val=y_validation,
            cat_col_names=categorical,
        )
        elapsed_seconds = time.perf_counter() - started
        probabilities = estimator.predict_proba(x_validation)[:, 1]
        prediction_frame = prediction_frame_from_arrays(
            image_indices=validation_frame["Image Index"],
            patient_ids=validation_frame["Patient ID"],
            labels=y_validation,
            probabilities=probabilities,
            fold=fold,
            split="validation",
            model_name=model_name,
            scenario="S1",
            feature_set="D",
            pretraining="not_applicable",
            checkpoint_epoch=-1,
            protocol_hash=protocol_hash_value,
            semantic_config_hash=semantic_hash,
            run_id=run_id,
        )
        metrics = compute_metrics(probabilities, y_validation)
        metrics["training_wall_seconds"] = float(elapsed_seconds)
        write_prediction_frame(prediction_frame, prediction_path)
        atomic_write_json(metrics_path, metrics)
        atomic_write_json(run_dir / "resolved_estimator.json", resolved)
        _atomic_pickle(estimator, run_dir / "model.pkl")
        upsert_registry(Path(protocol_dir) / "experiment_registry.csv", {
            "run_id": run_id,
            "phase": "C1",
            "scenario": "S1",
            "model": model_name,
            "fold": fold,
            "feature_set": "D",
            "pretraining": "not_applicable",
            "protocol_hash": protocol_hash_value,
            "semantic_config_hash": semantic_hash,
            "implementation_commit": implementation,
            "environment_hash": environment_hash_value,
            "pos_weight": "not_applicable",
            "best_epoch": "model_managed",
            "best_validation_auc": metrics["roc_auc"],
            "status": "done",
            "artifact_path": str(run_dir),
        })
        success_path.write_text("ok\n", encoding="utf-8")
        fold_predictions.append(prediction_frame)
        fold_metrics.append(metrics)

    oof = pd.concat(fold_predictions, ignore_index=True)
    validate_oof_coverage(oof, manifest["image_index"])
    oof_path = oof_path_for(
        protocol_dir,
        stage="C1",
        scenario="S1",
        model=model_name,
        pretraining="not_applicable",
        feature_set="D",
    )
    write_prediction_frame(oof, oof_path)
    summary = {
        "stage": "C1",
        "model": model_name,
        "feature_set": "D",
        "outer_metrics": compute_metrics(
            oof["probability"].to_numpy(), oof["true_label"].to_numpy()
        ),
        "fold_metrics": fold_metrics,
        "internal_fitting_metric": "cross_entropy",
        "oof_path": str(oof_path),
    }
    atomic_write_json(oof_path.with_suffix(".summary.json"), summary)
    return summary


def run_tabular_benchmark(
    *,
    protocol_dir: Path,
    models: Iterable[str] = TABULAR_MODELS,
    device: str | None = None,
) -> Dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    load_frozen_protocol(protocol_dir)
    selected = tuple(dict.fromkeys(models))
    unknown = set(selected) - set(TABULAR_MODELS)
    if unknown:
        raise ValueError(f"Unknown C1 models: {sorted(unknown)}")
    pytab_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    summaries: Dict[str, Any] = {}
    for model_name in selected:
        if model_name == "canonical_mlp":
            summaries[model_name] = run_cross_validation(
                "S1",
                stage="C1",
                protocol_dir=protocol_dir,
                backbone_name="canonical_mlp",
                pretraining="not_applicable",
                feature_set="D",
            )
        else:
            summaries[model_name] = _run_pytabkit_model(
                model_name, protocol_dir=protocol_dir, device=pytab_device
            )
    if all(
        oof_path_for(
            protocol_dir,
            stage="C1",
            scenario="S1",
            model=model_name,
            pretraining="not_applicable",
            feature_set="D",
        ).exists()
        for model_name in TABULAR_MODELS
    ):
        (protocol_dir / "screening" / "tabular" / "_SUCCESS").write_text(
            "C1 complete\n", encoding="utf-8"
        )
    return summaries

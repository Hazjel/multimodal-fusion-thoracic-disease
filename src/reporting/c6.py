"""Canonical C6 statistics, calibration, and XAI orchestration.

C6 consumes only immutable out-of-fold artifacts from C4/C5 and the official
NIH training pool.  It never opens the official test manifest.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_curve

from configs.config import cfg
from src.data.dataset import (
    NIHChestXrayDataset,
    TABULAR_FEATURE_SETS,
    build_image_index,
    get_transforms,
    load_and_prepare_metadata,
    load_official_training_pool,
)
from src.evaluation import calibration_table
from src.evaluation.stats import paired_patient_cluster_bootstrap
from src.models.architectures import build_model
from src.protocol.contracts import atomic_write_json, file_sha256, git_commit
from src.protocol.cuda_reproducibility import (
    configure_cublas_workspace,
    require_cublas_workspace,
)
from src.protocol.environment import collect_environment, environment_hash
from src.protocol.execution_environment import ensure_stage_environment
from src.protocol.stages import load_frozen_protocol, load_model_lock, oof_path_for
from src.reporting.canonical import (
    fold_metric_frame,
    load_canonical_context,
    load_oof_predictions,
    metric_table,
)
from src.xai import (
    compute_gradcam,
    compute_shap_values,
    make_shap_explainer,
    overlay_gradcam,
    plot_gradcam_grid,
    plot_shap_summary,
    proportional_oof_indices,
)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _main_oof_paths(protocol_dir: Path, lock: Mapping[str, Any]) -> Dict[str, Path]:
    backbone = str(lock["selected_backbone"])
    pretraining = str(lock["selected_pretraining"])
    return {
        "S1-D": oof_path_for(
            protocol_dir, stage="C4", scenario="S1", model="canonical_mlp",
            pretraining="not_applicable", feature_set="D",
        ),
        "S2-D": oof_path_for(
            protocol_dir, stage="C4", scenario="S2", model=backbone,
            pretraining=pretraining, feature_set="D",
        ),
        "S3-D": oof_path_for(
            protocol_dir, stage="C4", scenario="S3", model=backbone,
            pretraining=pretraining, feature_set="D",
        ),
    }


def load_c6_oof(protocol_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load and audit all C4/C5 OOF frames used by C6."""
    models = load_main_oof(protocol_dir)
    context = load_canonical_context(protocol_dir)
    lock = load_model_lock(protocol_dir)
    for scenario in ("S1", "S3"):
        for feature_set in ("A", "B", "C"):
            model = "canonical_mlp" if scenario == "S1" else lock["selected_backbone"]
            pretraining = (
                "not_applicable" if scenario == "S1" else lock["selected_pretraining"]
            )
            path = oof_path_for(
                protocol_dir,
                stage="C5",
                scenario=scenario,
                model=model,
                pretraining=pretraining,
                feature_set=feature_set,
            )
            models[f"{scenario}-{feature_set}"] = load_oof_predictions(
                context, path, expected_stage_directory="ablation"
            )
    return models


def load_main_oof(protocol_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load only the C4 OOF frames needed by SHAP and Grad-CAM."""
    context = load_canonical_context(protocol_dir)
    if context.protocol_dir.resolve() != Path(protocol_dir).resolve():
        raise RuntimeError("Requested protocol directory is not the canonical protocol")
    lock = load_model_lock(protocol_dir)
    return {
        name: load_oof_predictions(
            context, path, expected_stage_directory="main"
        )
        for name, path in _main_oof_paths(protocol_dir, lock).items()
    }


def align_paired_oof(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    probability_a: str = "probability_a",
    probability_b: str = "probability_b",
) -> pd.DataFrame:
    """One-to-one OOF alignment used by all paired C6 estimands."""
    keys = ["image_index", "patient_id", "true_label", "fold"]
    left = frame_a[keys + ["probability"]].rename(
        columns={"probability": probability_a}
    )
    right = frame_b[keys + ["probability"]].rename(
        columns={"probability": probability_b}
    )
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("Paired OOF artifacts do not have identical coverage")
    return merged.sort_values("image_index").reset_index(drop=True)


def youden_threshold(frame: pd.DataFrame) -> Dict[str, float]:
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    scores = frame["probability"].to_numpy(dtype=np.float64)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    objective = true_positive_rate - false_positive_rate
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise RuntimeError("Youden threshold calculation produced no finite threshold")
    candidates = np.flatnonzero(finite)
    best = int(candidates[np.argmax(objective[candidates])])
    return {
        "threshold": float(thresholds[best]),
        "youden_j": float(objective[best]),
        "sensitivity": float(true_positive_rate[best]),
        "specificity": float(1.0 - false_positive_rate[best]),
        "role": "secondary_operating_point_only",
    }


def _plot_reliability(
    tables: Mapping[str, pd.DataFrame], save_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    for name, table in tables.items():
        observed = table[table["count"] > 0]
        axis.plot(
            observed["mean_score"], observed["fraction_positive"],
            marker="o", linewidth=1.5, label=name,
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean model score", ylabel="Fraction abnormal")
    axis.set_title("OOF reliability diagram (10 uniform bins)")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_statistics(protocol_dir: Path) -> Dict[str, Any]:
    """Generate threshold-independent metrics, calibration, and paired CIs."""
    protocol_dir = Path(protocol_dir)
    output = protocol_dir / "statistics"
    marker = output / "_STATISTICS_SUCCESS"
    summary_path = output / "statistics_summary.json"
    if marker.is_file() and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print("[C6 statistics] Auditing C4/C5 OOF artifacts...", flush=True)
    models = load_c6_oof(protocol_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = metric_table(models).reset_index()
    fold_metrics = fold_metric_frame(models)
    _atomic_write_csv(metrics, output / "metrics.csv")
    _atomic_write_csv(fold_metrics, output / "fold_metrics.csv")

    calibration_dir = output / "calibration"
    calibration_tables: Dict[str, pd.DataFrame] = {}
    for name in ("S1-D", "S2-D", "S3-D"):
        table = calibration_table(
            models[name]["probability"].to_numpy(),
            models[name]["true_label"].to_numpy(),
            n_bins=cfg.evaluation.calibration_bins,
        )
        calibration_tables[name] = table
        _atomic_write_csv(table, calibration_dir / f"{name}.csv")
    _plot_reliability(calibration_tables, calibration_dir / "reliability_diagram.png")

    thresholds = {
        name: youden_threshold(models[name])
        for name in ("S1-D", "S2-D", "S3-D")
    }
    atomic_write_json(output / "youden_thresholds.json", thresholds)

    comparison_specs = [
        ("primary_S3-D_minus_S2", "S3-D", "S2-D"),
        ("ablation_S3-A_minus_S2", "S3-A", "S2-D"),
        ("ablation_S3-B_minus_S2", "S3-B", "S2-D"),
        ("ablation_S3-C_minus_S2", "S3-C", "S2-D"),
        ("descriptive_S3-B_minus_S3-A", "S3-B", "S3-A"),
        ("descriptive_S3-C_minus_S3-A", "S3-C", "S3-A"),
        ("descriptive_S3-D_minus_S3-A", "S3-D", "S3-A"),
    ]
    bootstrap_path = output / "bootstrap_comparisons.json"
    comparisons: Dict[str, Any] = {}
    if bootstrap_path.is_file():
        comparisons = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    for index, (name, model_a, model_b) in enumerate(comparison_specs):
        if name in comparisons:
            print(f"[C6 statistics] Reusing completed bootstrap: {name}", flush=True)
            continue
        print(
            f"[C6 statistics] Bootstrap {index + 1}/{len(comparison_specs)}: {name} "
            f"({cfg.evaluation.bootstrap_replicates} patient replicates)",
            flush=True,
        )
        aligned = align_paired_oof(models[model_a], models[model_b])
        result = paired_patient_cluster_bootstrap(
            aligned,
            probability_a="probability_a",
            probability_b="probability_b",
            n_boot=cfg.evaluation.bootstrap_replicates,
            alpha=cfg.evaluation.bootstrap_alpha,
            seed=cfg.train.seed + index,
        )
        result.update({"model_a": model_a, "model_b": model_b})
        comparisons[name] = result
        atomic_write_json(bootstrap_path, comparisons)
    comparisons["ablation_S3-D_minus_S2"] = {
        **comparisons["primary_S3-D_minus_S2"],
        "alias_of": "primary_S3-D_minus_S2",
    }
    atomic_write_json(bootstrap_path, comparisons)

    summary = {
        "status": "COMPLETE",
        "protocol_hash": Path(protocol_dir).name,
        "unit_of_prediction": "image_exam",
        "cluster_unit": "patient_id",
        "primary_estimand": "AUC(S3-D)-AUC(S2-D)",
        "bootstrap_replicates": cfg.evaluation.bootstrap_replicates,
        "bootstrap_caveat": (
            "Conditional on fitted CV models; no retraining occurs within replicates."
        ),
        "metrics_path": str(output / "metrics.csv"),
        "fold_metrics_path": str(output / "fold_metrics.csv"),
        "bootstrap_path": str(bootstrap_path),
        "calibration_role": "diagnostic_only_not_calibrated_probability",
    }
    atomic_write_json(summary_path, summary)
    marker.write_text("C6 statistics complete\n", encoding="utf-8")
    return summary


def select_local_xai_cases(
    frame: pd.DataFrame, *, threshold: float = 0.5, per_category: int = 2
) -> pd.DataFrame:
    """Select deterministic median-score TP/TN/FP/FN cases."""
    selected = frame.copy()
    predicted = (selected["probability"].to_numpy() >= threshold).astype(int)
    labels = selected["true_label"].to_numpy(dtype=int)
    selected["category"] = np.select(
        [
            (predicted == 1) & (labels == 1),
            (predicted == 0) & (labels == 0),
            (predicted == 1) & (labels == 0),
            (predicted == 0) & (labels == 1),
        ],
        ["TP", "TN", "FP", "FN"],
        default="INVALID",
    )
    rows = []
    for category in ("TP", "TN", "FP", "FN"):
        subset = selected[selected["category"] == category].copy()
        if len(subset) < per_category:
            raise RuntimeError(f"Not enough {category} cases for local XAI")
        median = float(subset["probability"].median())
        subset["distance_to_category_median"] = (subset["probability"] - median).abs()
        subset["category_median_probability"] = median
        subset = subset.sort_values(
            ["distance_to_category_median", "image_index"], kind="mergesort"
        ).head(per_category)
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def _load_training_pool(protocol_dir: Path):
    image_index = build_image_index(cfg.paths.image_dirs)
    metadata = load_and_prepare_metadata(cfg.paths.csv_path, image_index)
    training_pool = load_official_training_pool(metadata, cfg.paths.train_list_path)
    manifest = pd.read_csv(Path(protocol_dir) / "folds.csv")
    fold_lookup = manifest.set_index("image_index")["fold"]
    training_pool = training_pool.copy()
    training_pool["fold"] = training_pool["Image Index"].map(fold_lookup)
    if training_pool["fold"].isna().any():
        raise RuntimeError("Training pool differs from frozen folds.csv")
    return training_pool, image_index


def _load_fold_model(
    protocol_dir: Path,
    oof_fold: pd.DataFrame,
    *,
    scenario: str,
    fold: int,
    backbone: str,
    pretraining: str,
    device: torch.device,
) -> tuple[torch.nn.Module, Path]:
    run_ids = oof_fold["run_id"].astype(str).unique().tolist()
    if len(run_ids) != 1:
        raise RuntimeError(f"Fold {fold} has {len(run_ids)} run IDs for {scenario}")
    run_dir = Path(protocol_dir) / "runs" / run_ids[0]
    if not (run_dir / "_SUCCESS").is_file():
        raise RuntimeError(f"Canonical run is incomplete: {run_ids[0]}")
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        scenario,
        backbone_name=backbone,
        pretraining=pretraining,
        fold=fold,
        tabular_input_dim=4,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, run_dir


def run_shap(
    protocol_dir: Path, *, device: torch.device, nsamples: int = 128
) -> Dict[str, Any]:
    """Generate resumable fold-specific, actual-image-conditioned SHAP."""
    protocol_dir = Path(protocol_dir)
    output = protocol_dir / "xai" / "shap"
    marker = output / "_SUCCESS"
    summary_path = output / "summary.json"
    if marker.is_file() and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print("[C6 SHAP] Auditing main OOF artifacts and training-pool manifest...", flush=True)
    models = load_main_oof(protocol_dir)
    s3 = models["S3-D"]
    lock = load_model_lock(protocol_dir)
    training_pool, image_index = _load_training_pool(protocol_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_results = []
    feature_columns = list(TABULAR_FEATURE_SETS["D"])
    display_names = ["Age", "Gender", "View Position", "Follow-up #"]

    for fold in range(cfg.data.cv_splits):
        print(f"[C6 SHAP] Fold {fold + 1}/{cfg.data.cv_splits}", flush=True)
        fold_output = output / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        validation_oof = s3[s3["fold"] == fold].sort_values("image_index").reset_index(drop=True)
        local_indices = proportional_oof_indices(
            validation_oof["true_label"].to_numpy(),
            n_samples=cfg.evaluation.shap_samples_per_fold,
            seed=cfg.train.seed + fold,
        )
        selected_oof = validation_oof.iloc[local_indices].copy().reset_index(drop=True)
        _atomic_write_csv(
            selected_oof[["image_index", "patient_id", "fold", "true_label", "probability", "run_id"]],
            fold_output / "selected_cases.csv",
        )

        model, run_dir = _load_fold_model(
            protocol_dir, validation_oof, scenario="S3", fold=fold,
            backbone=lock["selected_backbone"], pretraining=lock["selected_pretraining"],
            device=device,
        )
        scaler_path = run_dir / "scaler.pkl"
        if not scaler_path.is_file():
            raise RuntimeError(f"Missing fold scaler: {scaler_path}")
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)

        train_frame = training_pool[training_pool["fold"] != fold].reset_index(drop=True)
        validation_frame = training_pool[training_pool["fold"] == fold].reset_index(drop=True)
        train_dataset = NIHChestXrayDataset(
            train_frame, scaler=scaler, tabular_cols=feature_columns, modalities=("tabular",)
        )
        validation_dataset = NIHChestXrayDataset(
            validation_frame,
            image_index=image_index,
            transform=get_transforms(False),
            scaler=scaler,
            tabular_cols=feature_columns,
            modalities=("image", "tabular"),
        )
        validation_positions = {
            str(name): index for index, name in enumerate(validation_frame["Image Index"])
        }
        rng = np.random.RandomState(cfg.train.seed + fold)
        background_positions = rng.choice(
            len(train_dataset),
            size=min(cfg.evaluation.shap_background_size, len(train_dataset)),
            replace=False,
        )
        background = np.stack(
            [train_dataset[int(index)]["tabular"].numpy() for index in background_positions]
        ).astype(np.float32)
        np.save(fold_output / "background_scaled.npy", background)
        background_manifest = train_frame.iloc[background_positions][
            ["Image Index", "Patient ID", "binary_label"]
        ].rename(columns={"Image Index": "image_index", "Patient ID": "patient_id", "binary_label": "true_label"})
        _atomic_write_csv(background_manifest, fold_output / "background_cases.csv")

        partial_path = fold_output / "shap_values.csv"
        completed = pd.read_csv(partial_path) if partial_path.is_file() else pd.DataFrame()
        completed_names = set(completed.get("image_index", pd.Series(dtype=str)).astype(str))
        rows = completed.to_dict("records")
        for _, oof_row in selected_oof.iterrows():
            image_name = str(oof_row["image_index"])
            if image_name in completed_names:
                continue
            print(
                f"[C6 SHAP] fold={fold} case={len(rows) + 1}/"
                f"{cfg.evaluation.shap_samples_per_fold} image={image_name}",
                flush=True,
            )
            item = validation_dataset[validation_positions[image_name]]
            image_tensor = item["image"].unsqueeze(0).to(device)
            tabular_scaled = item["tabular"].numpy().astype(np.float32)
            explainer = make_shap_explainer(
                model, background, device, fixed_image=image_tensor
            )
            values = np.asarray(
                compute_shap_values(explainer, tabular_scaled[None, :], nsamples=nsamples)
            ).squeeze()
            if values.shape != (4,):
                raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}")
            with torch.no_grad():
                model_score = float(torch.sigmoid(model(
                    image=image_tensor,
                    tabular=torch.from_numpy(tabular_scaled).unsqueeze(0).to(device),
                ).reshape(-1)[0]).item())
            if not np.isclose(
                model_score,
                float(oof_row["probability"]),
                rtol=1e-3,
                atol=1e-3,
            ):
                raise RuntimeError(
                    f"Recomputed S3 score differs from OOF artifact for {image_name}: "
                    f"recomputed={model_score}, oof={oof_row['probability']}"
                )
            expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
            raw_row = validation_frame.iloc[validation_positions[image_name]]
            row: Dict[str, Any] = {
                "image_index": image_name,
                "patient_id": int(oof_row["patient_id"]),
                "fold": fold,
                "true_label": int(oof_row["true_label"]),
                "oof_probability": float(oof_row["probability"]),
                "recomputed_model_score": model_score,
                "expected_value": expected,
                "run_id": str(oof_row["run_id"]),
            }
            for index, (column, display) in enumerate(zip(feature_columns, display_names)):
                slug = display.lower().replace(" ", "_").replace("#", "number")
                row[f"raw_{slug}"] = float(raw_row[column])
                row[f"scaled_{slug}"] = float(tabular_scaled[index])
                row[f"shap_{slug}"] = float(values[index])
            rows.append(row)
            _atomic_write_csv(pd.DataFrame(rows), partial_path)
        fold_results.append(pd.DataFrame(rows))
        (fold_output / "_SUCCESS").write_text("SHAP fold complete\n", encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined = pd.concat(fold_results, ignore_index=True).sort_values(
        ["fold", "image_index"]
    ).reset_index(drop=True)
    expected_count = cfg.data.cv_splits * cfg.evaluation.shap_samples_per_fold
    if len(combined) != expected_count or combined["image_index"].duplicated().any():
        raise RuntimeError(f"SHAP coverage mismatch: expected={expected_count}, rows={len(combined)}")
    combined_path = output / "shap_values.csv"
    _atomic_write_csv(combined, combined_path)
    shap_columns = [f"shap_{name}" for name in ("age", "gender", "view_position", "follow-up_number")]
    scaled_columns = [f"scaled_{name}" for name in ("age", "gender", "view_position", "follow-up_number")]
    mean_absolute = pd.DataFrame({
        "feature": display_names,
        "mean_absolute_shap": [float(combined[column].abs().mean()) for column in shap_columns],
    }).sort_values("mean_absolute_shap", ascending=False)
    _atomic_write_csv(mean_absolute, output / "mean_absolute_shap.csv")
    plot_shap_summary(
        combined[shap_columns].to_numpy(), combined[scaled_columns].to_numpy(),
        output / "summary.png",
    )
    summary = {
        "status": "COMPLETE",
        "n_oof_cases": len(combined),
        "cases_per_fold": cfg.evaluation.shap_samples_per_fold,
        "background_per_fold": cfg.evaluation.shap_background_size,
        "nsamples": int(nsamples),
        "conditioning": "actual_preprocessed_oof_xray_fixed_per_case",
        "metadata_space": "fold_specific_standard_scaled",
        "interpretation": "average conditional attribution magnitude across OOF cases",
    }
    atomic_write_json(summary_path, summary)
    marker.write_text("C6 OOF SHAP complete\n", encoding="utf-8")
    return summary


def run_gradcam(protocol_dir: Path, *, device: torch.device) -> Dict[str, Any]:
    """Generate paired OOF S2/S3 Grad-CAM for median-score error cases."""
    protocol_dir = Path(protocol_dir)
    output = protocol_dir / "xai" / "gradcam"
    marker = output / "_SUCCESS"
    summary_path = output / "summary.json"
    if marker.is_file() and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print("[C6 Grad-CAM] Auditing and aligning main S2/S3 OOF artifacts...", flush=True)
    models = load_main_oof(protocol_dir)
    s2, s3 = models["S2-D"], models["S3-D"]
    aligned = align_paired_oof(
        s3, s2, probability_a="probability", probability_b="s2_probability"
    )
    run_columns = s3[["image_index", "run_id"]].rename(columns={"run_id": "s3_run_id"})
    run_columns = run_columns.merge(
        s2[["image_index", "run_id"]].rename(columns={"run_id": "s2_run_id"}),
        on="image_index", validate="one_to_one",
    )
    aligned = aligned.merge(run_columns, on="image_index", validate="one_to_one")
    selected = select_local_xai_cases(aligned)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(selected, output / "selected_cases.csv")

    lock = load_model_lock(protocol_dir)
    training_pool, image_index = _load_training_pool(protocol_dir)
    images_for_grid = []
    heatmaps_for_grid = []
    titles = []
    case_rows = []
    for fold, fold_cases in selected.groupby("fold", sort=True):
        fold = int(fold)
        s2_fold = s2[s2["fold"] == fold]
        s3_fold = s3[s3["fold"] == fold]
        s2_model, _ = _load_fold_model(
            protocol_dir, s2_fold, scenario="S2", fold=fold,
            backbone=lock["selected_backbone"], pretraining=lock["selected_pretraining"],
            device=device,
        )
        s3_model, s3_run_dir = _load_fold_model(
            protocol_dir, s3_fold, scenario="S3", fold=fold,
            backbone=lock["selected_backbone"], pretraining=lock["selected_pretraining"],
            device=device,
        )
        with (s3_run_dir / "scaler.pkl").open("rb") as handle:
            scaler = pickle.load(handle)
        validation_frame = training_pool[training_pool["fold"] == fold].reset_index(drop=True)
        dataset = NIHChestXrayDataset(
            validation_frame,
            image_index=image_index,
            transform=get_transforms(False),
            scaler=scaler,
            tabular_cols=TABULAR_FEATURE_SETS["D"],
            modalities=("image", "tabular"),
        )
        positions = {str(name): index for index, name in enumerate(validation_frame["Image Index"])}
        for _, case in fold_cases.iterrows():
            image_name = str(case["image_index"])
            print(
                f"[C6 Grad-CAM] fold={fold} category={case['category']} image={image_name}",
                flush=True,
            )
            case_slug = f"{case['category']}_{image_name.rsplit('.', 1)[0]}"
            case_dir = output / "cases" / case_slug
            case_dir.mkdir(parents=True, exist_ok=True)
            item = dataset[positions[image_name]]
            image_tensor = item["image"].unsqueeze(0).to(device)
            tabular_tensor = item["tabular"].unsqueeze(0).to(device)
            s2_heatmap = compute_gradcam(
                s2_model, image_tensor, None, s2_model.get_cam_target_layer()
            )
            s3_heatmap = compute_gradcam(
                s3_model, image_tensor, tabular_tensor, s3_model.get_cam_target_layer()
            )
            np.save(case_dir / "S2_heatmap.npy", s2_heatmap.astype(np.float32))
            np.save(case_dir / "S3_heatmap.npy", s3_heatmap.astype(np.float32))
            with Image.open(image_index[image_name]) as source:
                original = source.convert("RGB")
            s2_overlay = overlay_gradcam(original, s2_heatmap)
            s3_overlay = overlay_gradcam(original, s3_heatmap)
            Image.fromarray(s2_overlay).save(case_dir / "S2_overlay.png")
            Image.fromarray(s3_overlay).save(case_dir / "S3_overlay.png")
            images_for_grid.extend([original, original])
            heatmaps_for_grid.extend([s2_heatmap, s3_heatmap])
            titles.extend([
                f"{case['category']} {image_name} — S2",
                f"{case['category']} {image_name} — S3",
            ])
            case_rows.append({
                "image_index": image_name,
                "patient_id": int(case["patient_id"]),
                "fold": fold,
                "true_label": int(case["true_label"]),
                "category": str(case["category"]),
                "s2_probability": float(case["s2_probability"]),
                "s3_probability": float(case["probability"]),
                "artifact_directory": str(case_dir),
            })
        del s2_model, s3_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _atomic_write_csv(pd.DataFrame(case_rows), output / "case_artifacts.csv")
    plot_gradcam_grid(
        images_for_grid, heatmaps_for_grid, titles, output / "paired_gradcam_grid.png",
        ncols=2,
    )
    summary = {
        "status": "COMPLETE",
        "n_cases": len(case_rows),
        "selection": "two_per_TP_TN_FP_FN_nearest_category_median_S3_score",
        "comparison": "paired_OOF_S2_vs_S3_same_cases",
    }
    atomic_write_json(summary_path, summary)
    marker.write_text("C6 paired OOF Grad-CAM complete\n", encoding="utf-8")
    return summary


def _finalize_c6(protocol_dir: Path, provenance: Mapping[str, Any]) -> bool:
    protocol_dir = Path(protocol_dir)
    required = (
        protocol_dir / "statistics" / "_STATISTICS_SUCCESS",
        protocol_dir / "xai" / "shap" / "_SUCCESS",
        protocol_dir / "xai" / "gradcam" / "_SUCCESS",
    )
    if not all(path.is_file() for path in required):
        return False
    xai_marker = protocol_dir / "xai" / "_SUCCESS"
    xai_marker.write_text("C6 OOF XAI complete\n", encoding="utf-8")
    artifact_roots = (protocol_dir / "statistics", protocol_dir / "xai")
    artifacts = {}
    for root in artifact_roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name not in {"artifact_manifest.json", "_SUCCESS"}:
                artifacts[str(path.relative_to(protocol_dir)).replace("\\", "/")] = {
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
    manifest = {
        "status": "COMPLETE",
        "protocol_hash": protocol_dir.name,
        **dict(provenance),
        "artifacts": artifacts,
    }
    atomic_write_json(protocol_dir / "statistics" / "artifact_manifest.json", manifest)
    (protocol_dir / "statistics" / "_SUCCESS").write_text(
        "C6 statistics and OOF XAI complete\n", encoding="utf-8"
    )
    return True


def run_c6(
    protocol_dir: Path,
    *,
    component: str = "all",
    device: torch.device | None = None,
    shap_nsamples: int = 128,
) -> Dict[str, Any]:
    if component not in {"all", "statistics", "shap", "gradcam"}:
        raise ValueError(f"Unsupported C6 component: {component}")
    protocol_dir = Path(protocol_dir)
    protocol = load_frozen_protocol(protocol_dir)
    load_model_lock(protocol_dir)
    if not (protocol_dir / "main" / "_SUCCESS").is_file():
        raise RuntimeError("C6 requires C4 main artifacts")
    if not (protocol_dir / "ablation" / "_SUCCESS").is_file():
        raise RuntimeError("C6 requires C5 ablation artifacts")

    configure_cublas_workspace()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_cublas_workspace(device)
    implementation = git_commit(cfg.paths.project_root)
    environment = collect_environment(implementation)
    environment["execution_device"] = str(device)
    environment_hash_value = environment_hash(environment)
    ensure_stage_environment(
        protocol_dir=protocol_dir,
        stage="C6",
        protocol_hash=protocol["protocol_hash"],
        environment_hash=environment_hash_value,
        environment=environment,
        implementation_commit=implementation,
    )
    result: Dict[str, Any] = {
        "component": component,
        "device": str(device),
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
    }
    if component in {"all", "statistics"}:
        result["statistics"] = run_statistics(protocol_dir)
    if component in {"all", "shap"}:
        result["shap"] = run_shap(
            protocol_dir, device=device, nsamples=shap_nsamples
        )
    if component in {"all", "gradcam"}:
        result["gradcam"] = run_gradcam(protocol_dir, device=device)
    result["C6_complete"] = _finalize_c6(
        protocol_dir,
        {
            "implementation_commit": implementation,
            "environment_hash": environment_hash_value,
            "execution_device": str(device),
        },
    )
    return result

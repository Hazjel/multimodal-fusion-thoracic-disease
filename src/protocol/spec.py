"""Scientific specification for protocol v1.0.0.

Only the returned ``scientific_spec`` is included in ``protocol_hash``.
Implementation and environment provenance are attached separately.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from configs.config import cfg


def build_scientific_spec(*, fold_manifest_hash: str, deployment_split_hash: str) -> Dict[str, Any]:
    return {
        "protocol_version": cfg.protocol_version,
        "task": {
            "dataset": "NIH ChestX-ray14",
            "target": "Normal_vs_Abnormal",
            "negative_definition": "Finding Labels equals No Finding",
            "positive_definition": "any listed pathology",
            "prediction_unit": "image_exam",
            "grouping_unit": "Patient ID",
            "metadata": list(cfg.data.tabular_features),
        },
        "splits": {
            "primary_cv": {
                "implementation": "StratifiedGroupKFold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42,
                "source": "official NIH training pool only",
                "fold_manifest_hash": fold_manifest_hash,
            },
            "deployment": {
                "implementation": "StratifiedGroupKFold",
                "n_splits": 10,
                "shuffle": True,
                "random_state": 42,
                "validation_fold": 0,
                "deployment_split_hash": deployment_split_hash,
            },
            "official_test": "secondary holdout with prior exposure; blocked before C7",
        },
        "runtime_contract": cfg.scientific_runtime_values(),
        "architectures": {
            "S1": "p-BN-ReLU-Dropout0.3-64-BN-ReLU-Dropout0.3-128-Linear1",
            "S2": "backbone-GAP-native_to_512-BN-ReLU-Dropout0.4-Linear1",
            "S3": "concat(image512,metadata128)-256-BN-ReLU-Dropout0.4-128-BN-ReLU-Dropout0.4-Linear1",
            "initialization": "S2/S3 image backbone and projection identical per fold; S3 never loads trained S2",
        },
        "fine_tuning": {
            "densenet121": {"frozen": "conv0 through transition3", "trainable": "denseblock4,norm5,projection,classifier"},
            "resnet50": {"frozen": "stem,layer1-layer3", "trainable": "layer4,projection,classifier"},
            "efficientnet_b0": {"frozen": "features[0:6]", "trainable": "features[6:9],projection,classifier"},
            "frozen_batchnorm": "affine parameters and running statistics frozen; frozen BN forced to eval after model.train()",
        },
        "image_preprocessing": {
            "train": [
                "RGB", "Resize256_BILINEAR_antialias", "RandomCrop224",
                "HorizontalFlip0.5", "Rotation10_BILINEAR_expandFalse_fill0",
                "ColorJitter_brightness0.1_contrast0.1", "ToTensor", "ImageNetNormalize",
            ],
            "evaluation": ["RGB", "Resize224_BILINEAR_antialias", "ToTensor", "ImageNetNormalize"],
        },
        "metadata_preprocessing": {
            "S1_S3": "Age clipped 0-100; F=0,M=1; AP=0,PA=1; Follow-up non-negative; fold StandardScaler",
            "RealMLP_TabM": "raw semantic dataframe with official model-specific preprocessing",
            "invalid_values": "hard error",
        },
        "tabular_benchmark": {
            "role": "preliminary characterization; MLP remains S1/S3 encoder",
            "candidates": ["canonical_mlp", "RealMLP_TD_Classifier", "TabM_D_Classifier"],
            "internal_fitting": {"canonical_mlp": "validation ROC-AUC", "RealMLP": "cross_entropy", "TabM": "cross_entropy"},
            "outer_metrics": ["ROC-AUC", "Average Precision"],
        },
        "cnn_selection": {
            "candidates": list(cfg.model.image_candidates),
            "rule": "OOF top candidate; paired patient-cluster CI set; then fold SD, trainable parameters, median wall time",
            "chexnet": "conditional DenseNet provenance audit; ImageNet tie-break when no clear separation",
            "no_model_chasing": True,
        },
        "estimands": {
            "primary": "AUC(S3-D)-AUC(S2)",
            "ablation_vs_S2": ["S3-A", "S3-B", "S3-C", "S3-D"],
            "descriptive_ablation": ["S3-B minus S3-A", "S3-C minus S3-A", "S3-D minus S3-A"],
        },
        "evaluation": {
            "roc_auc": "sklearn.metrics.roc_auc_score",
            "pr_auc": "sklearn.metrics.average_precision_score",
            "threshold": 0.5,
            "youden": "secondary OOF operating point only",
            "calibration": "Brier score plus 10 uniform bins; diagnostic only",
            "bootstrap": "2000 paired Patient-ID cluster replicates; percentile 95% CI; conditional on fitted CV models",
        },
        "xai": {
            "SHAP": "40 proportional OOF cases/fold; 100 training background/fold; same fold scaler; actual image fixed",
            "GradCAM": "OOF paired S2/S3 cases",
            "local_cases": "two each TP,TN,FP,FN nearest category median score",
        },
        "deployment": {
            "model": "final-refit S3 prototype",
            "threshold": 0.5,
            "display": "model score, not calibrated probability",
            "no_new_tuning": True,
        },
    }

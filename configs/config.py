"""Immutable runtime configuration for Canonical Protocol v1.0.0-rc2.

The values in this module mirror the scientific protocol. Canonical entry
points validate the resolved runtime configuration against the frozen
``scientific_spec`` before doing any work.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class PathConfig:
    raw_dataset: Path = Path(os.environ.get("NIH_DATASET_ROOT", r"D:\TA\nih-chest-xrays"))
    project_root: Path = Path(os.environ.get("NIH_PROJECT_ROOT", r"D:\TA\nih-multimodal"))

    @property
    def csv_path(self) -> Path:
        return self.raw_dataset / "Data_Entry_2017.csv"

    @property
    def bbox_path(self) -> Path:
        return self.raw_dataset / "BBox_List_2017.csv"

    @property
    def train_list_path(self) -> Path:
        return self.raw_dataset / "train_val_list.txt"

    @property
    def test_list_path(self) -> Path:
        return self.raw_dataset / "test_list.txt"

    @property
    def image_dirs(self) -> Tuple[Path, ...]:
        return tuple(sorted(self.raw_dataset.glob("images_*/images")))

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def canonical_dir(self) -> Path:
        return self.results_dir / "canonical"

    @property
    def checkpoint_dir(self) -> Path:
        return self.canonical_dir / "checkpoints"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def xai_dir(self) -> Path:
        return self.canonical_dir / "xai"


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 224
    train_resize: int = 256
    label_names: Tuple[str, ...] = (
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax",
    )
    num_classes: int = 1
    tabular_features: Tuple[str, ...] = (
        "Patient Age", "Patient Gender", "View Position", "Follow-up #",
    )
    encoded_tabular_features: Tuple[str, ...] = (
        "Patient Age", "gender_encoded", "view_PA", "Follow-up #",
    )
    age_clip_min: int = 0
    age_clip_max: int = 100
    gender_mapping: Tuple[Tuple[str, int], ...] = (("F", 0), ("M", 1))
    view_mapping: Tuple[Tuple[str, int], ...] = (("AP", 0), ("PA", 1))
    cv_splits: int = 5
    deployment_splits: int = 10
    deployment_validation_fold: int = 0
    num_workers: int = 4
    pin_memory: bool = True


@dataclass(frozen=True)
class ModelConfig:
    image_candidates: Tuple[str, ...] = (
        "densenet121", "resnet50", "efficientnet_b0",
    )
    weight_identifiers: Tuple[Tuple[str, str], ...] = (
        ("densenet121", "DenseNet121_Weights.IMAGENET1K_V1"),
        ("resnet50", "ResNet50_Weights.IMAGENET1K_V2"),
        ("efficientnet_b0", "EfficientNet_B0_Weights.IMAGENET1K_V1"),
    )
    chexnet_weights_path: str = r"D:\TA\nih-multimodal\models\model.pth.tar"
    chexnet_expected_sha256: str = "3777d98828c693da2178650f91679deb9a1eb0f8a96f0f22f1c531d15df9b21d"
    image_feature_dim: int = 512
    tabular_input_dim: int = 4
    tabular_hidden_dims: Tuple[int, ...] = (64, 128)
    tabular_feature_dim: int = 128
    fusion_hidden_dims: Tuple[int, ...] = (256, 128)
    tabular_dropout: float = 0.3
    image_dropout: float = 0.4
    fusion_dropout: float = 0.4


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 30
    lr_backbone: float = 1e-4
    lr_tabular: float = 1e-3
    lr_fusion: float = 5e-4
    weight_decay: float = 1e-5
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    amsgrad: bool = False
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stop_patience: int = 7
    use_amp: bool = True
    gradient_accumulation_steps: int = 2
    gradient_clip_norm: float = 1.0
    seed: int = 42
    checkpoint_metric: str = "roc_auc"
    scheduler_metric: str = "validation_loss"


@dataclass(frozen=True)
class EvaluationConfig:
    decision_threshold: float = 0.5
    calibration_bins: int = 10
    calibration_strategy: str = "uniform"
    bootstrap_replicates: int = 2000
    bootstrap_alpha: float = 0.05
    shap_samples_per_fold: int = 40
    shap_background_size: int = 100


@dataclass(frozen=True)
class Config:
    protocol_version: str = "1.0.0"
    protocol_candidate: str = "1.0.0-rc2"
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def scientific_runtime_values(self) -> Dict[str, Any]:
        """Return path-independent values that must match the protocol."""
        model = asdict(self.model)
        model.pop("chexnet_weights_path")
        return {
            "protocol_version": self.protocol_version,
            "data": asdict(self.data),
            "model": model,
            "training": asdict(self.train),
            "evaluation": asdict(self.evaluation),
        }


cfg = Config()

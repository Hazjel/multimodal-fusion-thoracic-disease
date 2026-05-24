"""
Centralized configuration for NIH Chest X-ray Multimodal Fusion.
All hyperparameters and paths defined here — single source of truth.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration."""
    raw_dataset: Path = Path(r"D:\TA\nih-chest-xrays")
    project_root: Path = Path(r"D:\TA\nih-multimodal")

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
    def image_dirs(self) -> List[Path]:
        """All 12 image subdirectories."""
        return sorted(self.raw_dataset.glob("images_*/images"))

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def checkpoint_dir(self) -> Path:
        return self.project_root / "results" / "checkpoints"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "results" / "figures"

    @property
    def xai_dir(self) -> Path:
        return self.project_root / "results" / "xai"


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing parameters."""
    image_size: int = 224
    # All 14 pathology labels in NIH Chest X-ray14
    label_names: tuple = (
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax",
    )
    num_classes: int = 1  # binary: Normal vs Abnormal (S1-S3)
    # Tabular features extracted from NIH metadata
    tabular_features: tuple = (
        "Patient Age", "Patient Gender", "View Position", "Follow-up #",
    )
    # Age clipping to remove outliers (max 414 in raw data)
    age_clip_min: int = 0
    age_clip_max: int = 100
    # Patient-level split ratio (from official lists)
    val_ratio: float = 0.1
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters."""
    # Image branch
    backbone: str = "densenet121"
    pretrained: bool = True
    # CheXNet: DenseNet-121 pretrained on chest X-ray (stronger than ImageNet)
    use_chexnet: bool = True
    chexnet_weights_path: str = r"D:\TA\nih-multimodal\models\model.pth.tar"
    # DenseNet-121 outputs 1024-dim features after GAP
    image_feature_dim: int = 512
    freeze_backbone_ratio: float = 0.75  # Freeze first 75% of backbone
    # Tabular branch
    tabular_input_dim: int = 4  # age, gender(M/F), view(PA/AP), follow-up
    tabular_hidden_dims: tuple = (64, 128)
    tabular_feature_dim: int = 128
    # Fusion layer
    fusion_hidden_dims: tuple = (256, 128)
    dropout_rate: float = 0.4


@dataclass
class TrainConfig:
    """Training hyperparameters — optimized for RTX 3060 6GB."""
    # Batch size conservative for 6.4GB VRAM with DenseNet-121
    batch_size: int = 16
    num_epochs: int = 30
    # Differential learning rates
    lr_backbone: float = 1e-4
    lr_tabular: float = 1e-3
    lr_fusion: float = 5e-4
    weight_decay: float = 1e-5
    # Scheduler
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    # Early stopping
    early_stop_patience: int = 7
    # Mixed precision for VRAM efficiency on RTX 3060
    use_amp: bool = True
    # Gradient accumulation to simulate larger batch
    gradient_accumulation_steps: int = 2  # effective batch = 32
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    # Random seed
    seed: int = 42


@dataclass(frozen=True)
class Config:
    """Master configuration — single import."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Singleton
cfg = Config()

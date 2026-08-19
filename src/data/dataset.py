"""Canonical NIH ChestX-ray14 data and preprocessing pipeline."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from configs.config import cfg


TABULAR_FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "A": ("Patient Age", "gender_encoded"),
    "B": ("Patient Age", "gender_encoded", "view_PA"),
    "C": ("Patient Age", "gender_encoded", "Follow-up #"),
    "D": tuple(cfg.data.encoded_tabular_features),
}


def build_image_index(image_dirs: Iterable[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for image_dir in image_dirs:
        if not Path(image_dir).exists():
            continue
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(".png"):
                if filename in index:
                    raise ValueError(f"Duplicate image filename across directories: {filename}")
                index[filename] = Path(image_dir) / filename
    return index


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"NIH metadata is missing required columns: {missing}")


def load_and_prepare_metadata(
    csv_path: Path,
    image_index: Optional[Dict[str, Path]] = None,
) -> pd.DataFrame:
    """Load, validate, and encode the four canonical metadata variables."""
    frame = pd.read_csv(csv_path)
    required = [
        "Image Index", "Finding Labels", "Patient ID", "Patient Age",
        "Patient Gender", "View Position", "Follow-up #",
    ]
    _require_columns(frame, required)
    if image_index is not None:
        frame = frame[frame["Image Index"].isin(image_index)].copy()
    else:
        frame = frame.copy()

    if frame[required].isna().any().any():
        bad = frame[required].columns[frame[required].isna().any()].tolist()
        raise ValueError(f"Missing canonical metadata values in columns: {bad}")

    frame["Patient Age"] = pd.to_numeric(frame["Patient Age"], errors="raise")
    frame["Follow-up #"] = pd.to_numeric(frame["Follow-up #"], errors="raise")
    frame["Patient ID"] = pd.to_numeric(frame["Patient ID"], errors="raise").astype(np.int64)

    numeric = frame[["Patient Age", "Follow-up #"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Age and Follow-up # must be finite")
    if (frame["Follow-up #"] < 0).any():
        raise ValueError("Follow-up # must be non-negative")

    gender_map = dict(cfg.data.gender_mapping)
    view_map = dict(cfg.data.view_mapping)
    unknown_gender = sorted(set(frame["Patient Gender"].unique()) - set(gender_map))
    unknown_view = sorted(set(frame["View Position"].unique()) - set(view_map))
    if unknown_gender:
        raise ValueError(f"Unsupported Patient Gender categories: {unknown_gender}")
    if unknown_view:
        raise ValueError(f"Unsupported View Position categories: {unknown_view}")

    frame["Patient Age"] = frame["Patient Age"].clip(
        cfg.data.age_clip_min, cfg.data.age_clip_max
    ).astype(np.float32)
    frame["Follow-up #"] = frame["Follow-up #"].astype(np.float32)
    frame["gender_encoded"] = frame["Patient Gender"].map(gender_map).astype(np.float32)
    frame["view_PA"] = frame["View Position"].map(view_map).astype(np.float32)
    frame["binary_label"] = (
        frame["Finding Labels"].astype(str).str.strip() != "No Finding"
    ).astype(np.int64)

    if frame["Image Index"].duplicated().any():
        duplicates = frame.loc[frame["Image Index"].duplicated(), "Image Index"].head().tolist()
        raise ValueError(f"Duplicate Image Index values: {duplicates}")
    return frame.reset_index(drop=True)


def load_official_partitions(
    frame: pd.DataFrame,
    train_list_path: Path,
    test_list_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_images = set(pd.read_csv(train_list_path, header=None)[0].astype(str))
    test_images = set(pd.read_csv(test_list_path, header=None)[0].astype(str))
    overlap = train_images & test_images
    if overlap:
        raise ValueError(f"Official NIH lists overlap for {len(overlap)} images")
    train = frame[frame["Image Index"].isin(train_images)].copy().reset_index(drop=True)
    test = frame[frame["Image Index"].isin(test_images)].copy().reset_index(drop=True)
    train_patients = set(train["Patient ID"])
    test_patients = set(test["Patient ID"])
    patient_overlap = train_patients & test_patients
    if patient_overlap:
        raise ValueError(f"Official partitions overlap for {len(patient_overlap)} patients")
    return train, test


def load_official_training_pool(
    frame: pd.DataFrame,
    train_list_path: Path,
) -> pd.DataFrame:
    """Load C1-C6 data without opening or parsing the official test list."""
    train_images = set(pd.read_csv(train_list_path, header=None)[0].astype(str))
    training = frame[frame["Image Index"].isin(train_images)].copy().reset_index(drop=True)
    if training.empty:
        raise ValueError("Official NIH training pool is empty")
    missing = train_images - set(training["Image Index"].astype(str))
    if missing:
        first = sorted(missing)[0]
        raise FileNotFoundError(
            f"Training manifest references {len(missing)} unavailable images; first={first}"
        )
    return training


def get_transforms(is_training: bool = True) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if is_training:
        return transforms.Compose([
            transforms.Resize(
                (cfg.data.train_resize, cfg.data.train_resize),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomCrop((cfg.data.image_size, cfg.data.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=10,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(
            (cfg.data.image_size, cfg.data.image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        normalize,
    ])


class NIHChestXrayDataset(Dataset):
    """Modality-aware dataset; S1 does not perform image I/O."""

    TABULAR_COLS_BASELINE = list(TABULAR_FEATURE_SETS["D"])
    TABULAR_COLS = TABULAR_COLS_BASELINE

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_index: Optional[Dict[str, Path]] = None,
        transform: Optional[transforms.Compose] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        tabular_cols: Optional[Sequence[str]] = None,
        modalities: Sequence[str] = ("image", "tabular"),
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_index = image_index or {}
        self.transform = transform
        self.modalities = frozenset(modalities)
        self.tabular_cols = tuple(tabular_cols or self.TABULAR_COLS_BASELINE)
        unknown_modalities = self.modalities - {"image", "tabular"}
        if unknown_modalities:
            raise ValueError(f"Unknown modalities: {sorted(unknown_modalities)}")

        self.scaler: Optional[StandardScaler] = None
        self.tabular_array: Optional[np.ndarray] = None
        if "tabular" in self.modalities:
            _require_columns(self.df, self.tabular_cols)
            values = self.df[list(self.tabular_cols)].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Tabular tensor contains non-finite values")
            if fit_scaler and scaler is not None:
                raise ValueError("Pass either fit_scaler=True or an existing scaler, not both")
            self.scaler = StandardScaler().fit(values) if fit_scaler else scaler
            if self.scaler is None:
                raise ValueError("Canonical tabular input requires a fold-specific StandardScaler")
            self.tabular_array = self.scaler.transform(values).astype(np.float32)

        if "image" in self.modalities:
            missing = sorted(set(self.df["Image Index"]) - set(self.image_index))
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} image files; first={missing[0]}")
            if self.transform is None:
                raise ValueError("Image modality requires an explicit transform")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        item: Dict[str, object] = {
            "label_binary": torch.tensor(float(row["binary_label"]), dtype=torch.float32),
            "image_index": str(row["Image Index"]),
            "patient_id": int(row["Patient ID"]),
        }
        if "image" in self.modalities:
            with Image.open(self.image_index[str(row["Image Index"])]) as image:
                item["image"] = self.transform(image.convert("RGB"))
        if "tabular" in self.modalities:
            assert self.tabular_array is not None
            item["tabular"] = torch.from_numpy(self.tabular_array[index])
        return item


def compute_pos_weight(frame: pd.DataFrame) -> torch.Tensor:
    positive = int(frame["binary_label"].sum())
    negative = int(len(frame) - positive)
    if positive == 0 or negative == 0:
        raise ValueError("Training fold must contain both classes")
    return torch.tensor([negative / positive], dtype=torch.float32)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_fold_dataloaders(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    image_index: Dict[str, Path],
    *,
    modalities: Sequence[str],
    feature_set: str = "D",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Optional[StandardScaler], torch.Tensor]:
    if feature_set not in TABULAR_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    columns = TABULAR_FEATURE_SETS[feature_set]
    uses_tabular = "tabular" in modalities
    train_ds = NIHChestXrayDataset(
        train_frame,
        image_index=image_index,
        transform=get_transforms(True) if "image" in modalities else None,
        fit_scaler=uses_tabular,
        tabular_cols=columns,
        modalities=modalities,
    )
    validation_ds = NIHChestXrayDataset(
        validation_frame,
        image_index=image_index,
        transform=get_transforms(False) if "image" in modalities else None,
        scaler=train_ds.scaler if uses_tabular else None,
        tabular_cols=columns,
        modalities=modalities,
    )
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1_000_000)
    common = {
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory,
        # Workers are recreated each epoch and deterministically seeded from
        # the DataLoader generator. Persistent worker RNG cannot be restored
        # exactly after a process restart.
        "persistent_workers": False,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        drop_last=False,
        generator=validation_generator,
        **common,
    )
    return train_loader, validation_loader, train_ds.scaler, compute_pos_weight(train_frame)


def raw_semantic_tabular_frame(frame: pd.DataFrame, feature_set: str = "D") -> pd.DataFrame:
    """Dataframe contract for RealMLP/TabM model-specific preprocessing."""
    if feature_set not in TABULAR_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    reverse = {
        "gender_encoded": "Patient Gender",
        "view_PA": "View Position",
    }
    columns = [reverse.get(name, name) for name in TABULAR_FEATURE_SETS[feature_set]]
    result = frame[columns].copy()
    for categorical in ("Patient Gender", "View Position"):
        if categorical in result:
            result[categorical] = result[categorical].astype("category")
    return result


def patient_level_split(*args, **kwargs):
    raise RuntimeError(
        "Legacy random patient_level_split is disabled for canonical execution. "
        "Use immutable folds.csv or deployment_split.csv."
    )


def create_dataloaders(*args, **kwargs):
    raise RuntimeError(
        "Legacy train/validation/test loader is disabled. Official test access is guarded until C7."
    )

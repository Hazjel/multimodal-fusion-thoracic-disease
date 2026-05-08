"""
NIH Chest X-ray14 Dataset with multimodal support.
Handles image loading (across 12 subdirs), tabular feature extraction,
multi-label encoding, and patient-level splitting.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import cfg


def build_image_index(image_dirs: List[Path]) -> Dict[str, Path]:
    """
    Build a lookup dict: filename -> full path across all 12 image subdirs.
    O(n) scan done once at initialization, O(1) lookup thereafter.
    """
    index = {}
    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for fname in os.listdir(img_dir):
            if fname.endswith(".png"):
                index[fname] = img_dir / fname
    return index


def load_and_prepare_metadata(
    csv_path: Path,
    image_index: Dict[str, Path],
) -> pd.DataFrame:
    """
    Load NIH metadata CSV, clean outliers, encode features,
    and filter to images that actually exist on disk.
    """
    df = pd.read_csv(csv_path)

    # Filter to images available on disk
    df = df[df["Image Index"].isin(image_index)].reset_index(drop=True)

    # Clean age outliers — physiologically impossible values
    df["Patient Age"] = df["Patient Age"].clip(
        cfg.data.age_clip_min, cfg.data.age_clip_max
    )

    # Encode multi-label targets: "Atelectasis|Effusion" → [1,0,...,1,...,0]
    label_names = list(cfg.data.label_names)
    for label in label_names:
        df[label] = df["Finding Labels"].apply(
            lambda x, lbl=label: 1.0 if lbl in x.split("|") else 0.0
        )

    # Binary label: 0=No Finding, 1=Any pathology
    df["binary_label"] = (
        df["Finding Labels"].apply(lambda x: 0.0 if x.strip() == "No Finding" else 1.0)
    )

    # Encode categorical tabular features
    df["gender_encoded"] = (df["Patient Gender"] == "M").astype(float)
    df["view_PA"] = (df["View Position"] == "PA").astype(float)

    return df


def patient_level_split(
    df: pd.DataFrame,
    train_list_path: Path,
    test_list_path: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level split using official NIH train/test lists.
    Validation set carved from train set at patient level to prevent data leakage.
    """
    train_images = set(
        pd.read_csv(train_list_path, header=None)[0].tolist()
    )
    test_images = set(
        pd.read_csv(test_list_path, header=None)[0].tolist()
    )

    df_train_full = df[df["Image Index"].isin(train_images)].copy()
    df_test = df[df["Image Index"].isin(test_images)].copy()

    # Split train → train + val at PATIENT level
    rng = np.random.RandomState(seed)
    all_patients = df_train_full["Patient ID"].unique()
    rng.shuffle(all_patients)
    n_val_patients = max(1, int(len(all_patients) * val_ratio))

    val_patients = set(all_patients[:n_val_patients])
    train_patients = set(all_patients[n_val_patients:])

    df_train = df_train_full[
        df_train_full["Patient ID"].isin(train_patients)
    ].reset_index(drop=True)
    df_val = df_train_full[
        df_train_full["Patient ID"].isin(val_patients)
    ].reset_index(drop=True)

    return df_train, df_val, df_test


class NIHChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH Chest X-ray14 with multimodal support.

    Returns:
        image: (3, 224, 224) tensor
        tabular: (num_tabular_features,) tensor
        label_binary: scalar tensor (0 or 1)
        label_multilabel: (14,) tensor
    """

    TABULAR_COLS = ["Patient Age", "gender_encoded", "view_PA", "Follow-up #"]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_index: Dict[str, Path],
        transform: Optional[transforms.Compose] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform
        self.label_names = list(cfg.data.label_names)

        # Fit or apply StandardScaler for tabular features
        tabular_data = self.df[self.TABULAR_COLS].values.astype(np.float32)
        if fit_scaler:
            self.scaler = StandardScaler().fit(tabular_data)
        else:
            self.scaler = scaler

        if self.scaler is not None:
            self.tabular_array = self.scaler.transform(tabular_data).astype(np.float32)
        else:
            self.tabular_array = tabular_data

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Image loading — grayscale X-ray → 3-channel for pretrained backbone
        img_path = self.image_index[row["Image Index"]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tabular features (already scaled)
        tabular = torch.tensor(self.tabular_array[idx], dtype=torch.float32)

        # Labels
        binary_label = torch.tensor(row["binary_label"], dtype=torch.float32)
        multilabel = torch.tensor(
            row[self.label_names].values.astype(np.float32),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "tabular": tabular,
            "label_binary": binary_label,
            "label_multilabel": multilabel,
        }


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Image preprocessing pipeline.
    Training: augmentation for robustness.
    Validation/Test: deterministic resize + normalize.
    """
    img_size = cfg.data.image_size
    # ImageNet normalization — standard for pretrained backbones
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def compute_pos_weights(df: pd.DataFrame, label_names: List[str]) -> torch.Tensor:
    """
    Compute positive class weights for weighted BCE loss.
    Formula: (num_negative / num_positive) per label.
    Handles extreme imbalance (e.g., Hernia at 0.2%).
    """
    labels = df[label_names].values
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts

    # Clip to avoid division by zero, apply sqrt to dampen extreme weights
    pos_weights = np.sqrt(neg_counts / np.maximum(pos_counts, 1.0))
    return torch.tensor(pos_weights, dtype=torch.float32)


def create_dataloaders(
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, torch.Tensor]:
    """
    Full pipeline: load data → split → create DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, scaler, pos_weights
    """
    if batch_size is None:
        batch_size = cfg.train.batch_size

    # Build image path index
    image_index = build_image_index(cfg.paths.image_dirs)
    print(f"[Data] Indexed {len(image_index)} images across {len(cfg.paths.image_dirs)} directories")

    # Load and prepare metadata
    df = load_and_prepare_metadata(cfg.paths.csv_path, image_index)
    print(f"[Data] Loaded metadata: {len(df)} samples, {df['Patient ID'].nunique()} patients")

    # Patient-level split
    df_train, df_val, df_test = patient_level_split(
        df,
        cfg.paths.train_list_path,
        cfg.paths.test_list_path,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.train.seed,
    )
    print(f"[Data] Split — Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Compute class weights from training set
    pos_weights = compute_pos_weights(df_train, list(cfg.data.label_names))

    # Create datasets with appropriate transforms
    train_ds = NIHChestXrayDataset(
        df_train, image_index,
        transform=get_transforms(is_training=True),
        fit_scaler=True,
    )
    val_ds = NIHChestXrayDataset(
        df_val, image_index,
        transform=get_transforms(is_training=False),
        scaler=train_ds.scaler,
    )
    test_ds = NIHChestXrayDataset(
        df_test, image_index,
        transform=get_transforms(is_training=False),
        scaler=train_ds.scaler,
    )

    # DataLoaders
    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, train_ds.scaler, pos_weights

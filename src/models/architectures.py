"""
Model architectures for three scenarios (S1-S3), all binary classification.

S1: TabularMLP       — MLP on patient metadata
S2: ImageCNN         — DenseNet-121 on chest X-ray
S3: MultimodalFusion — Intermediate fusion (tabular + image)

All share the same branch architectures for fair comparison.
"""
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import cfg


class TabularBranch(nn.Module):
    """
    MLP encoder for tabular patient metadata.
    Input: (batch, num_features) → Output: (batch, tabular_feature_dim)
    """

    def __init__(
        self,
        input_dim: int = cfg.model.tabular_input_dim,
        hidden_dims: tuple = cfg.model.tabular_hidden_dims,
        output_dim: int = cfg.model.tabular_feature_dim,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ImageBranch(nn.Module):
    """
    DenseNet-121 backbone for chest X-ray feature extraction.
    Input: (batch, 3, 224, 224) → Output: (batch, image_feature_dim)

    Supports two pretrained weight sources:
    - ImageNet (default): general visual features
    - CheXNet: DenseNet-121 pretrained on NIH ChestX-ray14 (domain-specific)
    """

    def __init__(
        self,
        output_dim: int = cfg.model.image_feature_dim,
        pretrained: bool = cfg.model.pretrained,
        freeze_ratio: float = cfg.model.freeze_backbone_ratio,
        dropout: float = cfg.model.dropout_rate,
        use_chexnet: bool = cfg.model.use_chexnet,
        chexnet_path: str = cfg.model.chexnet_weights_path,
    ):
        super().__init__()
        # CheXNet weights loaded manually — skip ImageNet init to avoid wasted download
        imagenet_weights = models.DenseNet121_Weights.IMAGENET1K_V1 if (pretrained and not use_chexnet) else None
        backbone = models.densenet121(weights=imagenet_weights)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if use_chexnet:
            self._load_chexnet_weights(chexnet_path)

        # Freeze early layers for stable fine-tuning
        all_params = list(self.features.parameters())
        freeze_count = int(len(all_params) * freeze_ratio)
        for param in all_params[:freeze_count]:
            param.requires_grad = False

        # Projection head: 1024 → output_dim
        self.projection = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _load_chexnet_weights(self, path: str) -> None:
        """Load DenseNet-121 feature weights from a CheXNet checkpoint."""
        import os
        if not os.path.exists(path):
            print(f"[ImageBranch] CheXNet weights not found at {path} — falling back to random init")
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw = checkpoint.get("state_dict", checkpoint)

        # Supports arnoweng/CheXNet format: "module.densenet121.features.*"
        # and plain formats: "densenet121.features.*" or "features.*"
        prefixes = [
            "module.densenet121.features.",
            "densenet121.features.",
            "features.",
        ]
        mapped = {}
        for k, v in raw.items():
            for prefix in prefixes:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    # Fix older torchvision naming: "norm.1" → "norm1", "conv.1" → "conv1"
                    new_key = new_key.replace("norm.1", "norm1").replace("norm.2", "norm2")
                    new_key = new_key.replace("conv.1", "conv1").replace("conv.2", "conv2")
                    mapped[new_key] = v
                    break

        if not mapped:
            print("[ImageBranch] CheXNet: could not map any weights — check checkpoint format")
            return

        missing, unexpected = self.features.load_state_dict(mapped, strict=False)
        if missing or unexpected:
            print(f"[ImageBranch] CheXNet weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
        else:
            print("[ImageBranch] CheXNet weights loaded — all keys matched perfectly")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = nn.functional.relu(features, inplace=True)
        pooled = self.pool(features).flatten(1)  # (batch, 1024)
        return self.projection(pooled)


# ======================== SCENARIO MODELS ========================


class TabularMLP(nn.Module):
    """Scenario S1: Tabular-only binary classification (Normal vs Abnormal)."""

    def __init__(self, num_classes: int = cfg.data.num_classes):
        super().__init__()
        self.branch = TabularBranch()
        self.classifier = nn.Linear(cfg.model.tabular_feature_dim, num_classes)

    def forward(self, tabular: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.branch(tabular)
        return self.classifier(features)  # raw logits — BCE applies sigmoid


class ImageCNN(nn.Module):
    """Scenario S2: Image-only binary classification using DenseNet-121."""

    def __init__(self, num_classes: int = cfg.data.num_classes):
        super().__init__()
        self.branch = ImageBranch()
        self.classifier = nn.Linear(cfg.model.image_feature_dim, num_classes)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.branch(image)
        return self.classifier(features)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        return self.branch.features.denseblock4


class MultimodalFusion(nn.Module):
    """
    Scenario S3: Intermediate Fusion — binary classification.
    Dual-branch (tabular MLP + image CNN) → concatenate → shared classifier.
    """

    def __init__(self, num_classes: int = cfg.data.num_classes):
        super().__init__()
        self.tabular_branch = TabularBranch()
        self.image_branch = ImageBranch()

        fused_dim = cfg.model.tabular_feature_dim + cfg.model.image_feature_dim
        fusion_layers = []
        prev_dim = fused_dim
        for h_dim in cfg.model.fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.model.dropout_rate),
            ])
            prev_dim = h_dim

        self.fusion = nn.Sequential(*fusion_layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        f_img = self.image_branch(image)      # (batch, 512)
        f_tab = self.tabular_branch(tabular)   # (batch, 128)
        fused = torch.cat([f_img, f_tab], dim=1)  # (batch, 640)
        fused = self.fusion(fused)
        return self.classifier(fused)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        return self.image_branch.features.denseblock4


def build_model(
    scenario: str,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """
    Factory function — construct model by scenario name.

    Args:
        scenario: "S1", "S2", or "S3" (all binary)
        num_classes: override default (default 1 for binary)
    """
    if num_classes is None:
        num_classes = 1

    model_map = {
        "S1": lambda: TabularMLP(num_classes=num_classes),
        "S2": lambda: ImageCNN(num_classes=num_classes),
        "S3": lambda: MultimodalFusion(num_classes=num_classes),
    }

    if scenario not in model_map:
        raise ValueError(f"Unknown scenario: {scenario}. Use S1/S2/S3.")

    model = model_map[scenario]()

    # Log parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {scenario} — Total: {total:,} params | Trainable: {trainable:,} params")

    return model

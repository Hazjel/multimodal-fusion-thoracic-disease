"""
Model architectures for all 4 scenarios.

S1: TabularMLP       — MLP on patient metadata
S2: ImageCNN         — DenseNet-121 on chest X-ray
S3: MultimodalFusion — Intermediate fusion, binary output
S4: MultimodalFusion — Intermediate fusion, multi-label output

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
        # Final projection to feature space
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ImageBranch(nn.Module):
    """
    DenseNet-121 backbone for chest X-ray feature extraction.
    Input: (batch, 3, 224, 224) → Output: (batch, image_feature_dim)

    Freezes first N% of backbone for transfer learning stability.
    """

    def __init__(
        self,
        output_dim: int = cfg.model.image_feature_dim,
        pretrained: bool = cfg.model.pretrained,
        freeze_ratio: float = cfg.model.freeze_backbone_ratio,
        dropout: float = cfg.model.dropout_rate,
    ):
        super().__init__()
        # Load pretrained DenseNet-121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # DenseNet-121 feature extractor: all layers except final classifier
        self.features = backbone.features
        # DenseNet-121 features output: (batch, 1024, 7, 7) for 224x224 input
        self.pool = nn.AdaptiveAvgPool2d(1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = nn.functional.relu(features, inplace=True)
        pooled = self.pool(features).flatten(1)  # (batch, 1024)
        return self.projection(pooled)


# ======================== SCENARIO MODELS ========================


class TabularMLP(nn.Module):
    """
    Scenario S1: Tabular-only classification.
    Supports both binary and multi-label output.
    """

    def __init__(self, num_classes: int = cfg.data.num_classes):
        super().__init__()
        self.branch = TabularBranch()
        self.classifier = nn.Linear(cfg.model.tabular_feature_dim, num_classes)

    def forward(self, tabular: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.branch(tabular)
        return self.classifier(features)  # raw logits — BCE applies sigmoid


class ImageCNN(nn.Module):
    """
    Scenario S2: Image-only classification using DenseNet-121.
    Supports both binary and multi-label output.
    """

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
    Scenario S3/S4: Intermediate Fusion.
    Dual-branch (tabular MLP + image CNN) → concatenate → shared classifier.

    S3: num_classes=1 for binary
    S4: num_classes=14 for multi-label
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
        scenario: "S1", "S2", "S3", or "S4"
        num_classes: override default. S3=1 (binary), S4=14 (multi-label)
    """
    if num_classes is None:
        num_classes = 1 if scenario == "S3" else cfg.data.num_classes

    model_map = {
        "S1": lambda: TabularMLP(num_classes=num_classes),
        "S2": lambda: ImageCNN(num_classes=num_classes),
        "S3": lambda: MultimodalFusion(num_classes=num_classes),
        "S4": lambda: MultimodalFusion(num_classes=num_classes),
    }

    if scenario not in model_map:
        raise ValueError(f"Unknown scenario: {scenario}. Use S1/S2/S3/S4.")

    model = model_map[scenario]()

    # Log parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {scenario} — Total: {total:,} params | Trainable: {trainable:,} params")

    return model

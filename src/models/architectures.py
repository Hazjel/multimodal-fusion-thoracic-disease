"""Canonical S1/S2/S3 architectures for protocol v1.0.0.

Legacy attention, gated fusion, and ad-hoc freeze-ratio models are purposely
not exposed by the canonical factory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from configs.config import cfg
from src.protocol.contracts import file_sha256, state_dict_sha256


BACKBONE_NATIVE_DIMS = {
    "densenet121": 1024,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
}


def _seed(value: int) -> None:
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class TabularBranch(nn.Module):
    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.model.tabular_dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.model.tabular_dropout),
        )

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        return self.encoder(tabular)


class ImageEncoder(nn.Module):
    """Architecture-specific structural fine-tuning with frozen BN buffers."""

    def __init__(
        self,
        backbone_name: str,
        *,
        pretraining: str = "imagenet",
        chexnet_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        if backbone_name not in BACKBONE_NATIVE_DIMS:
            raise ValueError(f"Unsupported canonical backbone: {backbone_name}")
        if pretraining not in {"imagenet", "chexnet", "none"}:
            raise ValueError(f"Unsupported pretraining: {pretraining}")
        if pretraining == "chexnet" and backbone_name != "densenet121":
            raise ValueError("CheXNet pretraining is only valid for DenseNet-121")

        self.backbone_name = backbone_name
        self.pretraining = pretraining
        self.weight_identifier = "NONE"
        self.weight_file_checksum: Optional[str] = None
        self.features = self._create_features(backbone_name, pretraining)
        self.pool = nn.AdaptiveAvgPool2d(1)

        if pretraining == "chexnet":
            path = Path(chexnet_path or cfg.model.chexnet_weights_path)
            self._load_chexnet(path)
            self.weight_identifier = "CheXNet_DenseNet121_NIH"
            self.weight_file_checksum = file_sha256(path)

        self.pretrained_state_checksum = state_dict_sha256(self.features)
        self._freeze_structural_stages()
        self.projection = nn.Sequential(
            nn.Linear(BACKBONE_NATIVE_DIMS[backbone_name], cfg.model.image_feature_dim),
            nn.BatchNorm1d(cfg.model.image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.model.image_dropout),
        )
        self.enforce_frozen_batchnorm_eval()

    def _create_features(self, name: str, pretraining: str) -> nn.Module:
        use_imagenet = pretraining == "imagenet"
        if name == "densenet121":
            weight = models.DenseNet121_Weights.IMAGENET1K_V1 if use_imagenet else None
            self.weight_identifier = "DenseNet121_Weights.IMAGENET1K_V1" if weight else "NONE"
            return models.densenet121(weights=weight).features
        if name == "resnet50":
            weight = models.ResNet50_Weights.IMAGENET1K_V2 if use_imagenet else None
            self.weight_identifier = "ResNet50_Weights.IMAGENET1K_V2" if weight else "NONE"
            backbone = models.resnet50(weights=weight)
            return nn.Sequential(*list(backbone.children())[:-2])
        weight = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_imagenet else None
        self.weight_identifier = "EfficientNet_B0_Weights.IMAGENET1K_V1" if weight else "NONE"
        return models.efficientnet_b0(weights=weight).features

    def _load_chexnet(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"CheXNet checkpoint not found: {path}")
        actual_checksum = file_sha256(path)
        if actual_checksum != cfg.model.chexnet_expected_sha256:
            raise RuntimeError(
                "CheXNet checkpoint checksum mismatch: "
                f"expected={cfg.model.chexnet_expected_sha256}, actual={actual_checksum}"
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw = checkpoint.get("state_dict", checkpoint)
        prefixes = (
            "module.densenet121.features.",
            "densenet121.features.",
            "features.",
        )
        mapped = {}
        for key, value in raw.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    new_key = new_key.replace("norm.1", "norm1").replace("norm.2", "norm2")
                    new_key = new_key.replace("conv.1", "conv1").replace("conv.2", "conv2")
                    mapped[new_key] = value
                    break
        if not mapped:
            raise RuntimeError("CheXNet checkpoint contained no mappable feature weights")
        missing, unexpected = self.features.load_state_dict(mapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "CheXNet feature mapping is incomplete: "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )

    def frozen_stage_modules(self) -> List[nn.Module]:
        if self.backbone_name == "densenet121":
            names = (
                "conv0", "norm0", "relu0", "pool0", "denseblock1", "transition1",
                "denseblock2", "transition2", "denseblock3", "transition3",
            )
            return [getattr(self.features, name) for name in names]
        if self.backbone_name == "resnet50":
            return [self.features[index] for index in range(7)]
        return [self.features[index] for index in range(6)]

    def trainable_stage_modules(self) -> List[nn.Module]:
        if self.backbone_name == "densenet121":
            return [self.features.denseblock4, self.features.norm5]
        if self.backbone_name == "resnet50":
            return [self.features[7]]
        return [self.features[index] for index in range(6, 9)]

    def _freeze_structural_stages(self) -> None:
        for module in self.frozen_stage_modules():
            for parameter in module.parameters():
                parameter.requires_grad = False
        for module in self.trainable_stage_modules():
            for parameter in module.parameters():
                parameter.requires_grad = True

    def enforce_frozen_batchnorm_eval(self) -> None:
        for stage in self.frozen_stage_modules():
            for module in stage.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def frozen_batchnorm_modules(self) -> List[nn.Module]:
        result: List[nn.Module] = []
        for stage in self.frozen_stage_modules():
            result.extend(
                module for module in stage.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            )
        return result

    def trainable_batchnorm_modules(self) -> List[nn.Module]:
        result: List[nn.Module] = []
        for stage in self.trainable_stage_modules():
            result.extend(
                module for module in stage.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            )
        result.extend(
            module for module in self.projection.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        )
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.enforce_frozen_batchnorm_eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(image)
        if self.backbone_name == "densenet121":
            feature_map = F.relu(feature_map, inplace=False)
        pooled = self.pool(feature_map).flatten(1)
        return self.projection(pooled)

    def gradcam_target_layer(self) -> nn.Module:
        if self.backbone_name == "densenet121":
            return self.features.denseblock4
        if self.backbone_name == "resnet50":
            return self.features[7]
        return self.features[7]


class TabularMLP(nn.Module):
    def __init__(self, num_classes: int = 1, input_dim: int = 4) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("Canonical task has exactly one binary logit")
        self.branch = TabularBranch(input_dim=input_dim)
        self.classifier = nn.Linear(cfg.model.tabular_feature_dim, 1)

    def forward(self, tabular: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.branch(tabular))


class ImageCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        *,
        backbone_name: str = "densenet121",
        pretraining: str = "imagenet",
        fold: int = 0,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("Canonical task has exactly one binary logit")
        _seed(cfg.train.seed + int(fold))
        self.branch = ImageEncoder(backbone_name, pretraining=pretraining)
        self.classifier = nn.Linear(cfg.model.image_feature_dim, 1)

    def forward(self, image: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.branch(image))

    def get_cam_target_layer(self) -> nn.Module:
        return self.branch.gradcam_target_layer()


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        *,
        backbone_name: str = "densenet121",
        pretraining: str = "imagenet",
        fold: int = 0,
        tabular_input_dim: int = 4,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("Canonical task has exactly one binary logit")
        _seed(cfg.train.seed + int(fold))
        self.image_branch = ImageEncoder(backbone_name, pretraining=pretraining)
        _seed(cfg.train.seed + 1000 + int(fold))
        self.tabular_branch = TabularBranch(input_dim=tabular_input_dim)
        _seed(cfg.train.seed + 2000 + int(fold))
        self.fusion = nn.Sequential(
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.model.fusion_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.model.fusion_dropout),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.image_branch(image)
        tabular_features = self.tabular_branch(tabular)
        return self.classifier(self.fusion(torch.cat([image_features, tabular_features], dim=1)))

    def get_cam_target_layer(self) -> nn.Module:
        return self.image_branch.gradcam_target_layer()


# Backward-compatible canonical name; unlike the former class it never uses a freeze ratio.
ImageBranch = ImageEncoder


def image_initial_hashes(model: nn.Module) -> Tuple[str, str]:
    branch = model.branch if isinstance(model, ImageCNN) else model.image_branch
    return state_dict_sha256(branch.features), state_dict_sha256(branch.projection)


def build_s2_s3_pair(
    *,
    backbone_name: str,
    fold: int,
    pretraining: str = "imagenet",
    tabular_input_dim: int = 4,
) -> Tuple[ImageCNN, MultimodalFusion]:
    s2 = ImageCNN(backbone_name=backbone_name, pretraining=pretraining, fold=fold)
    s3 = MultimodalFusion(
        backbone_name=backbone_name,
        pretraining=pretraining,
        fold=fold,
        tabular_input_dim=tabular_input_dim,
    )
    if image_initial_hashes(s2) != image_initial_hashes(s3):
        raise AssertionError("S2/S3 image backbone or projection initialization differs")
    return s2, s3


def build_model(
    scenario: str,
    num_classes: Optional[int] = None,
    *,
    backbone_name: str = "densenet121",
    pretraining: str = "imagenet",
    fold: int = 0,
    tabular_input_dim: int = 4,
) -> nn.Module:
    num_classes = 1 if num_classes is None else num_classes
    if scenario == "S1":
        _seed(cfg.train.seed + 1000 + int(fold))
        model: nn.Module = TabularMLP(num_classes=num_classes, input_dim=tabular_input_dim)
    elif scenario == "S2":
        model = ImageCNN(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretraining=pretraining,
            fold=fold,
        )
    elif scenario == "S3":
        model = MultimodalFusion(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretraining=pretraining,
            fold=fold,
            tabular_input_dim=tabular_input_dim,
        )
    else:
        raise ValueError("Canonical factory only allows S1, S2, or S3")
    return model

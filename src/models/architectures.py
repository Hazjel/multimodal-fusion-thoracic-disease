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


class EfficientImageBranch(nn.Module):
    """
    EfficientNet-B0 backbone variant for the image branch — exploratory
    alternative to DenseNet-121 (see ImageBranch).

    Motivation: a search of Scopus-indexed literature on NIH ChestX-ray14
    found EfficientNet is a common competitive backbone (e.g. Ucan et al.
    2025, PeerJ CS: EfficientNetB7+CoordinateAttention, AUC 0.8309 on
    multi-label). Unlike ImageBranch, no domain-specific pretrained weights
    (CheXNet) exist for EfficientNet — only ImageNet pretrain is available,
    so this is not an apples-to-apples comparison with S2/S2-attn (both use
    CheXNet). Included for completeness after the user explicitly requested
    testing this backbone.

    Input: (batch, 3, 224, 224) -> Output: (batch, image_feature_dim)
    """

    def __init__(
        self,
        output_dim: int = cfg.model.image_feature_dim,
        dropout: float = cfg.model.dropout_rate,
        freeze_ratio: float = cfg.model.freeze_backbone_ratio,
    ):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze early layers for stable fine-tuning (same ratio as ImageBranch)
        all_params = list(self.features.parameters())
        freeze_count = int(len(all_params) * freeze_ratio)
        for param in all_params[:freeze_count]:
            param.requires_grad = False

        # EfficientNet-B0 outputs 1280-dim features after GAP (vs 1024 for DenseNet-121)
        self.projection = nn.Sequential(
            nn.Linear(1280, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)  # (batch, 1280)
        return self.projection(pooled)


class ResNetImageBranch(nn.Module):
    """
    ResNet-50 backbone variant for the image branch — exploratory
    alternative to DenseNet-121 (see ImageBranch) and EfficientNet-B0 (see
    EfficientImageBranch).

    Motivation: same literature search that motivated EfficientImageBranch
    also found ResNet-50 is the backbone used in the most topically similar
    paper (Tang et al. 2025, JMIR Med Inform — ResNet50 + tabular clinical
    data, AUC 0.975 on osteoporosis fusion) and that attention modules give
    their largest gains specifically on ResNet backbones (Imran et al. 2026,
    Scientific Reports: Attention-ResNet101 AUC 0.853->0.872, vs
    EfficientNet-B0/B3 "minimal change"). Tested here without attention
    first for an apples-to-apples comparison against S2-eff (both ImageNet
    pretrain, no attention).

    Same caveat as EfficientImageBranch: no CheXNet-equivalent pretrained
    weights exist for ResNet-50 — only ImageNet, so this is not directly
    comparable to S2/S2-attn.

    Input: (batch, 3, 224, 224) -> Output: (batch, image_feature_dim)
    """

    def __init__(
        self,
        output_dim: int = cfg.model.image_feature_dim,
        dropout: float = cfg.model.dropout_rate,
        freeze_ratio: float = cfg.model.freeze_backbone_ratio,
    ):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Keep everything up to (and including) layer4, drop avgpool+fc
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        all_params = list(self.features.parameters())
        freeze_count = int(len(all_params) * freeze_ratio)
        for param in all_params[:freeze_count]:
            param.requires_grad = False

        # ResNet-50 outputs 2048-dim features after GAP (vs 1024 DenseNet-121, 1280 EfficientNet-B0)
        self.projection = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)  # (batch, 2048)
        return self.projection(pooled)


class ChannelSpatialAttention(nn.Module):
    """
    Two-part attention applied to the DenseNet-121 feature map before GAP,
    following A3Net's channel-wise + element-wise (spatial) attention design
    (Wang et al. 2020, "Triple attention learning for classification of 14
    thoracic diseases", Medical Image Analysis — DOI: 10.1016/j.media.2020.101846).
    Scale-wise attention (A3Net's third module, for multi-resolution feature
    pyramids) is omitted here since this branch operates on a single-scale
    7x7x1024 feature map, not a multi-scale pyramid.

    Motivation: a search of recent literature on NIH ChestX-ray14 found
    attention modules give the largest AUC gains on DenseNet/ResNet backbones
    (e.g. A3Net: DenseNet-121 + triple attention, per-class AUC 0.826) while
    EfficientNet shows "minimal change, diminishing returns" (Imran et al.
    2026, Scientific Reports) — so attention is added here to DenseNet-121
    rather than by switching backbones.

    channel attention (SE-style): learns per-channel importance weights from
        global context, re-weights which of the 1024 channels matter most.
    spatial attention: learns a per-pixel weight map from channel-pooled
        statistics, re-weights which image regions matter most — this is the
        part expected to concentrate activation on pathology-relevant areas,
        similar in spirit to what Grad-CAM visualizes post-hoc.
    """

    def __init__(self, channels: int = 1024, reduction: int = 16):
        super().__init__()
        # Channel attention: squeeze-excite over global average pooled features
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        # Spatial attention: 7x7 conv over channel-pooled (avg+max) maps
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ch_weights = self.channel_gate(x)              # (batch, C, 1, 1)
        x = x * ch_weights

        # Spatial attention
        avg_map = torch.mean(x, dim=1, keepdim=True)    # (batch, 1, H, W)
        max_map, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, H, W)
        sp_weights = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        x = x * sp_weights

        return x


class ImageBranchAttn(ImageBranch):
    """
    Exploratory variant of ImageBranch with channel+spatial attention
    inserted between the DenseNet-121 feature extractor and GAP.
    See ChannelSpatialAttention docstring for rationale.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = ChannelSpatialAttention(channels=1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = nn.functional.relu(features, inplace=True)
        features = self.attention(features)
        pooled = self.pool(features).flatten(1)
        return self.projection(pooled)


# ======================== SCENARIO MODELS ========================


class TabularMLP(nn.Module):
    """
    Scenario S1: Tabular-only binary classification (Normal vs Abnormal).

    input_dim override: pass 6 to use the extended feature set (baseline 4 +
    visit_count + pixel_spacing_x, see NIHChestXrayDataset.TABULAR_COLS_EXTENDED)
    for the S1-extended feature-engineering exploration.
    """

    def __init__(
        self,
        num_classes: int = cfg.data.num_classes,
        input_dim: int = cfg.model.tabular_input_dim,
    ):
        super().__init__()
        self.branch = TabularBranch(input_dim=input_dim)
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


class ImageCNNAttn(nn.Module):
    """
    Scenario S2-attn: exploratory — DenseNet-121 + channel/spatial attention
    (A3Net-style, see ChannelSpatialAttention). Same task as S2, isolates
    the effect of attention on the image branch alone before considering it
    for the fusion scenario.
    """

    def __init__(self, num_classes: int = cfg.data.num_classes):
        super().__init__()
        self.branch = ImageBranchAttn()
        self.classifier = nn.Linear(cfg.model.image_feature_dim, num_classes)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.branch(image)
        return self.classifier(features)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        return self.branch.features.denseblock4


class ImageCNNEfficientNet(nn.Module):
    """
    Scenario S2-eff: exploratory — EfficientNet-B0 backbone (ImageNet
    pretrain only, no CheXNet equivalent exists). See EfficientImageBranch
    docstring for rationale and the caveat that this is not directly
    comparable to S2/S2-attn (CheXNet-pretrained).

    freeze_ratio override: first S2-eff run used the same 75% freeze ratio
    as the CheXNet-pretrained branches. Since EfficientNet starts from
    generic ImageNet (not domain-adapted), it may need more layers unfrozen
    to re-learn chest X-ray-specific features — pass a lower value (e.g.
    0.3-0.5) for the S2-eff-tuned follow-up run.
    """

    def __init__(
        self,
        num_classes: int = cfg.data.num_classes,
        freeze_ratio: float = cfg.model.freeze_backbone_ratio,
    ):
        super().__init__()
        self.branch = EfficientImageBranch(freeze_ratio=freeze_ratio)
        self.classifier = nn.Linear(cfg.model.image_feature_dim, num_classes)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.branch(image)
        return self.classifier(features)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization (last conv block before final 1x1 projection)."""
        return self.branch.features[7]


class ImageCNNResNet(nn.Module):
    """
    Scenario S2-resnet: exploratory — ResNet-50 backbone (ImageNet pretrain
    only, no CheXNet equivalent exists). See ResNetImageBranch docstring for
    rationale. Same task as S2/S2-eff, isolates whether ResNet-50 fares
    better than EfficientNet-B0 when both use generic ImageNet pretrain.
    """

    def __init__(
        self,
        num_classes: int = cfg.data.num_classes,
        freeze_ratio: float = cfg.model.freeze_backbone_ratio,
    ):
        super().__init__()
        self.branch = ResNetImageBranch(freeze_ratio=freeze_ratio)
        self.classifier = nn.Linear(cfg.model.image_feature_dim, num_classes)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.branch(image)
        return self.classifier(features)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization (layer4, ResNet-50's last residual block)."""
        return self.branch.features[-1]


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


class GatedMultimodalFusion(nn.Module):
    """
    Scenario S3-gated: Gated Fusion variant — exploratory alternative to
    plain concatenation fusion.

    Motivation: DeLong test on S3 (concat fusion) vs S2 (image-only) showed
    no significant AUC difference (p=0.49, see results/tables/stats_significance.json).
    Tabular metadata is only 4 raw features vs. 1024-dim image representation,
    so plain concatenation may let uninformative tabular signal dilute the
    fused representation on a per-sample basis.

    Gated fusion learns a per-sample scalar gate g in [0,1] from both branches
    that weights the tabular contribution, instead of feeding it in at a fixed
    ratio for every sample:

        gate = sigmoid(Linear([f_img ; f_tab]))
        fused = concat([f_img, gate * f_tab])

    This lets the model down-weight tabular features when they are
    uninformative for a given sample, while still allowing them to contribute
    when useful — expected to behave at least as well as image-only in the
    worst case, unlike unconditional concatenation.
    """

    def __init__(self, num_classes: int = cfg.data.num_classes):
        super().__init__()
        self.tabular_branch = TabularBranch()
        self.image_branch = ImageBranch()

        img_dim = cfg.model.image_feature_dim
        tab_dim = cfg.model.tabular_feature_dim

        self.gate = nn.Sequential(
            nn.Linear(img_dim + tab_dim, tab_dim),
            nn.ReLU(inplace=True),
            nn.Linear(tab_dim, 1),
            nn.Sigmoid(),
        )

        fused_dim = img_dim + tab_dim
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

        gate_input = torch.cat([f_img, f_tab], dim=1)
        g = self.gate(gate_input)              # (batch, 1), in [0, 1]

        fused = torch.cat([f_img, g * f_tab], dim=1)  # (batch, 640)
        fused = self.fusion(fused)
        return self.classifier(fused)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        return self.image_branch.features.denseblock4

    def get_last_gate_values(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """Inspect the learned gate values for a batch (for analysis/XAI)."""
        with torch.no_grad():
            f_img = self.image_branch(image)
            f_tab = self.tabular_branch(tabular)
            gate_input = torch.cat([f_img, f_tab], dim=1)
            return self.gate(gate_input).squeeze(-1)  # (batch,)


class CrossAttentionFusion(nn.Module):
    """
    Scenario S3-attn: Cross-Attention Fusion variant — exploratory alternative
    to plain concatenation and gated fusion.

    Motivation: neither concat fusion (S3, p=0.65 vs S2) nor gated fusion
    (S3-gated, p=0.996 vs S3) produced a significant AUC improvement over
    image-only (see results/tables/stats_significance.json). Both are
    "static" combination rules applied uniformly. Cross-attention lets each
    modality's representation be refined by attending to the other — treating
    image and tabular features as a 2-token sequence and applying standard
    multi-head attention, so the model can learn richer, sample-dependent
    interactions (not just a scalar gate) before fusion.

    Architecture:
        f_img (512-dim), f_tab (128-dim) -> projected to a shared d_model
        -> stacked as a 2-token sequence [img_token, tab_token]
        -> nn.MultiheadAttention (self-attention over the 2 tokens, each
           token attends to itself and the other modality)
        -> flatten both refined tokens -> concat -> same fusion MLP as S3

    This keeps the rest of the pipeline (fusion_hidden_dims, classifier,
    Grad-CAM target layer) identical to S3 for a controlled comparison.
    """

    def __init__(
        self,
        num_classes: int = cfg.data.num_classes,
        d_model: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        self.tabular_branch = TabularBranch()
        self.image_branch = ImageBranch()

        img_dim = cfg.model.image_feature_dim
        tab_dim = cfg.model.tabular_feature_dim

        # Project each modality into a shared attention space
        self.img_proj = nn.Linear(img_dim, d_model)
        self.tab_proj = nn.Linear(tab_dim, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        fused_dim = d_model * 2  # concat of both attended tokens
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

        t_img = self.img_proj(f_img).unsqueeze(1)  # (batch, 1, d_model)
        t_tab = self.tab_proj(f_tab).unsqueeze(1)  # (batch, 1, d_model)
        tokens = torch.cat([t_img, t_tab], dim=1)  # (batch, 2, d_model)

        attended, _ = self.attn(tokens, tokens, tokens)  # self-attn over 2 tokens
        attended = self.attn_norm(attended + tokens)     # residual + norm

        fused = attended.flatten(1)  # (batch, 2*d_model)
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
        scenario: "S1", "S2", "S2-attn", "S2-eff", "S2-eff-tuned", "S3", "S3-gated",
            or "S3-attn" (all binary)
        num_classes: override default (default 1 for binary)
    """
    if num_classes is None:
        num_classes = 1

    model_map = {
        "S1": lambda: TabularMLP(num_classes=num_classes),
        "S2": lambda: ImageCNN(num_classes=num_classes),
        "S2-attn": lambda: ImageCNNAttn(num_classes=num_classes),
        "S2-eff": lambda: ImageCNNEfficientNet(num_classes=num_classes),
        # Lower freeze ratio: EfficientNet starts from generic ImageNet (not
        # CheXNet), may need more unfrozen layers to adapt to chest X-rays.
        "S2-eff-tuned": lambda: ImageCNNEfficientNet(num_classes=num_classes, freeze_ratio=0.4),
        "S2-resnet": lambda: ImageCNNResNet(num_classes=num_classes),
        "S3": lambda: MultimodalFusion(num_classes=num_classes),
        "S3-gated": lambda: GatedMultimodalFusion(num_classes=num_classes),
        "S3-attn": lambda: CrossAttentionFusion(num_classes=num_classes),
    }

    if scenario not in model_map:
        raise ValueError(f"Unknown scenario: {scenario}. Use S1/S2/S2-attn/S2-eff/S2-eff-tuned/S3/S3-gated/S3-attn.")

    model = model_map[scenario]()

    # Log parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {scenario} — Total: {total:,} params | Trainable: {trainable:,} params")

    return model

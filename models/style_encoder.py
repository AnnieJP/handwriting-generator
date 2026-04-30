"""
Style Encoder — CNN that extracts a compact style embedding from one or more
reference handwriting images.

Architecture: ResNet18 backbone with a custom projection head.
The encoder aggregates features across multiple reference images by
average-pooling the per-image embeddings (set-aggregation).

Output: a fixed-size style vector used to condition both the GAN generator
and the diffusion U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class ResNetStyleEncoder(nn.Module):
    """
    Style encoder based on a modified ResNet18.

    The network takes a batch of reference handwriting images (each from
    the same writer) and returns a single style embedding by mean-pooling
    over the per-image features.

    Args:
        style_dim: Dimensionality of the output style embedding.
        pretrained: Whether to initialise ResNet18 with ImageNet weights.
        freeze_backbone: Freeze ResNet parameters (fine-tune only the head).
    """

    def __init__(
        self,
        style_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, style_dim),
            nn.LayerNorm(style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim),
        )
        self.style_dim = style_dim

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image or a batch of images.

        Args:
            x: Tensor of shape (B, 1, H, W).

        Returns:
            Style embedding of shape (B, style_dim).
        """
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a set of reference images and aggregate into one style vector.

        Args:
            x: Tensor of shape (B, C, 1, H, W)  — batch of B writers,
               each with C reference images. Alternatively accepts
               (N, 1, H, W) and returns (N, style_dim).

        Returns:
            Style embedding of shape (B, style_dim) or (N, style_dim).
        """
        if x.ndim == 5:
            B, C, _, H, W = x.shape
            x_flat = x.view(B * C, 1, H, W)
            emb = self.encode_single(x_flat)
            emb = emb.view(B, C, self.style_dim).mean(dim=1)
            return emb
        return self.encode_single(x)


class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalisation (AdaIN).
    Modulates feature maps using per-sample style statistics.

        AdaIN(x, s) = σ(s) * ((x - μ(x)) / σ(x)) + μ(s)

    where μ and σ are produced by a linear projection of the style vector.
    """

    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_linear = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        params = self.style_linear(style)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # changed by Nani - applying normalization because the style vector is not guaranteed to have zero mean and unit variance
        return (1 + gamma) * self.norm(x) + beta
        # return gamma * self.norm(x) + beta


class StyleConditionedBlock(nn.Module):
    """
    Residual block with AdaIN style conditioning.
    Used in the GAN generator and optionally in the diffusion U-Net.
    """

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.adain1 = AdaptiveInstanceNorm(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.adain2 = AdaptiveInstanceNorm(channels, style_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.adain1(x, style)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.adain2(x, style)
        x = F.relu(x + residual, inplace=True)
        return x

"""
ScrabbleGAN-inspired Handwriting GAN
=====================================
Reference: Fogel et al., "ScrabbleGAN: Semi-Supervised Varying Length
           Handwritten Text Generation", CVPR 2020.

Architecture overview
---------------------
Generator
  - Character embedding + noise (z) per character → (B, C_emb+z_dim, 1, T)
  - BiLSTM temporal model → (B, hidden, 1, T)
  - Series of upsampling transpose-conv blocks (×2 per block) to reach
    target image height, with AdaIN style conditioning.
  - Outputs a grayscale image of shape (B, 1, img_height, T*char_width).

Discriminator (PatchGAN + auxiliary CTC text recogniser)
  - Real/fake PatchGAN head.
  - CTC-based OCR head for text auxiliary loss.

Training loss:
    L_GAN = λ1 * L1 + λ2 * L_perc + λ3 * L_adv    (generator)
    L_D   = L_hinge(real) + L_hinge(fake)           (discriminator)
    L_rec = L_CTC(ocr_pred, gt_text)                (shared recogniser)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from models.style_encoder import ResNetStyleEncoder, AdaptiveInstanceNorm
from utils.dataset import NUM_CLASSES, CHARSET


# ---------------------------------------------------------------------------
# Helper blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class UpsampleBlock(nn.Module):
    """2× upsample (height only) + optional AdaIN style conditioning."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        style_dim: int,
        upsample_h: bool = True,
        upsample_w: bool = True,
    ):
        super().__init__()
        stride = (2 if upsample_h else 1, 2 if upsample_w else 1)
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=stride, padding=1
        )
        self.adain = AdaptiveInstanceNorm(out_ch, style_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.adain(x, style)
        return self.act(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Variable-length handwriting image generator.

    Takes a character sequence and a style embedding and produces a
    handwritten image whose width scales with text length.

    Args:
        vocab_size: Number of characters (including blank).
        z_dim: Per-character noise dimension.
        style_dim: Style embedding dimension (from StyleEncoder).
        hidden_dim: BiLSTM hidden state size.
        img_height: Target image height (must be divisible by 2^n_up_h).
        char_width: Width of each character "slot" in the initial spatial map.
        n_up_h: Number of height upsampling blocks (2^n_up_h = img_height / 4).
        base_channels: Base channel count for convolution blocks.
    """

    def __init__(
        self,
        vocab_size: int = NUM_CLASSES,
        z_dim: int = 64,
        style_dim: int = 256,
        hidden_dim: int = 128,
        img_height: int = 64,
        char_width: int = 16,
        n_up_h: int = 4,
        base_channels: int = 256,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.char_width = char_width
        self.img_height = img_height
        self.n_up_h = n_up_h
        self.init_height = img_height // (2 ** n_up_h)  # e.g. 64/16 = 4
        assert self.init_height >= 1, "img_height must be >= 2^n_up_h"

        self.char_embed = nn.Embedding(vocab_size, 64)

        in_dim = 64 + z_dim
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        lstm_out = hidden_dim * 2

        self.init_conv = nn.Conv2d(lstm_out, base_channels, kernel_size=1)

        up_blocks = []
        ch = base_channels
        for i in range(n_up_h):
            out_ch = max(ch // 2, 64)
            up_blocks.append(UpsampleBlock(ch, out_ch, style_dim, upsample_h=True, upsample_w=False))
            ch = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out_conv = nn.Sequential(
            nn.Conv2d(ch, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        char_indices: torch.Tensor,
        style: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            char_indices: (B, T) long tensor of character indices.
            style: (B, style_dim) style embedding.
            z: Optional (B, T, z_dim) noise. Sampled from N(0,1) if None.

        Returns:
            Generated image (B, 1, img_height, T * char_width).
        """
        B, T = char_indices.shape
        device = char_indices.device

        emb = self.char_embed(char_indices)  # (B, T, 64)
        if z is None:
            z = torch.randn(B, T, self.z_dim, device=device)
        x = torch.cat([emb, z], dim=-1)  # (B, T, 64+z_dim)

        feat, _ = self.lstm(x)  # (B, T, hidden*2)

        feat = feat.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, T)
        feat = self.init_conv(feat)                 # (B, base_ch, 1, T)

        feat = feat.repeat(1, 1, self.init_height, self.char_width)  # (B, base_ch, init_h, T*cw)

        for block in self.up_blocks:
            feat = block(feat, style)

        return self.out_conv(feat)


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN + CTC recogniser)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for handwriting realism.
    Outputs a spatial map of real/fake scores.
    """

    def __init__(self, in_channels: int = 1, n_layers: int = 4, ndf: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = ndf
        for _ in range(1, n_layers):
            layers += [
                nn.Conv2d(ch, min(ch * 2, 512), 4, stride=2, padding=1),
                nn.BatchNorm2d(min(ch * 2, 512)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = min(ch * 2, 512)
        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CTCRecogniser(nn.Module):
    """
    Lightweight CTC-based text recogniser used as the auxiliary discriminator head.
    Takes the discriminator's intermediate feature map and outputs per-frame
    character probabilities for CTC decoding.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Feature map (B, C, H, W).

        Returns:
            Log-probabilities (T, B, num_classes) for CTC.
        """
        x = self.cnn(feat).squeeze(2)  # (B, 256, W)
        x = x.permute(0, 2, 1)         # (B, W, 256)
        x, _ = self.lstm(x)            # (B, W, 256)
        x = self.fc(x)                  # (B, W, num_classes)
        x = x.permute(1, 0, 2)          # (W, B, num_classes)
        return F.log_softmax(x, dim=-1)


class Discriminator(nn.Module):
    """
    Combined discriminator: PatchGAN real/fake head + CTC text recogniser.

    Args:
        ndf: Base feature channels.
        n_layers: Number of downsampling layers.
        num_classes: Number of character classes for CTC (including blank).
    """

    def __init__(
        self,
        ndf: int = 64,
        n_layers: int = 4,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(1, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = ndf
        for _ in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, next_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        self.shared = nn.Sequential(*layers)
        self.adv_head = nn.Conv2d(ch, 1, 4, stride=1, padding=1)
        self.ocr_head = CTCRecogniser(in_channels=ch, num_classes=num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            adv_logits: (B, 1, H', W') patch real/fake scores.
            ocr_logits: (T, B, num_classes) log-probs for CTC.
        """
        feat = self.shared(x)
        adv = self.adv_head(feat)
        ocr = self.ocr_head(feat)
        return adv, ocr


# ---------------------------------------------------------------------------
# Full ScrabbleGAN model wrapper
# ---------------------------------------------------------------------------

class ScrabbleGAN(nn.Module):
    """
    Top-level ScrabbleGAN wrapper combining:
      - Style encoder
      - Generator
      - Discriminator

    Args:
        style_dim: Style embedding dimensionality.
        z_dim: Per-character noise dimensionality.
        img_height: Target image height.
        char_width: Width of each character slot.
        n_up_h: Height upsampling steps in generator.
        pretrained_style_enc: Use ImageNet-pretrained style encoder.
    """

    def __init__(
        self,
        style_dim: int = 256,
        z_dim: int = 64,
        img_height: int = 64,
        char_width: int = 16,
        n_up_h: int = 4,
        pretrained_style_enc: bool = True,
    ):
        super().__init__()
        self.style_encoder = ResNetStyleEncoder(
            style_dim=style_dim, pretrained=pretrained_style_enc
        )
        self.generator = Generator(
            z_dim=z_dim,
            style_dim=style_dim,
            img_height=img_height,
            char_width=char_width,
            n_up_h=n_up_h,
        )
        self.discriminator = Discriminator()

    def generate(
        self,
        char_indices: torch.Tensor,
        style_refs: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate a handwritten image.

        Args:
            char_indices: (B, T) character indices.
            style_refs: (B, 1, H, W) or (B, K, 1, H, W) reference images.
            z: Optional noise.

        Returns:
            Generated image (B, 1, H, W').
        """
        style = self.style_encoder(style_refs)
        return self.generator(char_indices, style, z)

    def discriminate(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.discriminator(img)

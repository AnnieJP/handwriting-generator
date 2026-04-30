"""
Diffusion-based Handwriting Generator
=======================================
Inspired by:
  - One-DM: "One-Shot Diffusion Mimicker for Handwritten Text Generation", ECCV 2024.
  - DiffusionPen: "Towards Controlling the Style of Handwritten Text Generation", ECCV 2024.
  - DDPM: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.

Architecture overview
----------------------
HandwritingDiffusion
  ├── TextEncoder       — character embedding + Transformer → text context
  ├── StyleEncoder      — ResNet18 → style embedding (reused from models/style_encoder.py)
  └── UNet              — denoising U-Net with cross-attention for text+style conditioning

Noise schedule: linear beta schedule (Ho et al.)

Training:
  1. Sample x0 (real image), t ~ Uniform(1, T), ε ~ N(0, I).
  2. Compute x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε.
  3. Predict ε_θ(x_t, t, text_ctx, style_emb).
  4. Loss = L_denoise + λ_style * L_style + λ_text * L_text.

Sampling (DDPM reverse process):
  x_T ~ N(0,I) → iterative denoising → x_0.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.style_encoder import ResNetStyleEncoder
from utils.dataset import NUM_CLASSES


# ---------------------------------------------------------------------------
# Noise schedule utilities
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def precompute_schedule(betas: torch.Tensor):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Encodes a character sequence into a text context tensor.

    Architecture: learnable character embedding → Transformer encoder.
    Output: (B, T, text_dim) sequence or (B, text_dim) pooled summary.
    """

    def __init__(
        self,
        vocab_size: int = NUM_CLASSES,
        emb_dim: int = 128,
        text_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        max_len: int = 256,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(emb_dim, text_dim)
        self.text_dim = text_dim

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, T) character indices.
            mask:   (B, T) boolean mask (True = padding).

        Returns:
            seq_ctx: (B, T, text_dim) — per-token context.
            pooled:  (B, text_dim)    — mean-pooled context vector.
        """
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_embed(pos)
        x = self.transformer(x, src_key_padding_mask=mask)
        seq_ctx = self.proj(x)
        pooled = seq_ctx.mean(dim=1)
        return seq_ctx, pooled


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with time and style+text conditioning via FiLM."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim, out_ch * 2)
        )
        self.cond_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, out_ch * 2)
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(x)
        # Time conditioning
        t_params = self.time_mlp(t_emb)[:, :, None, None]
        ts, tb = t_params.chunk(2, dim=1)
        h = h * (1 + ts) + tb
        # Style+text conditioning
        c_params = self.cond_mlp(cond)[:, :, None, None]
        cs, cb = c_params.chunk(2, dim=1)
        h = h * (1 + cs) + cb
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Cross-attention between spatial features (query) and text context (key/value)."""

    def __init__(self, query_dim: int, context_dim: int, n_heads: int = 4):
        super().__init__()
        head_dim = query_dim // n_heads
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.out = nn.Linear(query_dim, query_dim)
        self.norm = nn.GroupNorm(min(8, query_dim), query_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h_flat = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        Q = self.to_q(h_flat)
        K = self.to_k(context)
        V = self.to_v(context)

        def split_heads(t):
            tb, tl, td = t.shape
            return t.view(tb, tl, self.n_heads, td // self.n_heads).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, H * W, C)
        out = self.out(out).permute(0, 2, 1).view(B, C, H, W)
        return x + out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim, use_attn, context_dim):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_dim, cond_dim)
        self.attn = (
            CrossAttention(out_ch, context_dim) if use_attn else nn.Identity()
        )
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb, cond, context=None):
        x = self.res(x, t_emb, cond)
        if context is not None and not isinstance(self.attn, nn.Identity):
            x = self.attn(x, context)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, cond_dim, use_attn, context_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, time_dim, cond_dim)
        self.attn = (
            CrossAttention(out_ch, context_dim) if use_attn else nn.Identity()
        )

    def forward(self, x, skip, t_emb, cond, context=None):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb, cond)
        if context is not None and not isinstance(self.attn, nn.Identity):
            x = self.attn(x, context)
        return x


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Conditional denoising U-Net for handwriting generation.

    Conditioning:
      - Timestep t  → sinusoidal embedding → MLP → t_emb
      - Style vector (ResNetStyleEncoder output) + text pooled vector
        → concatenated → cond vector (FiLM conditioning on every ResBlock)
      - Text sequence context → cross-attention at selected scales

    Args:
        in_channels: Image channels (1 for grayscale).
        base_channels: Channel count at first resolution.
        channel_mults: Channel multipliers per resolution level.
        time_dim: Sinusoidal embedding dimension.
        style_dim: Style embedding dimension.
        text_dim: Text context dimension.
        use_attn_at: Indices of resolution levels where cross-attention is used.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        time_dim: int = 256,
        style_dim: int = 256,
        text_dim: int = 256,
        use_attn_at: Tuple[int, ...] = (2, 3),
    ):
        super().__init__()
        cond_dim = style_dim + text_dim
        context_dim = text_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        in_ch = base_channels
        skip_ch_list = []
        for i, out_ch in enumerate(channels):
            use_attn = i in use_attn_at
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_dim, cond_dim, use_attn, context_dim)
            )
            skip_ch_list.append(out_ch)
            in_ch = out_ch

        self.mid_res1 = ResidualBlock(in_ch, in_ch, time_dim, cond_dim)
        self.mid_attn = CrossAttention(in_ch, context_dim)
        self.mid_res2 = ResidualBlock(in_ch, in_ch, time_dim, cond_dim)

        """
        for i, out_ch in enumerate(reversed(channels[:-1])):
            skip_ch = skip_ch_list[-(i + 1)]
            use_attn = (len(channels) - 2 - i) in use_attn_at
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, time_dim, cond_dim, use_attn, context_dim)
            )
            in_ch = out_ch

        """
        # changed by Nani - The last up block does not have a skip connection, so we use skip_ch=0 and concatenate a zero tensor
        for i, out_ch in enumerate(reversed(channels)):
            skip_ch = skip_ch_list[-(i + 1)]
            use_attn = (len(channels) - 1 - i) in use_attn_at
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, time_dim, cond_dim, use_attn, context_dim)
            )
            in_ch = out_ch

        self.final_res = ResidualBlock(in_ch, base_channels, time_dim, cond_dim)   

        """
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, 3, padding=1),
        )
        """
        # chnaged by Nani - changed the output conv to have a fixed in_ch
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        text_pooled: torch.Tensor,
        text_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:           (B, 1, H, W) noisy image x_t.
            t:           (B,) timestep indices.
            style:       (B, style_dim) style embedding.
            text_pooled: (B, text_dim) pooled text representation.
            text_ctx:    (B, T, text_dim) per-token text context for cross-attn.

        Returns:
            Predicted noise ε_θ of shape (B, 1, H, W).
        """
        t_emb = self.time_mlp(t)
        cond = torch.cat([style, text_pooled], dim=-1)

        x = self.init_conv(x)
        skips = []
        for block in self.down_blocks:
            x, skip = block(x, t_emb, cond, text_ctx)
            skips.append(skip)

        x = self.mid_res1(x, t_emb, cond)
        if text_ctx is not None:
            x = self.mid_attn(x, text_ctx)
        x = self.mid_res2(x, t_emb, cond)

        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip, t_emb, cond, text_ctx)

        # return self.out_conv(x)
        # changed by Nani
        x = self.final_res(x, t_emb, cond)
        return self.out_conv(x)


# ---------------------------------------------------------------------------
# DDPM wrapper
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Wraps the UNet and handles:
      - Forward diffusion (q): adds noise at timestep t.
      - Reverse diffusion (p): iterative denoising.

    Args:
        unet: UNet model.
        timesteps: Number of diffusion steps.
        schedule: 'linear' or 'cosine' beta schedule.
    """

    def __init__(
        self,
        unet: UNet,
        timesteps: int = 1000,
        schedule: str = "linear",
    ):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        sched = precompute_schedule(betas)
        for k, v in sched.items():
            self.register_buffer(k, v)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: x_t = √ᾱ_t * x0 + √(1-ᾱ_t) * ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_a * x0 + sqrt_1a * noise, noise

    def p_losses(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        text_pooled: torch.Tensor,
        text_ctx: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute denoising loss.

        Returns:
            (predicted_noise, target_noise) for external loss computation.
        """
        x_noisy, target_noise = self.q_sample(x0, t, noise)
        pred_noise = self.unet(x_noisy, t, style, text_pooled, text_ctx)
        return pred_noise, target_noise

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        style: torch.Tensor,
        text_pooled: torch.Tensor,
        text_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One step of the reverse process."""
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        pred_noise = self.unet(x, t_batch, style, text_pooled, text_ctx)

        alpha = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        beta = self.betas[t]

        x_prev = (
            (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise)
            / torch.sqrt(alpha)
        )
        if t > 0:
            noise = torch.randn_like(x)
            x_prev += torch.sqrt(self.posterior_variance[t]) * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        style: torch.Tensor,
        text_pooled: torch.Tensor,
        text_ctx: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Full reverse diffusion: x_T → x_0."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, style, text_pooled, text_ctx)
        return x


# ---------------------------------------------------------------------------
# Full HandwritingDiffusion model
# ---------------------------------------------------------------------------

class HandwritingDiffusion(nn.Module):
    """
    End-to-end handwriting generation via diffusion.

    Combines:
      - TextEncoder (character sequence → text context)
      - ResNetStyleEncoder (reference images → style embedding)
      - DDPM (U-Net-based denoising diffusion model)

    Args:
        style_dim: Style embedding dimension.
        text_dim: Text context / embedding dimension.
        timesteps: Number of diffusion steps.
        img_height: Height of generated images.
        base_channels: U-Net base channel count.
        schedule: 'linear' or 'cosine' beta schedule.
        pretrained_style_enc: Use ImageNet-pretrained style encoder.
    """

    def __init__(
        self,
        style_dim: int = 256,
        text_dim: int = 256,
        timesteps: int = 1000,
        img_height: int = 64,
        base_channels: int = 64,
        schedule: str = "linear",
        pretrained_style_enc: bool = True,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(text_dim=text_dim)
        self.style_encoder = ResNetStyleEncoder(
            style_dim=style_dim, pretrained=pretrained_style_enc
        )
        unet = UNet(
            in_channels=1,
            base_channels=base_channels,
            style_dim=style_dim,
            text_dim=text_dim,
        )
        self.ddpm = DDPM(unet, timesteps=timesteps, schedule=schedule)
        self.style_dim = style_dim
        self.text_dim = text_dim

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        char_tokens: torch.Tensor,
        style_refs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Args:
            x0:          (B, 1, H, W) clean image.
            t:           (B,) timestep indices.
            char_tokens: (B, T) character indices.
            style_refs:  (B, 1, H, W) or (B, K, 1, H, W) reference images.

        Returns:
            (predicted_noise, target_noise)
        """
        text_ctx, text_pooled = self.text_encoder(char_tokens)
        style = self.style_encoder(style_refs)
        return self.ddpm.p_losses(x0, t, style, text_pooled, text_ctx)

    @torch.no_grad()
    def generate(
        self,
        char_tokens: torch.Tensor,
        style_refs: torch.Tensor,
        img_height: int = 64,
        img_width: int = 256,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate a handwritten image from text tokens and style references.

        Args:
            char_tokens: (B, T) character indices.
            style_refs:  (B, 1, H, W) or (B, K, 1, H, W) reference images.
            img_height:  Height of the output image.
            img_width:   Width of the output image.
            device:      Target device.

        Returns:
            Generated image (B, 1, img_height, img_width) in [-1, 1].
        """
        self.eval()
        text_ctx, text_pooled = self.text_encoder(char_tokens.to(device))
        style = self.style_encoder(style_refs.to(device))
        B = char_tokens.shape[0]
        shape = (B, 1, img_height, img_width)
        return self.ddpm.sample(shape, style, text_pooled, text_ctx, device=device)

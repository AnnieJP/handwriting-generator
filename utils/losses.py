"""
Loss functions for both the GAN and diffusion model training.

GAN losses:
    L_GAN = λ1 * L1 + λ2 * L_perc + λ3 * L_adv

Diffusion losses:
    L_diff = L_denoise + λ_style * L_style + λ_text * L_text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional


# ---------------------------------------------------------------------------
# Perceptual Loss (VGG-based)
# ---------------------------------------------------------------------------

class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from a pretrained VGG19 network
    for computing perceptual loss.
    """

    def __init__(self, layers: List[int] = None):
        super().__init__()
        if layers is None:
            layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.slices = nn.ModuleList()
        prev = 0
        for layer_idx in layers:
            self.slices.append(nn.Sequential(*list(vgg.features[prev:layer_idx])))
            prev = layer_idx
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = []
        for s in self.slices:
            x = s(x)
            features.append(x)
        return features


class PerceptualLoss(nn.Module):
    """
    VGG perceptual loss: MSE between feature maps of generated and real images.
    """

    def __init__(self, layers: List[int] = None, weights: List[float] = None):
        super().__init__()
        self.vgg = VGGFeatureExtractor(layers)
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0, 1.0]

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        gen_feats = self.vgg(generated)
        real_feats = self.vgg(real)
        loss = torch.tensor(0.0, device=generated.device)
        for w, gf, rf in zip(self.weights, gen_feats, real_feats):
            loss = loss + w * F.mse_loss(gf, rf.detach())
        return loss


# ---------------------------------------------------------------------------
# Style Loss (Gram Matrix)
# ---------------------------------------------------------------------------

class StyleLoss(nn.Module):
    """
    Gram matrix style loss — measures texture/style similarity between images.
    """

    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor(layers=[3, 8, 15])

    @staticmethod
    def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
        return gram

    def forward(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        gen_feats = self.vgg(generated)
        ref_feats = self.vgg(reference)
        loss = torch.tensor(0.0, device=generated.device)
        for gf, rf in zip(gen_feats, ref_feats):
            loss = loss + F.mse_loss(self.gram_matrix(gf), self.gram_matrix(rf.detach()))
        return loss


# ---------------------------------------------------------------------------
# GAN Adversarial Loss (Hinge)
# ---------------------------------------------------------------------------

class HingeLoss(nn.Module):
    """
    Hinge loss for GAN training.
    """

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()

    def forward(self, logits: torch.Tensor, target_real: bool) -> torch.Tensor:
        if target_real:
            return F.relu(1.0 - logits).mean()
        else:
            return F.relu(1.0 + logits).mean()


class LSGANLoss(nn.Module):
    """
    Least-squares GAN loss (Mao et al., 2017).
    """

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * (F.mse_loss(real_logits, torch.ones_like(real_logits))
                      + F.mse_loss(fake_logits, torch.zeros_like(fake_logits)))

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


# ---------------------------------------------------------------------------
# CTC Loss wrapper
# ---------------------------------------------------------------------------

class CTCRecognitionLoss(nn.Module):
    """
    CTC loss for text recognizer training (auxiliary discriminator head).
    """

    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)

    def forward(
        self,
        log_probs: torch.Tensor,   # (T, N, C) log-softmax output
        targets: torch.Tensor,     # (sum of target lengths,) flattened
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


# ---------------------------------------------------------------------------
# GAN Total Loss
# ---------------------------------------------------------------------------

class GANTotalLoss(nn.Module):
    """
    Combined GAN generator loss:
        L_GAN = λ1 * L1 + λ2 * L_perc + λ3 * L_adv

    Args:
        lambda1: Weight for pixel-level L1 loss.
        lambda2: Weight for perceptual loss.
        lambda3: Weight for adversarial loss.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.1,
        lambda3: float = 1.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        self.adversarial = HingeLoss()

    def forward(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        l1_loss = self.l1(generated, real)
        perc_loss = self.perceptual(generated, real)
        adv_loss = self.adversarial.generator_loss(fake_logits)

        total = (
            self.lambda1 * l1_loss
            + self.lambda2 * perc_loss
            + self.lambda3 * adv_loss
        )
        return total, {
            "l1": l1_loss.item(),
            "perceptual": perc_loss.item(),
            "adversarial": adv_loss.item(),
        }


# ---------------------------------------------------------------------------
# Diffusion Losses
# ---------------------------------------------------------------------------

class DenoisingLoss(nn.Module):
    """
    Simple MSE denoising loss for DDPM:
        L_denoise = E[|| ε - ε_θ(x_t, t) ||²]
    """

    def forward(
        self, predicted_noise: torch.Tensor, target_noise: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(predicted_noise, target_noise)


class TextConsistencyLoss(nn.Module):
    """
    CTC-based text preservation loss for diffusion model.
    Ensures the generated image is readable by an OCR model.
    """

    def __init__(self, num_classes: int, blank: int = 0):
        super().__init__()
        self.ctc = CTCRecognitionLoss(blank=blank)
        self.num_classes = num_classes

    def forward(
        self,
        ocr_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(ocr_logits, dim=-1)
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


class DiffusionTotalLoss(nn.Module):
    """
    Combined diffusion training loss:
        L_diff = L_denoise + λ_style * L_style + λ_text * L_text

    Args:
        lambda_style: Weight for style Gram-matrix loss.
        lambda_text: Weight for CTC text consistency loss.
        num_classes: Number of character classes for CTC (including blank).
    """

    def __init__(
        self,
        lambda_style: float = 0.5,
        lambda_text: float = 0.5,
        num_classes: int = 96,
    ):
        super().__init__()
        self.lambda_style = lambda_style
        self.lambda_text = lambda_text
        self.denoise = DenoisingLoss()
        self.style = StyleLoss()
        self.text = TextConsistencyLoss(num_classes)

    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        generated_img: Optional[torch.Tensor] = None,
        style_ref: Optional[torch.Tensor] = None,
        ocr_logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        denoise_loss = self.denoise(predicted_noise, target_noise)
        losses = {"denoise": denoise_loss.item()}
        total = denoise_loss

        if generated_img is not None and style_ref is not None:
            style_loss = self.style(generated_img, style_ref)
            total = total + self.lambda_style * style_loss
            losses["style"] = style_loss.item()

        if (ocr_logits is not None and targets is not None
                and input_lengths is not None and target_lengths is not None):
            text_loss = self.text(ocr_logits, targets, input_lengths, target_lengths)
            total = total + self.lambda_text * text_loss
            losses["text"] = text_loss.item()

        return total, losses



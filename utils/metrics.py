"""
Evaluation metrics for handwriting generation.

Metrics implemented:
  - FID  (Fréchet Inception Distance)        – visual realism
  - CER  (Character Error Rate)              – content correctness
  - WER  (Word Error Rate)                   – content correctness
  - SSIM (Structural Similarity Index)       – style similarity
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity as sk_ssim

try:
    import editdistance
    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False


# ---------------------------------------------------------------------------
# FID — Fréchet Inception Distance
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """
    Extracts 2048-d pool3 features from InceptionV3.
    Input images are expected as float tensors in [0, 1] range, 3-channel.
    """

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
        )
        self.layers = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.layers(x).squeeze(-1).squeeze(-1)


def _compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 32,
) -> float:
    """
    Compute FID between two sets of images.

    Args:
        real_images: Tensor of shape (N, C, H, W) in [-1, 1] range.
        fake_images: Tensor of shape (M, C, H, W) in [-1, 1] range.
        device: Device for InceptionV3 inference.
        batch_size: Batch size for feature extraction.

    Returns:
        FID score (lower is better).
    """
    model = InceptionFeatureExtractor().to(device).eval()

    def _extract(imgs: torch.Tensor) -> np.ndarray:
        imgs = (imgs + 1.0) / 2.0  # [-1,1] → [0,1]
        ds = TensorDataset(imgs)
        dl = DataLoader(ds, batch_size=batch_size)
        feats = []
        with torch.no_grad():
            for (batch,) in dl:
                feats.append(model(batch.to(device)).cpu().numpy())
        return np.concatenate(feats, axis=0)

    real_feats = _extract(real_images)
    fake_feats = _extract(fake_images)

    mu_r, sigma_r = _compute_statistics(real_feats)
    mu_f, sigma_f = _compute_statistics(fake_feats)

    return _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


# ---------------------------------------------------------------------------
# CER / WER
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if EDITDISTANCE_AVAILABLE:
        return editdistance.eval(a, b)
    # Fallback: simple DP
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(hypotheses: List[str], references: List[str]) -> float:
    """
    Character Error Rate.

    CER = (substitutions + insertions + deletions) / total reference characters.

    Args:
        hypotheses: List of predicted transcriptions (from OCR).
        references:  List of ground-truth transcriptions.

    Returns:
        Mean CER over all pairs (lower is better).
    """
    assert len(hypotheses) == len(references), "Length mismatch"
    total_dist, total_len = 0, 0
    for hyp, ref in zip(hypotheses, references):
        total_dist += _edit_distance(hyp, ref)
        total_len += len(ref)
    return total_dist / max(total_len, 1)


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    """
    Word Error Rate.

    WER = (substitutions + insertions + deletions) / total reference words.

    Args:
        hypotheses: List of predicted transcriptions.
        references:  List of ground-truth transcriptions.

    Returns:
        Mean WER over all pairs (lower is better).
    """
    assert len(hypotheses) == len(references), "Length mismatch"
    total_dist, total_len = 0, 0
    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.split()
        ref_words = ref.split()
        total_dist += _edit_distance(hyp_words, ref_words)
        total_len += len(ref_words)
    return total_dist / max(total_len, 1)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    data_range: float = 2.0,
) -> float:
    """
    Compute SSIM between two single images.

    Args:
        img_a: np.ndarray (H, W) or (H, W, C) in [-1, 1] range.
        img_b: np.ndarray with the same shape.
        data_range: Range of image values (2.0 for [-1,1]).

    Returns:
        SSIM value in [-1, 1] (higher is better).
    """
    if img_a.ndim == 3:
        return sk_ssim(img_a, img_b, data_range=data_range, channel_axis=-1)
    return sk_ssim(img_a, img_b, data_range=data_range)


def compute_ssim_batch(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    """
    Compute mean SSIM over a batch of generated vs. reference images.

    Args:
        generated: Tensor (N, C, H, W) in [-1, 1].
        reference:  Tensor (N, C, H, W) in [-1, 1].

    Returns:
        Mean SSIM (higher is better).
    """
    assert generated.shape == reference.shape, "Shape mismatch"
    gen_np = generated.detach().cpu().numpy()
    ref_np = reference.detach().cpu().numpy()
    scores = []
    for g, r in zip(gen_np, ref_np):
        g_hw = g.transpose(1, 2, 0) if g.shape[0] > 1 else g.squeeze(0)
        r_hw = r.transpose(1, 2, 0) if r.shape[0] > 1 else r.squeeze(0)
        scores.append(compute_ssim_pair(g_hw, r_hw))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------

def evaluate_all(
    generated_imgs: torch.Tensor,
    real_imgs: torch.Tensor,
    hypotheses: List[str],
    references: List[str],
    device: str = "cpu",
) -> dict:
    """
    Run FID, CER, WER, and SSIM and return a summary dict.
    """
    fid = compute_fid(real_imgs, generated_imgs, device=device)
    cer = compute_cer(hypotheses, references)
    wer = compute_wer(hypotheses, references)
    ssim = compute_ssim_batch(generated_imgs, real_imgs)
    return {
        "FID": round(fid, 4),
        "CER": round(cer, 4),
        "WER": round(wer, 4),
        "SSIM": round(ssim, 4),
    }

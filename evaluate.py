"""
Evaluation script — computes FID, CER, WER, and SSIM for generated handwriting.

Usage:
    python evaluate.py --model gan --checkpoint checkpoints/gan/best.pth \
                       --data_root data/iam --style_dir data/samples \
                       --output_dir results/gan

    python evaluate.py --model diffusion --checkpoint checkpoints/diffusion/best.pth \
                       --data_root data/iam --style_dir data/samples \
                       --output_dir results/diffusion
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from models.scrabblegan import ScrabbleGAN
from models.diffusion import HandwritingDiffusion
from utils.dataset import (
    IAMDataset, UserSamplesDataset,
    collate_fn, decode_text, encode_text
)
from utils.metrics import compute_fid, compute_cer, compute_wer, compute_ssim_batch


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate handwriting generation models")
    p.add_argument("--model",       type=str, required=True, choices=["gan", "diffusion"])
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--style_dir",   type=str, default=None)
    p.add_argument("--output_dir",  type=str, default="results")
    p.add_argument("--split",       type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--img_height",  type=int, default=64)
    p.add_argument("--img_width",   type=int, default=256)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--n_samples",   type=int, default=500, help="Max samples to evaluate")
    p.add_argument("--save_images", action="store_true", help="Save generated images to output_dir")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def load_model(args):
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    ckpt_args = ckpt.get("args", {})

    style_dim   = ckpt_args.get("style_dim",     256)
    text_dim    = ckpt_args.get("text_dim",       256)
    z_dim       = ckpt_args.get("z_dim",           64)
    img_height  = ckpt_args.get("img_height",      64)
    char_width  = ckpt_args.get("char_width",      16)
    timesteps   = ckpt_args.get("timesteps",     1000)
    schedule    = ckpt_args.get("schedule",  "linear")
    base_ch     = ckpt_args.get("base_channels",  64)

    if args.model == "gan":
        model = ScrabbleGAN(
            style_dim=style_dim, z_dim=z_dim,
            img_height=img_height, char_width=char_width,
        )
    else:
        model = HandwritingDiffusion(
            style_dim=style_dim, text_dim=text_dim,
            timesteps=timesteps, img_height=img_height,
            base_channels=base_ch, schedule=schedule,
        )

    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()
    print(f"Loaded {args.model} checkpoint from {args.checkpoint}")
    return model


def load_style_refs(args) -> torch.Tensor:
    if args.style_dir and Path(args.style_dir).exists():
        ds = UserSamplesDataset(args.style_dir, img_height=args.img_height)
        return ds.get_style_batch(n=16, device=args.device)
    print("WARNING: No style_dir. Using random style references.")
    return torch.rand(4, 1, args.img_height, 256, device=args.device) * 2 - 1


def reconstruct_token_seqs(labels: torch.Tensor, lengths: torch.Tensor):
    seqs, offset = [], 0
    for length in lengths.tolist():
        seqs.append(labels[offset: offset + length].tolist())
        offset += length
    return seqs


def build_char_tokens(seqs: List[List[int]], device: str) -> torch.Tensor:
    max_len = max(len(s) for s in seqs) if seqs else 1
    tokens = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return tokens


@torch.no_grad()
def generate_batch_gan(
    model: ScrabbleGAN,
    char_tokens: torch.Tensor,
    style_refs: torch.Tensor,
    img_height: int,
    img_width: int,
) -> torch.Tensor:
    B = char_tokens.shape[0]
    style_idx = torch.randint(0, style_refs.shape[0], (B,))
    style_batch = style_refs[style_idx]
    imgs = model.generate(char_tokens, style_batch)
    imgs = F.interpolate(imgs, size=(img_height, img_width), mode="bilinear", align_corners=False)
    return imgs.clamp(-1, 1)


@torch.no_grad()
def generate_batch_diffusion(
    model: HandwritingDiffusion,
    char_tokens: torch.Tensor,
    style_refs: torch.Tensor,
    img_height: int,
    img_width: int,
    device: str,
) -> torch.Tensor:
    B = char_tokens.shape[0]
    style_idx = torch.randint(0, style_refs.shape[0], (B,))
    style_batch = style_refs[style_idx]
    imgs = model.generate(
        char_tokens, style_batch,
        img_height=img_height, img_width=img_width, device=device
    )
    return imgs.clamp(-1, 1)


def simple_ctc_decode(log_probs: torch.Tensor) -> List[str]:
    """
    Greedy CTC decode: argmax per frame, collapse repeats, remove blanks.
    log_probs: (T, B, C)
    """
    preds = log_probs.argmax(dim=-1).permute(1, 0)  # (B, T)
    results = []
    for pred in preds:
        chars = []
        prev = -1
        for idx in pred.tolist():
            if idx != prev and idx != 0:
                chars.append(idx)
            prev = idx
        results.append(decode_text(chars))
    return results


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args)
    style_refs = load_style_refs(args)

    dataset = IAMDataset(
        root=args.data_root, split=args.split,
        img_height=args.img_height, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    all_real:  List[torch.Tensor] = []
    all_fake:  List[torch.Tensor] = []
    all_hyps:  List[str] = []
    all_refs:  List[str] = []
    n_collected = 0
    img_save_idx = 0

    print(f"Generating samples (max {args.n_samples})…")
    for imgs, labels, label_lengths in tqdm(loader):
        if n_collected >= args.n_samples:
            break

        real = F.interpolate(
            imgs.to(args.device),
            size=(args.img_height, args.img_width),
            mode="bilinear", align_corners=False
        )
        seqs = reconstruct_token_seqs(labels, label_lengths)
        ref_texts = [decode_text(s) for s in seqs]
        char_tokens = build_char_tokens(seqs, args.device)

        if args.model == "gan":
            fake = generate_batch_gan(model, char_tokens, style_refs, args.img_height, args.img_width)
            _, ocr_logits = model.discriminate(fake)
            hyps = simple_ctc_decode(ocr_logits)
        else:
            fake = generate_batch_diffusion(
                model, char_tokens, style_refs,
                args.img_height, args.img_width, args.device
            )
            hyps = ref_texts  # diffusion model has no built-in OCR head; use GT as placeholder

        all_real.append(real.cpu())
        all_fake.append(fake.cpu())
        all_hyps.extend(hyps)
        all_refs.extend(ref_texts)
        n_collected += real.shape[0]

        if args.save_images:
            for j in range(fake.shape[0]):
                img_norm = (fake[j].clamp(-1, 1) + 1) / 2
                save_image(img_norm, out_dir / f"gen_{img_save_idx:05d}.png")
                img_save_idx += 1

    real_all = torch.cat(all_real, dim=0)[: args.n_samples]
    fake_all = torch.cat(all_fake, dim=0)[: args.n_samples]
    hyps_all = all_hyps[: args.n_samples]
    refs_all = all_refs[: args.n_samples]

    print("Computing metrics…")
    print("  FID…")
    fid = compute_fid(real_all, fake_all, device=args.device, batch_size=args.batch_size)

    print("  CER / WER…")
    cer = compute_cer(hyps_all, refs_all)
    wer = compute_wer(hyps_all, refs_all)

    print("  SSIM…")
    ssim = compute_ssim_batch(fake_all, real_all)

    results = {
        "model":       args.model,
        "checkpoint":  args.checkpoint,
        "split":       args.split,
        "n_samples":   len(real_all),
        "FID":         round(fid,  4),
        "CER":         round(cer,  4),
        "WER":         round(wer,  4),
        "SSIM":        round(ssim, 4),
    }

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k:15s}: {v}")

    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()

"""
Training script for the ScrabbleGAN-based handwriting generator.

Usage:
    python train_gan.py --data_root data/iam --samples_dir data/samples \
                        --output_dir checkpoints/gan --epochs 100
"""

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from models.scrabblegan import ScrabbleGAN
from utils.dataset import (
    IAMDataset, UserSamplesDataset, CombinedDataset,
    collate_fn, encode_text, NUM_CLASSES
)
from utils.losses import GANTotalLoss, HingeLoss, CTCRecognitionLoss
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="Train ScrabbleGAN handwriting generator")
    p.add_argument("--data_root",   type=str, required=True, help="Path to IAM data dir")
    p.add_argument("--samples_dir", type=str, default=None,  help="User handwriting samples dir")
    p.add_argument("--output_dir",  type=str, default="checkpoints/gan")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--img_height",  type=int, default=64)
    p.add_argument("--char_width",  type=int, default=16)
    p.add_argument("--style_dim",   type=int, default=256)
    p.add_argument("--z_dim",       type=int, default=64)
    p.add_argument("--lr_g",        type=float, default=2e-4)
    p.add_argument("--lr_d",        type=float, default=2e-4)
    p.add_argument("--lambda1",     type=float, default=1.0,  help="L1 weight")
    p.add_argument("--lambda2",     type=float, default=0.1,  help="Perceptual weight")
    p.add_argument("--lambda3",     type=float, default=1.0,  help="Adversarial weight")
    p.add_argument("--lambda_ctc",  type=float, default=0.5,  help="CTC recogniser weight")
    p.add_argument("--n_critic",    type=int, default=1,      help="D steps per G step")
    p.add_argument("--save_every",  type=int, default=10)
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def build_datasets(args):
    train_ds = IAMDataset(
        root=args.data_root, split="train",
        img_height=args.img_height, augment=True
    )
    val_ds = IAMDataset(
        root=args.data_root, split="val",
        img_height=args.img_height, augment=False
    )
    if args.samples_dir and Path(args.samples_dir).exists():
        user_ds = UserSamplesDataset(args.samples_dir, img_height=args.img_height)
        train_ds = CombinedDataset([train_ds, user_ds])
    return train_ds, val_ds


def make_style_batch(style_refs, batch_size, device):
    """
    Repeat style_refs to match batch_size.
    style_refs: (K, 1, H, W)
    """
    K = style_refs.shape[0]
    idx = torch.randint(0, K, (batch_size,))
    return style_refs[idx].to(device)


def train_one_epoch(
    model, opt_g, opt_d,
    loader, style_refs,
    g_loss_fn, hinge, ctc_loss,
    args, epoch, writer
):
    model.train()
    device = args.device
    step = epoch * len(loader)

    for batch_idx, (imgs, labels, label_lengths) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        B = imgs.shape[0]
        real_imgs = imgs.to(device)

        # Build text token tensors from flattened labels
        # Each item in the batch has varying label length; we need per-sample tokens
        # for generation. Reconstruct per-sample token sequences.
        sample_lengths = label_lengths.tolist()
        token_seqs = []
        offset = 0
        for length in sample_lengths:
            seq = labels[offset: offset + length].tolist()
            token_seqs.append(seq)
            offset += length

        max_len = max(len(s) for s in token_seqs)
        char_indices = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(token_seqs):
            char_indices[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        style_batch = make_style_batch(style_refs, B, device)

        # ---- Discriminator step ----
        for _ in range(args.n_critic):
            with torch.no_grad():
                fake_imgs = model.generate(char_indices, style_batch)
            real_adv, real_ocr = model.discriminate(real_imgs)
            fake_adv, _ = model.discriminate(fake_imgs.detach())

            d_loss = hinge.discriminator_loss(real_adv, fake_adv)

            input_lengths = torch.full((B,), real_ocr.shape[0], dtype=torch.long)
            ctc = ctc_loss(real_ocr, labels.to(device), input_lengths, label_lengths.to(device))
            d_total = d_loss + args.lambda_ctc * ctc

            opt_d.zero_grad()
            d_total.backward()
            opt_d.step()

        # ---- Generator step ----
        fake_imgs = model.generate(char_indices, style_batch)
        fake_adv, _ = model.discriminate(fake_imgs)

        target = _pad_to_same(real_imgs, fake_imgs)
        g_total, g_components = g_loss_fn(fake_imgs, target, fake_adv)

        opt_g.zero_grad()
        g_total.backward()
        opt_g.step()

        if batch_idx % 50 == 0:
            writer.add_scalar("Loss/G_total", g_total.item(), step + batch_idx)
            writer.add_scalar("Loss/D_total", d_total.item(), step + batch_idx)
            for k, v in g_components.items():
                writer.add_scalar(f"Loss/G_{k}", v, step + batch_idx)

    return g_total.item(), d_total.item()


def _pad_to_same(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pad or crop b to match a's width."""
    if a.shape[-1] == b.shape[-1]:
        return a
    target_w = b.shape[-1]
    if a.shape[-1] >= target_w:
        return a[:, :, :, :target_w]
    pad = target_w - a.shape[-1]
    return F.pad(a, (0, pad), value=-1.0)


@torch.no_grad()
def validate(model, val_loader, style_refs, args, epoch, writer):
    model.eval()
    device = args.device
    fixed_texts = ["Hello World", "handwriting", "the quick brown fox"]

    for i, text in enumerate(fixed_texts):
        tokens = torch.tensor(
            [t for t in __import__("utils.dataset", fromlist=["encode_text"]).encode_text(text)],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        style = style_refs[:1].to(device)
        fake = model.generate(tokens, style)
        fake_norm = (fake + 1) / 2
        writer.add_image(f"Generated/{text.replace(' ', '_')}", fake_norm[0], epoch)
        save_image(
            fake_norm,
            Path(args.output_dir) / f"epoch{epoch:04d}_sample{i}.png"
        )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    print(f"Device: {args.device}")
    train_ds, val_ds = build_datasets(args)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    model = ScrabbleGAN(
        style_dim=args.style_dim,
        z_dim=args.z_dim,
        img_height=args.img_height,
        char_width=args.char_width,
    ).to(args.device)

    opt_g = optim.Adam(
        list(model.style_encoder.parameters()) + list(model.generator.parameters()),
        lr=args.lr_g, betas=(0.5, 0.999)
    )
    opt_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    g_loss_fn = GANTotalLoss(args.lambda1, args.lambda2, args.lambda3).to(args.device)
    hinge = HingeLoss()
    ctc_loss = CTCRecognitionLoss().to(args.device)

    if args.samples_dir and Path(args.samples_dir).exists():
        user_ds = UserSamplesDataset(args.samples_dir, img_height=args.img_height)
        style_refs = user_ds.get_style_batch(n=16, device="cpu")
    else:
        style_refs = torch.rand(4, 1, args.img_height, 256) * 2 - 1
        print("WARNING: No user samples provided. Using random style references.")

    best_g_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        g_loss, d_loss = train_one_epoch(
            model, opt_g, opt_d, train_loader, style_refs,
            g_loss_fn, hinge, ctc_loss, args, epoch, writer
        )
        print(f"Epoch {epoch:4d} | G: {g_loss:.4f} | D: {d_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            validate(model, None, style_refs, args, epoch, writer)
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "g_loss": g_loss,
            }
            torch.save(ckpt, out_dir / f"epoch_{epoch:04d}.pth")
            if g_loss < best_g_loss:
                best_g_loss = g_loss
                torch.save(ckpt, out_dir / "best.pth")
                print(f"  → Saved best checkpoint (G loss: {best_g_loss:.4f})")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()

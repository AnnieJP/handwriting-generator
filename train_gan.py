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
from typing import Optional

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
from utils.test_log import append_test_log
import torch.nn.functional as F

# changed by Nani - added device selection
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


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
    p.add_argument("--device", type=str, default=get_device())
    # changed by Nani - removed device arg since we auto-detect it now
    # p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--smoke_test",  action="store_true",
                   help="Run a short pipeline-verification cycle and print a pass/fail report")
    p.add_argument("--smoke_epochs", type=int, default=5)
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint to resume from, or 'auto' to use the latest in --output_dir.")
    return p.parse_args()


def resolve_resume_path(spec: Optional[str], out_dir: Path) -> Optional[Path]:
    """'auto' → latest epoch_*.pth in out_dir; literal path → that file."""
    if not spec:
        return None
    if spec == "auto":
        candidates = sorted(out_dir.glob("epoch_*.pth"))
        return candidates[-1] if candidates else None
    p = Path(spec)
    return p if p.exists() else None


def apply_smoke_overrides(args):
    """Force fast settings for a pipeline sanity check.

    Note: CTC is skipped (lambda_ctc=0) because torch.ctc_loss falls back to CPU
    on MPS, dominating wall time. Smoke test verifies pipeline wiring, not
    legibility — full training runs should re-enable CTC.
    """
    args.epochs = args.smoke_epochs
    args.batch_size = 4
    args.num_workers = 2
    args.save_every = 1
    args.lambda_ctc = 0.0
    if args.output_dir == "checkpoints/gan":
        args.output_dir = "checkpoints/gan_smoke"
    print(f"[smoke_test] epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lambda_ctc=0 (skipped — MPS CPU-fallback bottleneck), "
          f"output_dir={args.output_dir}")
    return args


def print_smoke_report(epoch_g_losses, epoch_d_losses, output_dir, model_name="gan"):
    """For GAN, use D-loss as primary verdict signal (cleaner than 3-component G-loss)."""
    if len(epoch_d_losses) < 2:
        print(f"\n=== {model_name} smoke test: SKIPPED (need >= 2 epochs) ===")
        return

    g_start, g_end = epoch_g_losses[0], epoch_g_losses[-1]
    d_start, d_end = epoch_d_losses[0], epoch_d_losses[-1]
    g_ratio = g_end / g_start if g_start else float("inf")
    d_ratio = d_end / d_start if d_start else float("inf")

    if d_ratio < 0.7:
        verdict = "PASS — D-loss decreasing healthily; pipeline is wired correctly"
    elif d_ratio < 1.0:
        verdict = "MARGINAL — D-loss decreasing slowly; try more epochs"
    else:
        verdict = "FAIL — D-loss not decreasing; check LR, data, model dims, device"

    samples = sorted(Path(output_dir).glob("epoch*_sample*.png"))
    bar = "=" * 60
    n = len(epoch_d_losses)
    print(f"\n{bar}\n  {model_name.upper()} SMOKE TEST REPORT\n{bar}")
    print(f"  Epochs run     : {n}")
    print(f"  D-loss epoch 1 : {d_start:.4f}   |  G-loss epoch 1 : {g_start:.4f}")
    print(f"  D-loss epoch {n:<2}: {d_end:.4f}   |  G-loss epoch {n:<2}: {g_end:.4f}")
    print(f"  D ratio        : {d_ratio:.3f}  (pass < 0.70, primary signal)")
    print(f"  G ratio        : {g_ratio:.3f}  (informational; GAN G-loss is noisy in short runs)")
    print(f"  Verdict        : {verdict}")
    if samples:
        print(f"  Sample images  : {len(samples)} files in {output_dir}/")
        print(f"                   compare {samples[0].name} vs {samples[-1].name}")
    print(bar + "\n")


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
    g_losses, d_losses = [], []

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

            if args.lambda_ctc > 0:
                input_lengths = torch.full((B,), real_ocr.shape[0], dtype=torch.long)
                ctc = ctc_loss(real_ocr, labels.to(device), input_lengths, label_lengths.to(device))
                d_total = d_loss + args.lambda_ctc * ctc
            else:
                d_total = d_loss

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

        g_losses.append(g_total.item())
        d_losses.append(d_total.item())

        if batch_idx % 50 == 0:
            writer.add_scalar("Loss/G_total", g_total.item(), step + batch_idx)
            writer.add_scalar("Loss/D_total", d_total.item(), step + batch_idx)
            for k, v in g_components.items():
                writer.add_scalar(f"Loss/G_{k}", v, step + batch_idx)

    avg_g = sum(g_losses) / max(1, len(g_losses))
    avg_d = sum(d_losses) / max(1, len(d_losses))
    return avg_g, avg_d


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
    if args.smoke_test:
        args = apply_smoke_overrides(args)
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
        # changed by Nani - set pin_memory based on device
        pin_memory = (args.device == "cuda"),
        # pin_memory=True,
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
    epoch_g_losses = []
    epoch_d_losses = []
    start_epoch = 1

    resume_path = resolve_resume_path(args.resume_from, out_dir)
    if resume_path is not None:
        print(f"[resume] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        best_g_loss = ckpt.get("best_g_loss", float("inf"))
        print(f"[resume] Continuing from epoch {start_epoch} (best_g_loss={best_g_loss:.4f})")
    elif args.resume_from:
        print(f"[resume] No checkpoint found for '{args.resume_from}'; starting fresh.")

    train_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        g_loss, d_loss = train_one_epoch(
            model, opt_g, opt_d, train_loader, style_refs,
            g_loss_fn, hinge, ctc_loss, args, epoch, writer
        )
        print(f"Epoch {epoch:4d} | G: {g_loss:.4f} | D: {d_loss:.4f}")
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            validate(model, None, style_refs, args, epoch, writer)
            if g_loss < best_g_loss:
                best_g_loss = g_loss
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "g_loss": g_loss,
                "best_g_loss": best_g_loss,
            }
            torch.save(ckpt, out_dir / f"epoch_{epoch:04d}.pth")
            torch.save(ckpt, out_dir / "latest.pth")
            if g_loss == best_g_loss:
                torch.save(ckpt, out_dir / "best.pth")
                print(f"  → Saved best checkpoint (G loss: {best_g_loss:.4f})")

    writer.close()
    print("Training complete.")

    if args.smoke_test:
        print_smoke_report(epoch_g_losses, epoch_d_losses, out_dir, model_name="gan")
        append_test_log(
            model_name="gan",
            config={
                "data_root": args.data_root,
                "samples_dir": args.samples_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr_g": args.lr_g,
                "lr_d": args.lr_d,
                "lambda1": args.lambda1,
                "lambda2": args.lambda2,
                "lambda3": args.lambda3,
                "lambda_ctc": args.lambda_ctc,
                "n_critic": args.n_critic,
                "device": args.device,
            },
            epoch_losses=epoch_d_losses,
            aux_losses={"G-loss": epoch_g_losses},
            wall_time_seconds=time.time() - train_start,
            output_dir=out_dir,
            notes="Primary verdict signal is D-loss (cleaner than 3-component G-loss for GAN smoke).",
        )


if __name__ == "__main__":
    main()

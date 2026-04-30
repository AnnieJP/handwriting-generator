"""
Training script for the diffusion-based handwriting generator.

Usage:
    python train_diffusion.py --data_root data/iam --samples_dir data/samples \
                              --output_dir checkpoints/diffusion --epochs 100
"""

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from models.diffusion import HandwritingDiffusion
from utils.dataset import (
    IAMDataset, UserSamplesDataset, CombinedDataset,
    collate_fn, encode_text
)
from utils.losses import DiffusionTotalLoss
from utils.test_log import append_test_log

# changed by Nani - added device selection
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion handwriting generator")
    p.add_argument("--data_root",     type=str, required=True)
    p.add_argument("--samples_dir",   type=str, default=None)
    p.add_argument("--output_dir",    type=str, default="checkpoints/diffusion")
    p.add_argument("--epochs",        type=int, default=100)
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--img_height",    type=int, default=64)
    p.add_argument("--img_width",     type=int, default=256)
    p.add_argument("--style_dim",     type=int, default=256)
    p.add_argument("--text_dim",      type=int, default=256)
    p.add_argument("--timesteps",     type=int, default=1000)
    p.add_argument("--schedule",      type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--lambda_style",  type=float, default=0.5)
    p.add_argument("--lambda_text",   type=float, default=0.5)
    p.add_argument("--save_every",    type=int, default=10)
    p.add_argument("--sample_every",  type=int, default=5)
    p.add_argument("--device", type=str, default=get_device())
    # changed by Nani - removed device arg since we auto-detect it now
    # p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--resume_from",   type=str, default=None,
                   help="Path to checkpoint to resume from, or 'auto' to use the latest in --output_dir.")
    p.add_argument("--smoke_test",    action="store_true",
                   help="Run a short pipeline-verification cycle and print a pass/fail report")
    p.add_argument("--smoke_epochs",  type=int, default=5)
    return p.parse_args()


def apply_smoke_overrides(args):
    """Force fast settings for a pipeline sanity check."""
    args.epochs = args.smoke_epochs
    args.batch_size = 4
    args.timesteps = 250
    args.num_workers = 2
    args.sample_every = 1
    args.save_every = args.smoke_epochs
    if args.output_dir == "checkpoints/diffusion":
        args.output_dir = "checkpoints/diffusion_smoke"
    print(f"[smoke_test] epochs={args.epochs}, batch_size={args.batch_size}, "
          f"timesteps={args.timesteps}, output_dir={args.output_dir}")
    return args


def print_smoke_report(epoch_losses, output_dir, model_name):
    if len(epoch_losses) < 2:
        print(f"\n=== {model_name} smoke test: SKIPPED (need >= 2 epochs) ===")
        return

    start, end = epoch_losses[0], epoch_losses[-1]
    ratio = end / start if start > 0 else float("inf")

    if ratio < 0.7:
        verdict = "PASS — loss decreasing healthily; pipeline is wired correctly"
    elif ratio < 1.0:
        verdict = "MARGINAL — loss decreasing slowly; try more epochs or tune LR"
    else:
        verdict = "FAIL — loss not decreasing; check LR, data, model dims, device"

    samples = sorted(Path(output_dir).glob("epoch*_sample*.png"))
    bar = "=" * 60
    print(f"\n{bar}\n  {model_name.upper()} SMOKE TEST REPORT\n{bar}")
    print(f"  Epochs run     : {len(epoch_losses)}")
    print(f"  Loss epoch 1   : {start:.4f}")
    print(f"  Loss epoch {len(epoch_losses):<3}: {end:.4f}")
    print(f"  Ratio end/start: {ratio:.3f}  (pass < 0.70)")
    print(f"  Verdict        : {verdict}")
    if samples:
        print(f"  Sample images  : {len(samples)} files in {output_dir}/")
        print(f"                   compare {samples[0].name} vs {samples[-1].name}")
    print(bar + "\n")


def resolve_resume_path(spec: Optional[str], out_dir: Path) -> Optional[Path]:
    """'auto' → latest epoch_*.pth in out_dir; literal path → that file."""
    if not spec:
        return None
    if spec == "auto":
        candidates = sorted(out_dir.glob("epoch_*.pth"))
        return candidates[-1] if candidates else None
    p = Path(spec)
    return p if p.exists() else None


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


def preprocess_batch(imgs, labels, label_lengths, img_height, img_width, device):
    """Resize images to fixed (img_height, img_width) and build token tensors."""
    import torch.nn.functional as F
    imgs = F.interpolate(imgs.to(device), size=(img_height, img_width), mode="bilinear", align_corners=False)

    B = imgs.shape[0]
    sample_lengths = label_lengths.tolist()
    token_seqs = []
    offset = 0
    for length in sample_lengths:
        seq = labels[offset: offset + length].tolist()
        token_seqs.append(seq)
        offset += length

    max_len = max(len(s) for s in token_seqs)
    char_tokens = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(token_seqs):
        char_tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return imgs, char_tokens


def train_one_epoch(
    model, optimizer, loss_fn,
    loader, style_refs,
    args, epoch, writer
):
    model.train()
    device = args.device
    total_loss = 0.0
    step_offset = (epoch - 1) * len(loader)

    for batch_idx, (imgs, labels, label_lengths) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        B = imgs.shape[0]
        x0, char_tokens = preprocess_batch(
            imgs, labels, label_lengths,
            args.img_height, args.img_width, device
        )

        style_batch_idx = torch.randint(0, style_refs.shape[0], (B,))
        style_batch = style_refs[style_batch_idx].to(device)

        t = torch.randint(0, args.timesteps, (B,), device=device).long()

        pred_noise, target_noise = model(x0, t, char_tokens, style_batch)
        loss, loss_dict = loss_fn(pred_noise, target_noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        global_step = step_offset + batch_idx
        if batch_idx % 50 == 0:
            writer.add_scalar("Loss/diffusion_total", loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"Loss/{k}", v, global_step)

    return total_loss / len(loader)


@torch.no_grad()
def sample_and_log(model, style_refs, args, epoch, writer, out_dir):
    model.eval()
    device = args.device
    fixed_texts = ["Hello World", "the quick brown fox", r"f(x) = sin(x)"]

    for i, text in enumerate(fixed_texts):
        tokens = torch.tensor(
            encode_text(text), dtype=torch.long, device=device
        ).unsqueeze(0)
        style = style_refs[:1].to(device)
        generated = model.generate(
            tokens, style,
            img_height=args.img_height,
            img_width=args.img_width,
            device=device
        )
        img_norm = (generated.clamp(-1, 1) + 1) / 2
        writer.add_image(f"Generated/{text[:20].replace(' ', '_')}", img_norm[0], epoch)
        save_image(img_norm, out_dir / f"epoch{epoch:04d}_sample{i}.png")


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

    model = HandwritingDiffusion(
        style_dim=args.style_dim,
        text_dim=args.text_dim,
        timesteps=args.timesteps,
        img_height=args.img_height,
        base_channels=args.base_channels,
        schedule=args.schedule,
    ).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    loss_fn = DiffusionTotalLoss(
        lambda_style=args.lambda_style,
        lambda_text=args.lambda_text,
    ).to(args.device)

    if args.samples_dir and Path(args.samples_dir).exists():
        user_ds = UserSamplesDataset(args.samples_dir, img_height=args.img_height)
        style_refs = user_ds.get_style_batch(n=16, device="cpu")
    else:
        style_refs = torch.rand(4, 1, args.img_height, 256) * 2 - 1
        print("WARNING: No user samples provided. Using random style references.")

    best_loss = float("inf")
    epoch_losses = []
    start_epoch = 1

    resume_path = resolve_resume_path(args.resume_from, out_dir)
    if resume_path is not None:
        print(f"[resume] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[resume] Continuing from epoch {start_epoch} (best_loss={best_loss:.4f})")
    elif args.resume_from:
        print(f"[resume] No checkpoint found for '{args.resume_from}'; starting fresh.")

    train_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(
            model, optimizer, loss_fn,
            train_loader, style_refs,
            args, epoch, writer
        )
        scheduler.step()
        print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        writer.add_scalar("Train/avg_loss", avg_loss, epoch)
        epoch_losses.append(avg_loss)

        if epoch % args.sample_every == 0:
            sample_and_log(model, style_refs, args, epoch, writer, out_dir)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            if avg_loss < best_loss:
                best_loss = avg_loss
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": avg_loss,
                "best_loss": best_loss,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / f"epoch_{epoch:04d}.pth")
            torch.save(ckpt, out_dir / "latest.pth")
            if avg_loss == best_loss:
                torch.save(ckpt, out_dir / "best.pth")
                print(f"  → Saved best checkpoint (loss: {best_loss:.4f})")

    writer.close()
    print("Training complete.")

    if args.smoke_test:
        print_smoke_report(epoch_losses, out_dir, model_name="diffusion")
        append_test_log(
            model_name="diffusion",
            config={
                "data_root": args.data_root,
                "samples_dir": args.samples_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "timesteps": args.timesteps,
                "schedule": args.schedule,
                "lr": args.lr,
                "device": args.device,
            },
            epoch_losses=epoch_losses,
            wall_time_seconds=time.time() - train_start,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()

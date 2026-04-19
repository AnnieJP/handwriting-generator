"""
Inference script — generate handwritten images from text or LaTeX input.

Usage:
    # Plain text with GAN model
    python inference.py --model gan --checkpoint checkpoints/gan/best.pth \
                        --text "Hello World" --style_dir data/samples \
                        --output output/hello.png

    # LaTeX expression with diffusion model
    python inference.py --model diffusion --checkpoint checkpoints/diffusion/best.pth \
                        --latex "\\frac{d}{dx} e^x = e^x" \
                        --style_dir data/samples --output output/math.png

    # Multiple lines from a text file
    python inference.py --model diffusion --checkpoint checkpoints/diffusion/best.pth \
                        --input_file my_notes.txt --style_dir data/samples \
                        --output_dir output/notes/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.utils import save_image

from models.scrabblegan import ScrabbleGAN
from models.diffusion import HandwritingDiffusion
from utils.dataset import encode_text, UserSamplesDataset
from utils.latex_parser import preprocess_input, latex_to_text


def parse_args():
    p = argparse.ArgumentParser(description="Handwriting generation inference")
    p.add_argument("--model",       type=str, required=True, choices=["gan", "diffusion"])
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--style_dir",   type=str, default=None, help="User handwriting samples dir")
    p.add_argument("--text",        type=str, default=None,  help="Plain text to generate")
    p.add_argument("--latex",       type=str, default=None,  help="LaTeX expression to generate")
    p.add_argument("--input_file",  type=str, default=None,  help="Text file with one line per input")
    p.add_argument("--output",      type=str, default=None,  help="Single output image path")
    p.add_argument("--output_dir",  type=str, default=None,  help="Output directory for multiple lines")
    p.add_argument("--img_height",  type=int, default=64)
    p.add_argument("--img_width",   type=int, default=256)
    p.add_argument("--n_style_refs",type=int, default=8,     help="Number of style reference images")
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(args):
    ckpt = torch.load(args.checkpoint, map_location=args.device)

    ckpt_args = ckpt.get("args", {})
    style_dim   = ckpt_args.get("style_dim",   256)
    text_dim    = ckpt_args.get("text_dim",    256)
    z_dim       = ckpt_args.get("z_dim",        64)
    img_height  = ckpt_args.get("img_height",   64)
    char_width  = ckpt_args.get("char_width",   16)
    timesteps   = ckpt_args.get("timesteps",  1000)
    schedule    = ckpt_args.get("schedule", "linear")
    base_ch     = ckpt_args.get("base_channels", 64)

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
    return model


def load_style_refs(args) -> torch.Tensor:
    if args.style_dir and Path(args.style_dir).exists():
        ds = UserSamplesDataset(args.style_dir, img_height=args.img_height)
        return ds.get_style_batch(n=args.n_style_refs, device=args.device)
    print("WARNING: No style_dir provided. Using random style references.")
    return torch.rand(4, 1, args.img_height, 256, device=args.device) * 2 - 1


def text_to_tokens(text: str, device: str) -> torch.Tensor:
    indices = encode_text(text)
    if not indices:
        indices = [0]
    return torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def generate_gan(model, text: str, style_refs: torch.Tensor, args) -> torch.Tensor:
    tokens = text_to_tokens(text, args.device)
    style = style_refs[:1]
    img = model.generate(tokens, style)
    return img.clamp(-1, 1)


@torch.no_grad()
def generate_diffusion(model, text: str, style_refs: torch.Tensor, args) -> torch.Tensor:
    tokens = text_to_tokens(text, args.device)
    style = style_refs[:1]
    img = model.generate(
        tokens, style,
        img_height=args.img_height,
        img_width=args.img_width,
        device=args.device,
    )
    return img.clamp(-1, 1)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    img = (img_tensor.squeeze(0).cpu().numpy() + 1) / 2  # [-1,1] → [0,1]
    if img.shape[0] == 1:
        img = img.squeeze(0)
    else:
        img = img.transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L" if img.ndim == 2 else "RGB")


def collect_inputs(args) -> List[str]:
    inputs = []
    if args.text:
        inputs.append(args.text)
    if args.latex:
        converted = latex_to_text(args.latex)
        print(f"LaTeX → text: {converted}")
        inputs.append(converted)
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                processed = preprocess_input(line)
                inputs.append(processed)
    if not inputs:
        print("ERROR: Provide --text, --latex, or --input_file.")
        sys.exit(1)
    return inputs


def main():
    args = parse_args()

    model = load_model(args)
    style_refs = load_style_refs(args)
    inputs = collect_inputs(args)

    generate_fn = generate_gan if args.model == "gan" else generate_diffusion

    if len(inputs) == 1 and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_tensor = generate_fn(model, inputs[0], style_refs, args)
        pil_img = tensor_to_pil(img_tensor)
        pil_img.save(out_path)
        print(f"Saved: {out_path}")
    else:
        out_dir = Path(args.output_dir or "output")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(inputs):
            img_tensor = generate_fn(model, text, style_refs, args)
            pil_img = tensor_to_pil(img_tensor)
            fname = out_dir / f"generated_{i:04d}.png"
            pil_img.save(fname)
            print(f"[{i+1}/{len(inputs)}] '{text[:40]}' → {fname}")

    print("Done.")


if __name__ == "__main__":
    main()

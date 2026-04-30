"""
Gradio web UI for the handwriting generator.

Wraps inference.py so you can upload style references, type text or LaTeX,
and generate handwriting images from a browser.

Launch:
    python app.py
    python app.py --gan_ckpt checkpoints/gan/best.pth \
                  --diff_ckpt checkpoints/diffusion/best.pth
    python app.py --share        # public Gradio link (tunnel)
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image

from inference import (
    generate_diffusion,
    generate_gan,
    load_model as _load_model,
    tensor_to_pil,
)
from utils.dataset import UserSamplesDataset
from utils.latex_parser import latex_to_text, preprocess_input


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


DEVICE = _select_device()
IMG_HEIGHT = 64
IMG_WIDTH = 256
N_STYLE_REFS = 8

_models: dict = {"gan": None, "diffusion": None}


def _ckpt_args(model_name: str, checkpoint: str) -> SimpleNamespace:
    return SimpleNamespace(model=model_name, checkpoint=checkpoint, device=DEVICE)


def _gen_args() -> SimpleNamespace:
    return SimpleNamespace(device=DEVICE, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)


def load_models(gan_ckpt: Optional[str], diff_ckpt: Optional[str]) -> List[str]:
    notes = []
    if gan_ckpt and Path(gan_ckpt).exists():
        print(f"Loading GAN checkpoint: {gan_ckpt}")
        _models["gan"] = _load_model(_ckpt_args("gan", gan_ckpt))
        notes.append(f"GAN: loaded ({gan_ckpt})")
    else:
        notes.append(f"GAN: not loaded (no checkpoint at {gan_ckpt})")

    if diff_ckpt and Path(diff_ckpt).exists():
        print(f"Loading Diffusion checkpoint: {diff_ckpt}")
        _models["diffusion"] = _load_model(_ckpt_args("diffusion", diff_ckpt))
        notes.append(f"Diffusion: loaded ({diff_ckpt})")
    else:
        notes.append(f"Diffusion: not loaded (no checkpoint at {diff_ckpt})")

    return notes


def _style_refs_from_uploads(uploaded: Optional[List]) -> torch.Tensor:
    """Build a (N, 1, H, W) style tensor from uploaded files, or fall back to noise."""
    if not uploaded:
        return torch.rand(4, 1, IMG_HEIGHT, IMG_WIDTH, device=DEVICE) * 2 - 1

    tmpdir = Path(tempfile.mkdtemp(prefix="hwgen_style_"))
    kept = 0
    for i, f in enumerate(uploaded):
        src = Path(f.name if hasattr(f, "name") else f)
        if src.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        shutil.copy(src, tmpdir / f"sample_{i:03d}{src.suffix.lower()}")
        kept += 1

    if kept == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return torch.rand(4, 1, IMG_HEIGHT, IMG_WIDTH, device=DEVICE) * 2 - 1

    try:
        ds = UserSamplesDataset(str(tmpdir), img_height=IMG_HEIGHT)
        refs = ds.get_style_batch(n=N_STYLE_REFS, device=DEVICE)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return refs


def generate(
    text: str,
    is_latex: bool,
    style_files: Optional[List],
    model_choice: str,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    if not text or not text.strip():
        return None, None, "Please enter some text."

    text_processed = latex_to_text(text) if is_latex else preprocess_input(text)
    style_refs = _style_refs_from_uploads(style_files)
    args = _gen_args()

    gan_img: Optional[Image.Image] = None
    diff_img: Optional[Image.Image] = None
    timings: List[str] = []

    want_gan = model_choice in ("gan", "both")
    want_diff = model_choice in ("diffusion", "both")

    if want_gan:
        if _models["gan"] is None:
            return None, None, "GAN checkpoint not loaded — pass --gan_ckpt to app.py."
        t0 = time.time()
        out = generate_gan(_models["gan"], text_processed, style_refs, args)
        gan_img = tensor_to_pil(out)
        timings.append(f"GAN {time.time() - t0:.2f}s")

    if want_diff:
        if _models["diffusion"] is None:
            return None, None, "Diffusion checkpoint not loaded — pass --diff_ckpt to app.py."
        t0 = time.time()
        out = generate_diffusion(_models["diffusion"], text_processed, style_refs, args)
        diff_img = tensor_to_pil(out)
        timings.append(f"Diffusion {time.time() - t0:.2f}s")

    status = (
        f"**Input:** `{text_processed}`  |  "
        f"**Device:** `{DEVICE}`  |  "
        f"**Style refs:** {len(style_files) if style_files else 0} uploaded  |  "
        f"**Time:** {' · '.join(timings) if timings else 'n/a'}"
    )
    return gan_img, diff_img, status


def build_ui(load_notes: List[str]) -> gr.Blocks:
    notes_md = "\n".join(f"- {n}" for n in load_notes)
    with gr.Blocks(title="Handwriting Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Personalized Handwriting Generator\n"
            "CS 6384 — IAM dataset · ScrabbleGAN vs. Diffusion\n\n"
            f"**Status:**\n{notes_md}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Style references")
                style_files = gr.File(
                    label="Upload 1–8 handwriting samples (PNG / JPG)",
                    file_count="multiple",
                    file_types=["image"],
                )

                gr.Markdown("### 2. Input")
                is_latex = gr.Checkbox(label="LaTeX mode", value=False)
                text = gr.Textbox(
                    label="Text to render",
                    value="Hello World",
                    lines=2,
                    placeholder=r"Hello World   /   \frac{d}{dx} e^x = e^x",
                )

                gr.Markdown("### 3. Model")
                model_choice = gr.Radio(
                    choices=["gan", "diffusion", "both"],
                    value="gan",
                    label="Generator (diffusion is slower but higher quality)",
                )

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Output")
                with gr.Row():
                    gan_out = gr.Image(label="ScrabbleGAN", type="pil", height=160)
                    diff_out = gr.Image(label="Diffusion", type="pil", height=160)
                status = gr.Markdown("")

                gr.Examples(
                    examples=[
                        ["Hello World", False, "gan"],
                        ["the quick brown fox", False, "both"],
                        [r"\frac{d}{dx} e^x = e^x", True, "diffusion"],
                        [r"\alpha + \beta = \gamma", True, "both"],
                    ],
                    inputs=[text, is_latex, model_choice],
                    label="Examples",
                )

        generate_btn.click(
            fn=generate,
            inputs=[text, is_latex, style_files, model_choice],
            outputs=[gan_out, diff_out, status],
        )

    return demo


def parse_args():
    p = argparse.ArgumentParser(description="Gradio UI for the handwriting generator")
    p.add_argument("--gan_ckpt", type=str, default="checkpoints/gan/best.pth")
    p.add_argument("--diff_ckpt", type=str, default="checkpoints/diffusion/best.pth")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    notes = load_models(args.gan_ckpt, args.diff_ckpt)
    if _models["gan"] is None and _models["diffusion"] is None:
        print("WARNING: No checkpoints loaded. UI will start but generation will fail.")
    demo = build_ui(notes)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

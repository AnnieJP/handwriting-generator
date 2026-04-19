# Personalized Handwriting Generation from Printed and LaTeX Text

CS 4391 – Computer Vision | University of Texas at Dallas

## Overview

This project generates personalized handwritten text from printed input (including LaTeX expressions) conditioned on a user's handwriting style. Two approaches are implemented and compared:

1. **ScrabbleGAN** – GAN-based variable-length handwriting generation
2. **Diffusion Model** – DDPM-based generation conditioned on style and text (One-DM / DiffusionPen inspired)

## Project Structure

```
HandwritingGenerator/
├── data/
│   ├── iam/                  # IAM Handwriting Database (download separately)
│   │   ├── words/            # Word images
│   │   ├── lines/            # Line images
│   │   ├── xml/              # XML annotations
│   │   └── splits/           # Train/val/test split files
│   ├── crohme/               # CROHME dataset (download separately)
│   └── samples/              # User handwriting samples (place your images here)
├── models/
│   ├── __init__.py
│   ├── style_encoder.py      # CNN-based style feature extractor
│   ├── scrabblegan.py        # GAN generator + discriminator
│   └── diffusion.py          # DDPM U-Net + HandwritingDiffusion model
├── utils/
│   ├── __init__.py
│   ├── dataset.py            # IAM / CROHME / user sample data loaders
│   ├── latex_parser.py       # LaTeX → symbolic text conversion
│   ├── losses.py             # Perceptual, GAN, denoising, style, text losses
│   └── metrics.py            # FID, CER/WER, SSIM evaluation
├── train_gan.py              # GAN training script
├── train_diffusion.py        # Diffusion model training script
├── inference.py              # Generate handwriting from text / LaTeX input
├── evaluate.py               # Compute FID, CER/WER, SSIM on generated samples
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

### IAM Handwriting Database
1. Register and download from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
2. Extract to `data/iam/` following the structure above

### CROHME
1. Download from https://tc11.cvc.uab.es/datasets/CROHME_task1_2019
2. Extract to `data/crohme/`

### User Handwriting Samples
Place your handwriting sample images (PNG/JPG) in `data/samples/`. Each image should contain a single line of handwriting. A corresponding `.txt` file with the same name should contain the transcription.

Example:
```
data/samples/
  sample_01.png
  sample_01.txt   # contents: "Hello world this is my handwriting"
  sample_02.png
  sample_02.txt
```

## Training

### GAN Approach (ScrabbleGAN)

```bash
python train_gan.py \
    --data_root data/iam \
    --samples_dir data/samples \
    --output_dir checkpoints/gan \
    --epochs 100 \
    --batch_size 16 \
    --img_height 64 \
    --lr_g 2e-4 \
    --lr_d 2e-4 \
    --lambda1 1.0 \
    --lambda2 0.1 \
    --lambda3 1.0
```

### Diffusion Approach (One-DM / DiffusionPen)

```bash
python train_diffusion.py \
    --data_root data/iam \
    --samples_dir data/samples \
    --output_dir checkpoints/diffusion \
    --epochs 100 \
    --batch_size 8 \
    --img_height 64 \
    --timesteps 1000 \
    --lambda_style 0.5 \
    --lambda_text 0.5
```

## Inference

```bash
# Generate from plain text
python inference.py \
    --model gan \
    --checkpoint checkpoints/gan/best.pth \
    --text "Hello World" \
    --style_dir data/samples \
    --output output/generated.png

# Generate from LaTeX
python inference.py \
    --model diffusion \
    --checkpoint checkpoints/diffusion/best.pth \
    --latex "\frac{d}{dx} e^x = e^x" \
    --style_dir data/samples \
    --output output/generated_math.png
```

## Evaluation

```bash
python evaluate.py \
    --model gan \
    --checkpoint checkpoints/gan/best.pth \
    --data_root data/iam \
    --style_dir data/samples \
    --output_dir results/gan
```

## Loss Functions

**GAN:**
$$L_{GAN} = \lambda_1 L_1 + \lambda_2 L_{perc} + \lambda_3 L_{adv}$$

**Diffusion:**
$$L_{diff} = L_{denoise} + \lambda_{style} L_{style} + \lambda_{text} L_{text}$$

## Evaluation Metrics

- **FID** – Fréchet Inception Distance (visual realism)
- **CER / WER** – Character / Word Error Rate (content correctness)
- **SSIM** – Structural Similarity Index (style similarity)

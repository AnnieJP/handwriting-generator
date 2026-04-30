# Handwriting Generation Project — Pipeline Overview

**Course:** CS 6384 Computer Vision  
**Dataset:** IAM Handwriting Database  
**Goal:** Generate realistic, style-personalized handwritten text images from input strings (including LaTeX math expressions), using two complementary deep learning architectures: ScrabbleGAN and a DDPM-based Diffusion Model.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [High-Level Pipeline](#high-level-pipeline)
3. [Step 1: Dataset Preparation](#step-1-dataset-preparation)
4. [Step 2: Data Loading & Preprocessing](#step-2-data-loading--preprocessing)
5. [Step 3: Style Encoding](#step-3-style-encoding)
6. [Step 4a: ScrabbleGAN (GAN Path)](#step-4a-scrabblegan-gan-path)
7. [Step 4b: Diffusion Model (DDPM Path)](#step-4b-diffusion-model-ddpm-path)
8. [Step 5: Training](#step-5-training)
9. [Step 6: Inference](#step-6-inference)
10. [Step 7: Evaluation](#step-7-evaluation)
11. [LaTeX Support](#latex-support)
12. [Loss Functions](#loss-functions)
13. [Dependencies](#dependencies)

---

## Project Structure

```
handwriting-generator/
├── models/
│   ├── scrabblegan.py       # GAN generator + discriminator
│   ├── diffusion.py         # DDPM U-Net denoiser + noise schedule
│   └── style_encoder.py     # ResNet18-based style extractor
├── utils/
│   ├── dataset.py           # IAM, CROHME, and user-sample dataset loaders
│   ├── losses.py            # All loss functions (L1, perceptual, CTC, denoising, style)
│   ├── metrics.py           # FID, CER, WER, SSIM evaluation metrics
│   └── latex_parser.py      # LaTeX → Unicode text conversion
├── train_gan.py             # GAN training entry point
├── train_diffusion.py       # Diffusion model training entry point
├── inference.py             # Generate handwriting from text/LaTeX
├── evaluate.py              # Compute evaluation metrics on generated images
├── prepare_splits.py        # Create writer-independent train/val/test splits
├── check_dataset.py         # Verify dataset loads correctly
└── requirements.txt
```

---

## High-Level Pipeline

```
INPUT TEXT (or LaTeX)
        │
        ▼
  [LaTeX Parser]          ← converts math commands to Unicode if needed
        │
        ▼
  [Character Encoding]    ← maps each character to an integer index (95-char ASCII vocab)
        │
        ▼
  [Style Encoder]         ← extracts writer style from reference handwriting images
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  [ScrabbleGAN Path]                     [Diffusion Model Path]
  BiLSTM + Upsampling                    Text Transformer + U-Net DDPM
  + AdaIN conditioning                   + FiLM / Cross-Attention conditioning
        │                                          │
        └──────────────────────────────────────────┘
                          │
                          ▼
               GENERATED HANDWRITING IMAGE (PNG)
                          │
                          ▼
                   [Evaluation]
           FID · CER · WER · SSIM
```

Both models are trained on the IAM dataset and can optionally be conditioned on user-provided handwriting samples to replicate a specific person's handwriting style.

---

## Step 1: Dataset Preparation

**File:** `prepare_splits.py`

The IAM dataset contains thousands of handwritten word images annotated with writer IDs and transcriptions in XML metadata. Before training, we split the dataset in a **writer-independent** manner — meaning no single writer's samples appear in more than one split. This is critical for evaluating true generalization.

**Process:**
1. Parse all IAM XML files to extract writer IDs (e.g., `a01`, `b03`).
2. Collect all word image paths grouped by writer.
3. Randomly shuffle writers, then assign:
   - 80% of writers → **train split**
   - 10% of writers → **validation split**
   - 10% of writers → **test split**
4. Write the resulting file paths and labels to `train.txt`, `val.txt`, `test.txt`.

This guarantees the model cannot memorize specific writers' styles, forcing it to learn general handwriting generation.

---

## Step 2: Data Loading & Preprocessing

**File:** `utils/dataset.py`

Three dataset classes handle different data sources:

### IAMDataset
The primary dataset. Each sample is a grayscale word image + its text transcription.

**Preprocessing pipeline:**
1. Load the PNG image in grayscale mode (`PIL Image, mode='L'`).
2. Resize to a fixed height (default **64 px**) while preserving the aspect ratio: `new_width = min(H_target × (orig_W / orig_H), max_width)`. This uses `BICUBIC` interpolation.
3. Convert to a PyTorch tensor and normalize pixel values from `[0, 255]` to `**[-1, 1]**` using: `(pixel / 255 - 0.5) / 0.5`.
4. **Training augmentation only:** Random affine transforms (±2° rotation, ±2% translation, 95–105% scale) and ColorJitter (±30% brightness/contrast) to improve robustness.
5. Encode the text label as a sequence of character indices using a 95-character ASCII vocabulary (space + printable symbols).

**Collation:** Since word images have different widths, a custom `collate_fn` pads all images in a batch to the maximum width using `-1.0` (white background in normalized space).

### UserSamplesDataset
Loads user-provided handwriting images paired with `.txt` transcript files. Used to supply **style reference images** during inference so the model can mimic a specific person's handwriting.

### CROHMEDataset
Handles math expression handwriting data (PNG renders paired with `.txt` labels). Uses the same image preprocessing pipeline as IAMDataset.

---

## Step 3: Style Encoding

**File:** `models/style_encoder.py`

Both models share the same style encoder. Its job is to extract a compact vector representation of a writer's handwriting style from one or more reference images.

### Architecture
- **Backbone:** ResNet18 pretrained on ImageNet, modified to accept single-channel (grayscale) input by replacing the first `Conv2d` with a `1×1` projection.
- **Global pooling:** The spatial feature map output from ResNet18 is average-pooled to a `(B, 512)` vector.
- **Projection head:** A two-layer MLP projects the 512-dim features down to a **256-dim style embedding**.

### Multi-image Aggregation
When multiple reference images are available for a writer (e.g., 8 samples), each is individually encoded and the resulting embeddings are **mean-pooled** into a single 256-dim vector. This makes the style representation more robust.

### AdaIN (Adaptive Instance Normalization)
The 256-dim style embedding is injected into the generator/denoiser via AdaIN:

```
AdaIN(x, style) = σ(style) × ((x − μ(x)) / σ(x)) + μ(style)
```

where `μ` and `σ` are learned linear projections of the style embedding. This modulates the spatial feature maps in the generator to match the target style's statistics.

---

## Step 4a: ScrabbleGAN (GAN Path)

**File:** `models/scrabblegan.py`

ScrabbleGAN is inspired by the original ScrabbleGAN paper (Fogel et al., 2020). It generates a full word image in a single forward pass by treating each character independently and combining them spatially.

### Generator Architecture

1. **Character Embedding:** Each character index is mapped to a 64-dim learned embedding.
2. **Noise Concatenation:** A random noise vector (`z`, 64-dim) is concatenated to each character embedding → 128-dim per character.
3. **BiLSTM:** A 2-layer Bidirectional LSTM (hidden=128) processes the sequence of character vectors. This allows the generator to model inter-character dependencies (e.g., ligatures, spacing). Output: `(B, T, 256)`.
4. **Spatial Reshaping:** The LSTM output is reshaped to `(B, channels, 4, T)` — a small spatial grid with one column per character.
5. **Upsampling Blocks (×4):** Each block applies:
   - `ConvTranspose2d` with stride 2 (doubles spatial resolution)
   - `AdaIN` style conditioning (injects writer style)
   - `ReLU` activation
   
   After 4 blocks: spatial size goes from `(4, T)` → `(64, T × char_width)`.
6. **Output Conv:** A final `Conv2d + Tanh` produces the output image `(B, 1, 64, W)` in `[-1, 1]`.

### Discriminator Architecture

A **PatchGAN** discriminator with a dual-head output:

1. **Feature Extractor:** 4 strided `Conv2d` blocks with BatchNorm and LeakyReLU(0.2), progressively increasing channels: `1 → 64 → 128 → 256 → 512`.
2. **Adversarial Head:** A `Conv2d → 1` layer that outputs a spatial real/fake score map. Using a patch-based output means the discriminator judges local texture realism, not just global appearance.
3. **CTC Recognizer Head:** A shared feature map passes through a small CNN + BiLSTM to produce per-frame character logits. A CTC loss is applied to encourage the generated images to be legible (text-recognizable).

### Why ScrabbleGAN?
GANs produce images in a single forward pass — fast inference. The CTC head in the discriminator acts as a "legibility supervisor," ensuring generated text is readable while the adversarial loss drives photorealism.

---

## Step 4b: Diffusion Model (DDPM Path)

**File:** `models/diffusion.py`

The diffusion model follows the DDPM framework (Ho et al., 2020), taking inspiration from One-DM and DiffusionPen. Instead of generating images in one step, it **iteratively denoises** a random Gaussian noise image over 1000 steps, guided by text and style conditioning.

### Noise Schedule

A **linear beta schedule** is used:
- `β_t` linearly increases from `1e-4` to `0.02` over `T = 1000` timesteps.
- Derived quantities are precomputed: `ᾱ_t = ∏ᵢ (1 − βᵢ)`, `√ᾱ_t`, `√(1 − ᾱ_t)`, posterior variance.

**Forward process** (training — adding noise):
```
x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)
```

**Reverse process** (inference — removing noise):
```
x_{t−1} = (1/√αt) · (x_t − (βt / √(1−ᾱt)) · ε_θ(x_t, t)) + σ_t · z
```

### Text Encoder

Converts the input character sequence to a continuous representation:
1. **Character Embedding:** Learnable `(vocab_size, 128)` embedding table + sinusoidal positional encoding.
2. **Transformer Encoder:** 3 layers, 4 attention heads, 512-dim feedforward. Processes the full sequence.
3. **Outputs two representations:**
   - Per-token context: `(B, T, 256)` — used for cross-attention in the U-Net
   - Pooled summary: `(B, 256)` — concatenated with style embedding for FiLM conditioning

### U-Net Denoiser

The core model that predicts the noise `ε_θ(x_t, t, text, style)`:

**Time Embedding:**
- Sinusoidal encoding: `emb_t = [sin(ω_i · t), cos(ω_i · t)]` for logarithmically-spaced frequencies
- Passed through a 2-layer MLP to produce a 256-dim embedding

**Conditioning:**
- Style embedding (256-dim) + text pooled (256-dim) are concatenated → 512-dim
- Each ResidualBlock modulates its features via **FiLM (Feature-wise Linear Modulation)**:
  ```
  h = h × (1 + γ) + β
  ```
  where `γ, β` are learned linear projections of the 512-dim conditioning vector.

**Architecture (4 resolution levels):**
```
Input x_t  →  [DownBlock 0] → skip₀
              [DownBlock 1] → skip₁
              [DownBlock 2] → skip₂  ← cross-attention with text tokens
              [DownBlock 3] → skip₃  ← cross-attention with text tokens
              [Bottleneck ResBlock]
              [UpBlock 3] ← skip₃
              [UpBlock 2] ← skip₂
              [UpBlock 1] ← skip₁
              [UpBlock 0] ← skip₀
              Conv → predicted noise
```

Channel multipliers: `[1, 2, 4, 8]` — deeper layers have more channels for richer representations.

**Cross-Attention** (at coarse resolutions):
- Spatial features `(B, C, H', W')` are flattened to `(B, H'W', C)` as queries
- Text context `(B, T, text_dim)` provides keys and values
- This directly aligns spatial regions of the generated image with specific characters

### Why Diffusion?
Diffusion models generally produce higher-quality, more diverse outputs than GANs and are easier to train stably. The tradeoff is slow inference (1000 denoising steps vs. one GAN forward pass).

---

## Step 5: Training

### GAN Training (`train_gan.py`)

The GAN uses an alternating optimization scheme:

**Discriminator update** (runs `n_critic` times per generator update):
1. Generate fake images with the generator (no grad).
2. Score real and fake images with the discriminator.
3. Compute **hinge loss**: `max(0, 1 − D(real)) + max(0, 1 + D(fake))`
4. Compute **CTC recognition loss** on the discriminator's text head.
5. Backpropagate and update discriminator weights.

**Generator update** (once per `n_critic` discriminator steps):
1. Generate fake images.
2. Compute adversarial loss: `-mean(D(fake))` (wants discriminator to score fake as real)
3. Compute **L1 reconstruction loss** against real images.
4. Compute **perceptual loss** (VGG feature matching at layers 3, 8, 15, 22).
5. Backpropagate combined loss: `λ₁·L1 + λ₂·L_perceptual + λ₃·L_adv`

**Logging:** Every N steps, sample fixed test texts, log generated images to TensorBoard, and save best checkpoint based on generator loss.

### Diffusion Training (`train_diffusion.py`)

1. Sample a random timestep `t ~ Uniform(0, T)` for each image in the batch.
2. Corrupt the real image: `x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε`
3. Feed `(x_t, t, text_tokens, style_refs)` to the U-Net to predict `ε_θ`.
4. Compute **denoising MSE loss**: `||ε − ε_θ||²`
5. Optionally add **Gram matrix style loss** (VGG texture matching) and **CTC text loss**.
6. Backpropagate with gradient clipping (`max_norm = 1.0`).
7. Use **cosine annealing** learning rate schedule.

---

## Step 6: Inference

**File:** `inference.py`

Generates a handwriting image for a given text string:

1. **Load model checkpoint** (contains saved weights + architecture config).
2. **Load style references** from user samples directory (up to 8 images), or use random noise if no style is provided.
3. **Encode the input text** to character indices.
4. **Generate image:**
   - *GAN path:* Single forward pass through the generator → immediate output.
   - *Diffusion path:* Start from `x_T ~ N(0, I)`, then run the full 1000-step reverse denoising loop guided by text + style conditioning.
5. **Post-process:** Denormalize from `[-1, 1]` to `[0, 255]` and save as a PNG file.

---

## Step 7: Evaluation

**File:** `evaluate.py` and `utils/metrics.py`

Four metrics are computed to assess model quality:

| Metric | What It Measures | Direction |
|--------|-----------------|-----------|
| **FID** (Fréchet Inception Distance) | Visual realism — how close generated image statistics are to real image statistics in InceptionV3 feature space | Lower is better |
| **CER** (Character Error Rate) | OCR accuracy — whether the generated handwriting is legible at the character level | Lower is better |
| **WER** (Word Error Rate) | OCR accuracy at the word level (split by whitespace) | Lower is better |
| **SSIM** (Structural Similarity Index) | Structural and texture similarity to reference style images | Higher is better |

### FID Detail
FID extracts 2048-dim features from the `pool3` layer of InceptionV3 for both real and generated images, then computes:
```
FID = ||μ_real − μ_fake||² + Tr(Σ_real + Σ_fake − 2·(Σ_real · Σ_fake)^0.5)
```

### CER / WER Detail
An OCR model reads the generated images. The predicted text is compared to the ground-truth label using **edit distance** (Levenshtein distance) normalized by the true length.

---

## LaTeX Support

**File:** `utils/latex_parser.py`

To support mathematical handwriting (e.g., for CROHME dataset or user-provided math expressions), a rule-based LaTeX-to-Unicode converter is implemented. The conversion happens before character encoding.

**Conversion rules:**
- Greek letters: `\alpha` → `α`, `\beta` → `β`, `\Sigma` → `Σ`, etc.
- Operators: `\cdot` → `·`, `\leq` → `≤`, `\int` → `∫`, `\infty` → `∞`
- Fractions: `\frac{d}{dx}` → `(d/dx)` via recursive brace parsing
- Superscripts: `e^x` → `eˣ` (Unicode superscript for single char), `e^{xy}` → `e^(xy)`
- Subscripts: `x_0` → `x₀` (Unicode subscript for single digit)
- Square roots: `\sqrt{x}` → `√(x)`
- Cleanup: Remove remaining unknown `\commands` and stray braces

**Example:**
```
Input:   \frac{d}{dx} e^x = e^x + \alpha
Output:  (d/dx) eˣ = eˣ + α
```

---

## Loss Functions

**File:** `utils/losses.py`

| Loss | Used By | Formula / Description |
|------|---------|----------------------|
| **L1** | GAN Generator | `E[|fake − real|]` — pixel-level reconstruction |
| **Perceptual** | GAN Generator | VGG16 feature matching at layers [3, 8, 15, 22] — preserves high-level structure |
| **Adversarial (Hinge)** | GAN Discriminator | `max(0, 1 − D(real)) + max(0, 1 + D(fake))` |
| **CTC** | GAN Discriminator + Diffusion | Connectionist Temporal Classification for text legibility |
| **Denoising MSE** | Diffusion | `||ε − ε_θ||²` — core diffusion training objective |
| **Style (Gram)** | Diffusion | Gram matrix matching on VGG features — preserves texture |

---

## Dependencies

```
torch >= 2.0
torchvision
numpy
Pillow
scipy
scikit-image
einops
sympy
opencv-python
lxml
editdistance
regex
tensorboard
```

---

## Summary

This project implements a complete pipeline for **personalized handwriting generation**:

- **Data:** IAM handwriting database with writer-independent splits
- **Style:** ResNet18 encoder extracts writer style embeddings; AdaIN injects them into generation
- **Two generation architectures:** ScrabbleGAN (fast, GAN-based) and Diffusion (high-quality, DDPM-based)
- **Text conditioning:** Character embeddings + BiLSTM (GAN) or Transformer + cross-attention (Diffusion)
- **Evaluation:** FID, CER, WER, SSIM cover both visual quality and legibility
- **LaTeX:** Full math expression support via rule-based Unicode conversion

The dual-architecture design allows direct comparison between GAN and diffusion approaches on the same data and evaluation protocol.

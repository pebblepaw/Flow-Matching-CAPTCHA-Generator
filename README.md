# CAPTCHA Generation with Flow Matching

This project implements a **Conditional Rectified Flow / Flow Matching** model to generate **CAPTCHA characters** (0–9, a–z), which are then composed into full CAPTCHA images. The core model is a **52.5M parameter U-Net** with attention, **Classifier-Free Guidance (CFG)**, and **EMA**.

---

## Table of Contents

1. [Project Overview](#-project-overview)  
2. [Key Features](#-key-features)  
3. [Results Summary](#-results-summary)  
4. [Model Architecture](#-model-architecture)  
5. [Requirements](#-requirements)  
6. [Installation (Local)](#-installation-local)  
7. [Dataset Preparation](#-dataset-preparation)  
8. [Training](#-training)  
9. [Generating CAPTCHAs](#-generating-captchas)  
10. [Evaluation](#-evaluation)  
11. [Training History & V2 Improvements](#-training-history--v2-improvements)  
12. [Expected Performance & Future Work](#-expected-performance--future-work)  
13. [References](#-references)  

---

## Project Overview

This project explores **Rectified Flow Matching** as an alternative to diffusion for image generation:

- Learn a **velocity field** that transports Gaussian noise to real character images along **straight-line paths** in image space.
- Train on **single characters** extracted from CAPTCHAs, then assemble them into full CAPTCHA images.
- Use **conditional flow matching** to control which character is generated.

Rectified flow is:

-  **Deterministic** (ODE, not SDE)  
-  **Simpler** target (constant velocity)  
-  **Faster** sampling (≈150 ODE steps instead of 1000+ diffusion steps)  

---

## Key Features

- **Model type**: Conditional Rectified Flow / Flow Matching.
- **Character-level generation**:  
  - Generates individual characters **0–9, a–z**.  
- **Advanced conditioning**:  
  - **Cross-attention** conditioning for spatially-aware class control.  
- **Classifier-Free Guidance**:  
  - 10% label dropout during training.  
  - Guidance scale `w = 3.0` at sampling.  
- **EMA weights**:  
  - **Exponential Moving Average** (decay = 0.9999) for stable, high-quality generation.  
- **Image format**:  
  - Characters are generated and then composed into **80×640 CAPTCHA images**.  

---

## Results Summary

| Metric              | Value                                   |
|---------------------|-----------------------------------------|
| **Model Size**      | 52.5M parameters                        |
| **Training Epochs** | 196 (early-stopped with patience = 50)  |
| **Best Val Loss**   | 0.0422                                  |
| **Test/Train Ratio**| 0.89× (good generalization)             |
| **Training Time**   | ~12 hours (NVIDIA TITAN RTX, SoC cluster) |
| **Sampling Steps**  | 150 (CFG scale 3.0)                     |

**Qualitative**:  
- Clear, readable characters with diverse styles.  
- Low artifacts and consistent structure.

### Training Progression

![Training Progression](cs4243_progression_montage.png)

*Visual progression of the generated "CS4243" across 8 training checkpoints*

---

## Model Architecture

### Conditional U-Net

**Input**: `(x_t, t, c)`  
- `x_t` – noised character image at time `t`  
- `t` – scalar flow time (0→1), embedded via sinusoidal time embedding  
- `c` – character class embedding (0–35 for 0–9, a–z)

**High-level structure**

```text
Input (3×H×W)
  ↓
[Time Embedding + Class Embedding]

Input Conv (3 → 96)

ENCODER
├─ Down1: ResBlocks(96)   → Downsample (96 → 192)
├─ Down2: ResBlocks(192)  → Downsample (192 → 384)
└─ Down3: ResBlocks(384)  → Downsample (384 → 768)

BOTTLENECK (8×8)
├─ ResBlock(768) + conditioning
├─ Multi-scale self-attention
├─ ResBlock(768) + conditioning
├─ Multi-scale self-attention
└─ Cross-attention for class conditioning

DECODER (with skip connections)
├─ Up3: Upsample(768 → 384) + ResBlocks(384)
├─ Up2: Upsample(384 → 192) + ResBlocks(192)
└─ Up1: Upsample(192 → 96)  + ResBlocks(96)

Output Conv (96 → 3)

Output: predicted velocity v_θ(x_t, t, c)
```

**Key design points**

* **Channels**: 96 → 192 → 384 → 768 (1.5× wider than V1).
* **Attention**: Multi-scale attention in the bottleneck for global structure + fine details.
* **Conditioning**: Cross-attention allows the model to focus class information on relevant spatial regions.
* **CFG training**: 10% unconditional batches enable classifier-free guidance.
* **EMA**: Separate EMA weights used for generation.

---

## Requirements

* **Python** ≥ 3.10
* **PyTorch** ≥ 2.0 (with CUDA for GPU training)
* **CUDA** ≥ 12.1 (for GPU runs)
* ≥ 8GB RAM (CPU) or ≥ 4GB VRAM (GPU) for basic usage
* For full training with large batch sizes:

  * Recommended: ≥ 16GB VRAM (e.g. V100, 3090, H100)
  * 32GB+ system RAM for comfortable training

All Python dependencies are in `requirements.txt`.

---

## Installation (Local)

```bash
# Clone repository
git clone https://github.com/xplus2g4/CS4243-Project-AY2526.git
cd CS4243-Project-AY2526
git checkout flow-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
FlowModel/                          # Flow matching model
├── README.md                       # This documentation
├── train_character_model_v2.py     # Training script
├── generate_captchas_v2.py         # CAPTCHA generation
├── evaluate_captchas_v2.py         # Model evaluation
└── requirements.txt                # Python dependencies

data/characters/                    # Extracted character dataset (user provides)
├── train/                          # Training characters by class (0-9, a-z)
└── test/                           # Test characters by class

cs4243_progression_montage.png  # Visualisation of results across epochs
```

---

## Dataset Preparation

Unfortunately the **CAPTCHA dataset** is not public, you may find your own relevant dataset. 

1. Place your raw CAPTCHAs into:

   * `train/` – training CAPTCHAs
   * `test/` – test CAPTCHAs

2. Extract character images:

```bash
python3 extract_characters.py \
    --input_dir train \
    --output_dir data/characters
```

This creates:

* `data/characters/train/` – training characters by class
* `data/characters/test/` – test characters by class
* 36 classes: `0–9` (digits), `a–z` (lowercase)

Approx. numbers (dataset-dependent):

* ~1.4k images per class
* Total: 41,268 training characters, 10,305 test characters (in original setup)

---

## Training

```bash
# GPU
python3 train_character_model_v2.py \
    --train_dir data/characters/train \
    --test_dir data/characters/test \
    --epochs 300 \
    --batch_size 96 \
    --device cuda
```

---

## Generating CAPTCHAs

Once you have a trained model (e.g. `best_model.pt`):

```bash
python3 generate_captchas_v2.py \
    --checkpoint results/character_model_v2/checkpoints/best_model.pt \
    --num_captchas 100 \
    --captcha_length 4 \
    --output_dir generated_captchas \
    --device cuda \
    --num_steps 150 \
    --guidance_scale 3.0
```

**Important arguments**

* `--checkpoint` – path to trained model (EMA weights)
* `--num_captchas` – number of CAPTCHAs to generate
* `--captcha_length` – characters per CAPTCHA (usually 4)
* `--num_steps` – ODE steps (50–200; higher = better but slower)
* `--guidance_scale` – CFG strength (1.0 = no guidance, 3.0–5.0 = sharper)
* `--device` – `cuda` or `cpu`

**Output**

* `generated_captchas/captcha_00000.png` …
* `generated_captchas/labels.txt` – ground truth labels

Approx. generation speed:

* ~2 seconds per character on GPU
* ~30 seconds per character on CPU

---

## Evaluation

Use a separate character classifier (e.g. CharacterCNN) to evaluate generated CAPTCHAs:

```bash
python3 evaluate_captchas_v2.py \
    --classifier path/to/character_classifier.pt \
    --captchas_dir generated_captchas \
    --labels_file generated_captchas/labels.txt \
    --output evaluation_results_v2.json
```

**Metrics**

* Character-level accuracy
* CAPTCHA-level accuracy
* Per-class accuracy
* Confusion matrices / examples

---

## Training History & V2 Improvements

### V2 vs V1

| Component        | V1               | V2                | Improvement                                |
| ---------------- | ---------------- | ----------------- | ------------------------------------------ |
| Parameters       | 31.7M            | 52.5M             | +66% capacity                              |
| Channels         | 64→128→256→512   | 96→192→384→768    | 1.5× wider per stage                       |
| Conditioning     | Additive         | Cross-attention   | Spatially-aware class conditioning         |
| Attention        | 1 layer          | 2× multi-scale    | Better global + local features             |
| CFG              | No               | Yes (10% dropout) | Sharper, more class-faithful outputs       |
| EMA              | No               | Yes (0.9999)      | More stable sampling                       |
| Scheduler        | Cosine annealing | Warm restarts     | Helps escape local minima                  |
| Augmentation     | None             | Mild (±5°, color) | Better generalization                      |
| Sampling steps   | 50               | 150               | Higher quality CAPTCHAs                    |
| Training epochs  | 100              | 196               | Better convergence                         |
| Test/train ratio | 1.00×            | **0.89×**         | Improved generalization (less overfitting) |

---

## Expected Performance & Future Work

### Target performance (V2)

* Clear, recognizable characters
* Natural font and style diversity
* Minimal artifacts

**Target metrics**

* Character recognition accuracy: **> 85%**
* CAPTCHA recognition accuracy (4 chars correct): **> 50%**
* FID score: **< 50** (targeted)

### Future improvements

1. Experiment with training the model on entire CAPCTHAs instead of single characters, so the model can gain information from relative character height, and from observing the font for characters it can safely recognize.
2. Style conditioning (font, color, background).
3. Adding artefacts in the post-processing, including hairlines and more. 

---

### References

**Flow Matching & Rectified Flow**

* Lipman et al., *Flow Matching for Generative Modeling* (2023)
* Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* (2023)

**Architecture & Guidance**

* Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)
* Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022)
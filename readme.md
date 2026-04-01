# Emb2Heights: Urban Structure and Land Cover Prediction

This repository is a baseline for the **Emb2Heights challenge**. It trains and runs inference for a model that predicts sub-pixel land cover percentages (Building, Vegetation, Water) and continuous structure heights (nDSM) directly from Earth Observation embeddings. Predictions are saved as `.npy` files with **4 output channels**: `[% Building, % Vegetation, % Water, Height (m)]`.

## Project Overview

Predicting urban morphology from satellite imagery is challenging: building footprints are sparse, and height values operate on a different scale than land-cover probabilities. This project addresses these challenges through a composite loss with **4 terms**:

- **MAE** (with background/foreground split): direct pixel-level regression.
- **SSIM + Gradient Loss**: enforces sharp structural boundaries on land-cover channels.
- **Tversky Loss**: penalizes false negatives heavily, forcing the model to capture sparse building footprints (α=0.3, β=0.7).
- **Structure-Boosted Height Loss**: height errors on building pixels are penalized 2x more than background pixels.

Training is further stabilized with AdamW (weight decay) and gradient clipping to prevent collapse on complex urban patches.

---

## Repository Structure

```text
emb2heights_baselines/
├── core/
│   ├── __init__.py
│   ├── model.py        # LightUNet + Decoder model factory
│   ├── dataset.py      # Dataset classes + embedding/label pairing utilities
│   └── losses.py       # ImprovedCompositeLoss (MAE, SSIM, Gradient, Tversky)
├── train.py            # Training entrypoint (fully CLI-configurable)
├── predict.py          # Inference entrypoint (loads checkpoint, saves .npy predictions)
├── environment.yml     # Conda environment definition
├── readme.md
└── runs/               # Auto-generated experiment outputs
    └── <experiment_name>/
        ├── model_best.pth
        ├── model_last.pth
        ├── loss_curve.png
        ├── training_params.txt
        ├── visualizations/
        └── predictions/
```

---

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate emb2heights
```

---

## Model Architecture

Architecture is selected via `--model-type`:

| Value | Description |
|---|---|
| `lightunet` | Lightweight encoder-decoder with skip connections |
| `decoder` | Transposed-convolution decoder |
| `decoder_residual` | Deeper decoder with residual blocks + global embedding skip fusion (recommended for high-channel embeddings) |
| `auto` | Selects `decoder` when input channels = 768, otherwise `lightunet` |

**Output**: a 4-channel tensor — `[0: % Building, 1: % Vegetation, 2: % Water, 3: Height (m)]`.

**Loss function**: `ImprovedCompositeLoss` with 4 terms — see [Project Overview](#project-overview).

---

## Training

Run training from the CLI — no file edits needed.

```bash
python train.py \
    --model-type decoder_residual \
    --train-embeddings-dir /path/to/train/embeddings \
    --train-targets-dir /path/to/train/labels \
    --test-embeddings-dir /path/to/test/embeddings \
    --test-targets-dir /path/to/test/labels \
    --experiment-name my_run \
    --epochs 30 \
    --batch-size 8 \
    --patch-size 256
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--model-type` | `auto` | Architecture: `auto`, `lightunet`, `decoder`, `decoder_residual` |
| `--train-embeddings-dir` | — | Path to training embedding `.tif` files |
| `--train-targets-dir` | — | Path to training label `.tif` files |
| `--test-embeddings-dir` | — | Path to test embeddings (used for post-training visualization) |
| `--test-targets-dir` | — | Path to test labels (used for post-training visualization) |
| `--experiment-name` | `terramid_run02` | Subfolder name under `./runs/` |
| `--epochs` | `30` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--patch-size` | `256` | Spatial crop size for dataset loader |

Outputs are written to `./runs/<experiment_name>/`: hyperparameter log, `model_best.pth`, `model_last.pth`, loss curve, and sample visualizations.

---

## Inference

Load a trained checkpoint and save predictions as `.npy` files (shape `[4, H, W]`, channels: building %, vegetation %, water %, height in meters).

```bash
python predict.py \
    --experiment-name my_run \
    --model-type decoder_residual \
    --test-embeddings-dir /path/to/test/embeddings \
    --test-targets-dir /path/to/test/labels
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--experiment-name` | `terramind_decoder_run01` | Experiment folder under `--base-dir` |
| `--base-dir` | `./runs` | Root directory of experiment folders |
| `--model-type` | `decoder_residual` | Architecture (must match training) |
| `--model-path` | `<base-dir>/<experiment-name>/model_best.pth` | Path to `.pth` checkpoint |
| `--test-embeddings-dir` | required | Directory with embedding `.tif` files |
| `--test-targets-dir` | required | Directory with label `.tif` files (used only for file pairing) |
| `--predictions-dir` | `<base-dir>/<experiment-name>/predictions` | Output directory for `.npy` files |
| `--patch-size` | `256` | Spatial crop size |
| `--max-samples` | `0` (all) | Limit inference to N samples |

Each output file is named `pred_<core_id>.npy` and contains a `float32` array of shape `[4, H, W]`:
- Channel 0: Building coverage (0–1)
- Channel 1: Vegetation coverage (0–1)
- Channel 2: Water coverage (0–1)
- Channel 3: Normalized surface height in meters
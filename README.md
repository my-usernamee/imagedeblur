# Multi-Stage Image Deblurring (V2_SAM + CSFF)

PyTorch implementation of a **3-stage** U-Net deblurring network that combines Supervised Attention Modules (SAM) with Sobel edge guidance and Cross-Stage Feature Fusion (CSFF). [file:1]  
The included script is notebook-export / Kaggle-first: it bundles dataset setup, training, checkpointing, and a small qualitative visualization on test samples. [file:1]

---

## What’s inside

- **GoPro paired dataset loader** (blur/sharp) using the Kaggle dataset layout under `DBlur/Gopro/...`. [file:1]
- **3-stage architecture**: each stage is a U-Net that produces an RGB prediction + intermediate features. [file:1]
- **SAM (Supervised Attention Module)**: uses Sobel edges computed from the *previous stage output* to guide refinement. [file:1]
- **CSFF (Cross-Stage Feature Fusion)**: fuses features across stages using a 1×1 convolution. [file:1]
- **Training utilities**:
  - Multi-stage weighted L1 loss with weights `[0.5, 0.5, 1.0]`. [file:1]
  - Adam optimizer (`lr = 2e-4`) + cosine annealing scheduler. [file:1]
  - Best checkpoint saving + periodic checkpoints. [file:1]
- **Inference helper**: `deblur_image()` runs prediction at 256×256 and resizes output back to the original input size. [file:1]

---

## File

- `v2_sam_-_csff.py` — end-to-end pipeline (deps install, dataset, model, training loop, and visualization). [file:1]

---

## Requirements

The script installs dependencies inline (Kaggle/Colab style): `torch`, `torchvision`, `pillow`, `tqdm`, `matplotlib`. [file:1]  
CUDA is recommended; the script selects `cuda` if available, otherwise falls back to CPU. [file:1]

---

## Dataset

Default dataset root used in the script: [file:1]

- `/kaggle/input/a-curated-list-of-image-deblurring-datasets` [file:1]

Expected GoPro structure: [file:1]

- `/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/train/blur` [file:1]
- `/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/train/sharp` [file:1]
- `/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/test/blur` [file:1]
- `/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/test/sharp` [file:1]

Running locally: update the `path = ...` variable in the script to point to your dataset root. [file:1]

---

## How to run

### Kaggle / Colab (as-is)
1. Attach the Kaggle dataset input (or ensure the dataset path matches your environment). [file:1]
2. Run `v2_sam_-_csff.py` top-to-bottom. [file:1]

### Local (recommended cleanup)
This script is a notebook export and contains Kaggle/Colab-specific lines (e.g., `!pip install ...`). [file:1]  
For a clean local run:

- Remove shell magics like `!pip ...`. [file:1]
- Install dependencies normally (example):
[file:1]
- Replace Kaggle paths with local filesystem paths. [file:1]

---

## Training settings (defaults)

- Image size: **256×256** (dataset resizes during loading). [file:1]
- Batch size: **8**. [file:1]
- Epochs: **30**. [file:1]
- Optimizer: **Adam**, `lr = 2e-4`. [file:1]
- Scheduler: **CosineAnnealingLR**, `T_max = 50`. [file:1]
- Loss: weighted multi-stage **L1** with weights `[0.5, 0.5, 1.0]`. [file:1]

---

## Outputs

Checkpoints: [file:1]
- Best model: `multistage_deblur_best.pth`. [file:1]
- Periodic checkpoints: `checkpoint_epoch_10.pth`, `checkpoint_epoch_20.pth`, ... [file:1]

Visualization: [file:1]
- `deblur_results.png` (columns: blurry / ground truth / deblurred). [file:1]

---

## Notes / known issues

- The file is auto-generated from a notebook and may need formatting fixes before “production” use (e.g., stray notebook remnants). [file:1]
- The forward pass currently calls `stage2` and `stage3` more than once (once for features and again for outputs), which works but is inefficient; it can be refactored to compute each stage once and reuse outputs/features. [file:1]

---

## License

No license is included yet—add one if you plan to publish or reuse this code (e.g., MIT / Apache-2.0). [file:1]

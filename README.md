# Semiconductor Inspection Image Super-Resolution

This repository contains the training and inference pipeline for super-resolving `.npy` images using the Hybrid Attention Transformer (HAT) architecture, optimized for execution on consumer hardware.

## Files Included
- `main.py`: The wrapper for the `HATModel` that handles memory-efficient tiled inference and window-size compatible padding.
- `train.py`: The main entry point script used for model training, seamlessly integrating with the BasicSR framework.
- `inference.py`: The evaluation script designed to process the `.npy` noisy low-resolution inputs and output standard `(256, 256)` float32 arrays.
- `npy_dataset.py`: A custom PyTorch `Dataset` loader designed to ingest and parse single-channel 2D `.npy` files effectively for the BasicSR pipeline.
- `train_hat.yml`: The hyperparameter and environmental configuration optimized specifically for fitting HAT training into a 6GB VRAM constraint (batch size 1, gt size 64).

## Requirements
- Python 3.10+
- PyTorch (CUDA enabled)
- BasicSR (`pip install basicsr`)
- EINOPs (`pip install einops`)
- Official HAT repository in path.

## How to Run Inference (Evaluation Script)
The standalone evaluation script is `inference.py`. It accepts paths for test images and output directories, and runs inference automatically without manual edits.

Run the evaluation script using the following command:
```bash
python inference.py --test_dir "path/to/test_images" --output_dir "path/to/output_images"
```
*(Note: It automatically loads the trained model from `weights/net_g_40000.pth` and applies it to all `.npy` files found in `--test_dir`)*

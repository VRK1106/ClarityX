# AI-Based Restoration of Degraded Semiconductor Images (KLA Problem Statement)

This repository contains the official evaluation pipeline and super-resolution restoration model for degraded semiconductor inspection images using the **Hybrid Attention Transformer (HAT)** architecture.

---

## 📁 Repository Structure

```
submission_repo/
├── run.py                 # Primary entry point script required for evaluation
├── requirements.txt       # Dependencies with version specifications
├── README.md              # Setup and execution documentation
├── main.py                # Wrapper for HATModel (handles window padding & tiled inference)
├── npy_dataset.py         # PyTorch Dataset loader for single-channel .npy arrays
├── train_hat.yml          # Hyperparameter and network configuration file
├── train.py               # Model training script integrating with BasicSR
├── inference.py           # Legacy wrapper delegating to run.py
├── models/
│   └── net_g_40000.pth    # Trained PyTorch generator checkpoint
└── HAT/                   # Hybrid Attention Transformer architecture code
```

---

## ⚡ Setup & Dependencies

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (evaluated on CUDA 11.8 / CUDA 12.1)

### Installation
Install all required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note:** All required model weights (`models/net_g_40000.pth`) and supporting code modules (`HAT/`) are pre-packaged. The solution runs **completely offline** without requiring internet access, API keys, additional model downloads, or user interaction.

---

## 🚀 Execution Instructions

Run the restoration pipeline using the entry point `run.py`:

```bash
python run.py <input-dir> <output-dir>
```

### Example Usage:
```bash
python run.py path/to/input_noisy_npy path/to/output_restored_npy
```

### Script Specifications (`run.py`)
- **Input:** Reads all `.npy` degraded image files from `<input-dir>`.
- **Output Directory:** Automatically creates `<output-dir>` if it does not exist.
- **Output Format:** Generates standard 2D grayscale `.npy` arrays with shape `(H, W)`.
- **Filename Consistency:** Preserves the exact input filename for every generated output file.
- **Value Bounds:** All output pixel values are cleaned of `NaN` / `Inf` values and strictly clipped within `[0.0, 1.0]`.
- **Resolution:** Super-resolves 2x (e.g. 128x128 degraded input to 256x256 high-resolution output).

---

## 🛠️ Verification & Compliance Summary

- ✅ Entry point: `python run.py <input-dir> <output-dir>`
- ✅ Single-channel `.npy` input & output processing supported
- ✅ Automatic output directory creation
- ✅ Output values strictly in `[0, 1]` range with zero `NaN`/`Inf`
- ✅ Fully offline GPU inference compatible

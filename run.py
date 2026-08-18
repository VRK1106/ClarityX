import os
import sys
import glob
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp

# Add HAT directory to python path
script_dir = osp.dirname(osp.abspath(__file__))
hat_path = osp.join(script_dir, 'HAT')
if hat_path not in sys.path:
    sys.path.append(hat_path)

try:
    from hat.archs.hat_arch import HAT
    import main  # registers HATModel
except ImportError:
    try:
        from hat.archs.hat_arch import HAT
    except ImportError:
        print("Error: Could not import HAT architecture. Ensure the HAT module is in the python path.")
        sys.exit(1)


def run():
    parser = argparse.ArgumentParser(description="KLA Problem Statement - AI-Based Restoration of Degraded Images")
    parser.add_argument("input_dir", type=str, help="Path to input directory containing degraded .npy files")
    parser.add_argument("output_dir", type=str, help="Path to output directory where restored .npy files will be saved")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model weights (.pth)")
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML file")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Locate model checkpoint
    if args.model_path:
        model_path = args.model_path
    else:
        candidates = [
            osp.join(script_dir, "models", "net_g_40000.pth"),
            osp.join(script_dir, "weights", "net_g_40000.pth"),
            "models/net_g_40000.pth",
            "weights/net_g_40000.pth",
        ]
        model_path = None
        for cand in candidates:
            if osp.exists(cand):
                model_path = cand
                break

        if not model_path:
            print("Error: Could not locate model checkpoint in models/ or weights/.")
            sys.exit(1)

    # Locate config file
    if args.config:
        config_path = args.config
    else:
        config_path = osp.join(script_dir, "train_hat.yml")
        if not osp.exists(config_path):
            config_path = "train_hat.yml"

    if not osp.exists(config_path):
        print(f"Error: Could not find configuration file '{config_path}'.")
        sys.exit(1)

    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net_config = opt["network_g"]
    net = HAT(**net_config)

    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "params_ema" in checkpoint:
            key = "params_ema"
        elif "params" in checkpoint:
            key = "params"
        else:
            key = None

        if key:
            net.load_state_dict(checkpoint[key], strict=True)
        else:
            net.load_state_dict(checkpoint, strict=True)
    else:
        net.load_state_dict(checkpoint, strict=True)

    net.eval()
    net.to(device)

    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"Found {len(npy_files)} .npy files in '{input_dir}' to process.")

    window_size = net_config.get("window_size", 16)
    scale = opt.get("scale", 2)

    with torch.no_grad():
        for file_path in tqdm(npy_files, desc="Restoring Images"):
            filename = os.path.basename(file_path)
            save_path = os.path.join(output_dir, filename)

            # Load .npy file
            img_np = np.load(file_path)
            orig_h, orig_w = img_np.shape[:2]

            if img_np.ndim == 2:
                img_np = np.expand_dims(img_np, axis=2)

            img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().unsqueeze(0).to(device)

            # Padding to align with window_size
            mod_pad_h = (window_size - orig_h % window_size) % window_size
            mod_pad_w = (window_size - orig_w % window_size) % window_size

            if mod_pad_h != 0 or mod_pad_w != 0:
                img_padded = torch.nn.functional.pad(img_tensor, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")
            else:
                img_padded = img_tensor

            # Forward pass
            output_padded = net(img_padded)

            # Remove padding
            if mod_pad_h != 0 or mod_pad_w != 0:
                out_h = (orig_h + mod_pad_h) * scale
                out_w = (orig_w + mod_pad_w) * scale
                final_h = out_h - mod_pad_h * scale
                final_w = out_w - mod_pad_w * scale
                output = output_padded[:, :, :final_h, :final_w]
            else:
                output = output_padded

            # Convert back to numpy (H, W)
            output_np = output.squeeze(0).cpu().numpy()
            output_np = np.transpose(output_np, (1, 2, 0))
            if output_np.ndim == 3 and output_np.shape[2] == 1:
                output_np = output_np.squeeze(2)

            # Ensure non-NaN/Inf and clip values strictly within [0, 1]
            output_np = np.nan_to_num(output_np, nan=0.0, posinf=1.0, neginf=0.0)
            output_np = np.clip(output_np, 0.0, 1.0)

            np.save(save_path, output_np)

    print(f"Inference completed successfully! Restored .npy files saved to '{output_dir}'.")


if __name__ == "__main__":
    run()

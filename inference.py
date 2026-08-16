import os
import glob
import math
import torch
import numpy as np
import yaml
from tqdm import tqdm
import sys
import argparse
import os.path as osp

# Add HAT to path if it exists locally, or rely on basicsr installation
sys.path.append(osp.abspath('HAT'))
try:
    from hat.archs.hat_arch import HAT
    import main  # registers HATModel
except ImportError:
    print("Please ensure the official HAT repository is in the working directory.")
    sys.exit(1)

def run_inference(model_path, input_folder, output_folder, config_path='train_hat.yml'):
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Instantiate architecture
    net_config = opt['network_g']
    net = HAT(**net_config)
    
    # Load model weights
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if 'params_ema' in checkpoint:
        key = 'params_ema'
    elif 'params' in checkpoint:
        key = 'params'
    else:
        key = None
        
    if key:
        net.load_state_dict(checkpoint[key], strict=True)
    else:
        net.load_state_dict(checkpoint, strict=True)
        
    net.eval()
    net.to(device)
    
    os.makedirs(output_folder, exist_ok=True)
    npy_files = sorted(glob.glob(os.path.join(input_folder, '*.npy')))
    print(f"Found {len(npy_files)} files in {input_folder} to process.")
    
    window_size = net_config.get('window_size', 16)
    scale = opt.get('scale', 2)
    
    with torch.no_grad():
        for file_path in tqdm(npy_files, desc="Running Inference"):
            filename = os.path.basename(file_path)
            save_path = os.path.join(output_folder, filename)
            
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
                img_padded = torch.nn.functional.pad(img_tensor, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
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
                
            # Convert back to numpy (H, W) or (H, W, C)
            output_np = output.squeeze(0).cpu().numpy()
            output_np = np.transpose(output_np, (1, 2, 0))
            if output_np.shape[2] == 1:
                output_np = output_np.squeeze(2)
                
            np.save(save_path, output_np)
            
    print(f"Inference completed! All output .npy files are saved to {output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script for KLA Degradation Super-Resolution")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to the directory containing NoisyLR .npy files")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the directory where restored outputs will be saved")
    parser.add_argument('--model_path', type=str, default='weights/net_g_40000.pth', help="Path to the trained PyTorch checkpoint (.pth)")
    parser.add_argument('--config', type=str, default='train_hat.yml', help="Path to the HAT YAML configuration file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_dir):
        print(f"Error: The test directory '{args.test_dir}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"Error: The model checkpoint '{args.model_path}' does not exist.")
        sys.exit(1)
        
    run_inference(args.model_path, args.test_dir, args.output_dir, args.config)

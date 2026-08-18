import os
import sys
import argparse
import os.path as osp

# Alias/wrapper pointing to run.py logic for backward compatibility
from run import run as main_run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script for KLA Degradation Super-Resolution")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to directory containing input .npy files")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to directory where output .npy files will be saved")
    parser.add_argument('--model_path', type=str, default=None, help="Path to trained PyTorch checkpoint (.pth)")
    parser.add_argument('--config', type=str, default=None, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Delegate to positional arguments format expected by run.py logic
    sys.argv = [
        sys.argv[0],
        args.test_dir,
        args.output_dir
    ]
    if args.model_path:
        sys.argv.extend(['--model_path', args.model_path])
    if args.config:
        sys.argv.extend(['--config', args.config])
        
    main_run()

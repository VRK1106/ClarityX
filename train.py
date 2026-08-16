import sys
import os.path as osp

# Import custom code
import main # This registers HATModel
import npy_dataset # This registers NpyDataset

# Add HAT to path so it can be imported properly
sys.path.append(osp.abspath('HAT'))
from hat.archs.hat_arch import HAT # This registers HAT

# Import BasicSR's training pipeline
from basicsr.train import train_pipeline

if __name__ == '__main__':
    # BasicSR's train_pipeline usually figures out root_path from the executing script
    # We can pass root_path as '.'
    train_pipeline('.')

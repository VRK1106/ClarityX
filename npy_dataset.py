import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY
import random

@DATASET_REGISTRY.register()
class NpyDataset(Dataset):
    """
    Custom Dataset for loading .npy image pairs (GT and NoisyLR).
    """
    def __init__(self, opt):
        super(NpyDataset, self).__init__()
        self.opt = opt
        
        self.gt_folder = opt.get('dataroot_gt', None)
        self.lq_folder = opt.get('dataroot_lq', None)
        self.phase = opt.get('phase', 'train')
        self.patch_size = opt.get('gt_size', None)
        
        if self.gt_folder is not None:
            self.gt_paths = sorted(glob.glob(os.path.join(self.gt_folder, '*.npy')))
        else:
            self.gt_paths = []
            
        if self.lq_folder is not None:
            self.lq_paths = sorted(glob.glob(os.path.join(self.lq_folder, '*.npy')))
        else:
            self.lq_paths = []
            
        # Optional validation
        if len(self.gt_paths) > 0 and len(self.lq_paths) > 0:
            assert len(self.gt_paths) == len(self.lq_paths), "GT and LQ datasets must have the same length"
            
    def __len__(self):
        return len(self.lq_paths)

    def augment(self, gt, lq):
        """Random flip and rotation."""
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        if hflip:
            gt = np.flip(gt, axis=1)
            lq = np.flip(lq, axis=1)
        if vflip:
            gt = np.flip(gt, axis=0)
            lq = np.flip(lq, axis=0)
        if rot90:
            gt = np.transpose(gt, (1, 0, 2))
            lq = np.transpose(lq, (1, 0, 2))
            
        return np.ascontiguousarray(gt), np.ascontiguousarray(lq)

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        lq_img = np.load(lq_path) # Expected shape (H, W, C)
        if lq_img.ndim == 2:
            lq_img = np.expand_dims(lq_img, axis=2)
            
        # Determine scale dynamically based on sizes if not provided
        # or rely on BasicSR's options
        scale = self.opt.get('scale', 2)

        gt_img = None
        gt_path = ""
        
        if len(self.gt_paths) > 0:
            gt_path = self.gt_paths[index]
            gt_img = np.load(gt_path)
            if gt_img.ndim == 2:
                gt_img = np.expand_dims(gt_img, axis=2)

            if self.phase == 'train' and self.patch_size is not None:
                # Random crop
                h_lq, w_lq, _ = lq_img.shape
                lq_patch_size = self.patch_size // scale
                
                if h_lq > lq_patch_size and w_lq > lq_patch_size:
                    top = random.randint(0, h_lq - lq_patch_size)
                    left = random.randint(0, w_lq - lq_patch_size)
                    
                    lq_img = lq_img[top:top + lq_patch_size, left:left + lq_patch_size, :]
                    gt_img = gt_img[top * scale:(top + lq_patch_size) * scale, 
                                    left * scale:(left + lq_patch_size) * scale, :]
                                    
                # Augmentation
                if self.opt.get('use_hflip', True) or self.opt.get('use_rot', True):
                    gt_img, lq_img = self.augment(gt_img, lq_img)
        
        # Convert HWC to CHW for PyTorch
        lq_img = np.transpose(lq_img, (2, 0, 1))
        lq_tensor = torch.from_numpy(lq_img).float()
        
        # BasicSR normalizes [0, 255] images by /255. 
        # Check if they are uint8
        if lq_img.dtype == np.uint8:
            lq_tensor = lq_tensor / 255.0

        if gt_img is not None:
            if gt_img.ndim == 2:
                gt_img = np.expand_dims(gt_img, axis=2)
            gt_img = np.transpose(gt_img, (2, 0, 1))
            gt_tensor = torch.from_numpy(gt_img).float()
            
            if gt_img.dtype == np.uint8:
                gt_tensor = gt_tensor / 255.0
                
            return {
                'lq': lq_tensor,
                'gt': gt_tensor,
                'lq_path': lq_path,
                'gt_path': gt_path
            }
        else:
            return {
                'lq': lq_tensor,
                'lq_path': lq_path
            }

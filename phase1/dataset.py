"""
ClarityGAN Dataset Module
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random


class GoProDataset(Dataset):
    def __init__(self, root_dir, phase='train', crop_size=256, augment=True):
        """
        GoPro Dataset for image deblurring
        
        Args:
            root_dir: Root directory containing train/test folders
            phase: 'train' or 'test'
            crop_size: Size to crop images to. Use 0 for full images
            augment: Whether to apply data augmentation
        """
        self.root_dir = os.path.join(root_dir, phase)
        self.blur_dir = os.path.join(self.root_dir, 'blur')
        self.sharp_dir = os.path.join(self.root_dir, 'sharp')
        self.blur_files = os.listdir(self.blur_dir)
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        subdir = self.blur_files[idx]
        blur_path = os.path.join(self.blur_dir, subdir)
        sharp_path = os.path.join(self.sharp_dir, subdir)

        # Load images and normalize
        blur_img = cv2.imread(blur_path) / 255.0
        sharp_img = cv2.imread(sharp_path) / 255.0

        h, w = blur_img.shape[:2]
        
        # Handle cropping
        if self.crop_size == 0:  # Use full image for testing
            blur_crop = blur_img
            sharp_crop = sharp_img
        else:
            # Ensure we don't go out of bounds
            max_x = max(0, w - self.crop_size)
            max_y = max(0, h - self.crop_size)
            
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0

            blur_crop = blur_img[y:y+self.crop_size, x:x+self.crop_size]
            sharp_crop = sharp_img[y:y+self.crop_size, x:x+self.crop_size]

        # Data augmentation
        if self.augment and random.random() > 0.5:
            blur_crop = np.fliplr(blur_crop)
            sharp_crop = np.fliplr(sharp_crop)
                
        # Convert to tensors
        blur_tensor = torch.from_numpy(blur_crop.transpose(2, 0, 1).copy()).float()
        sharp_tensor = torch.from_numpy(sharp_crop.transpose(2, 0, 1).copy()).float()

        blur_tensor = blur_tensor * 2 - 1
        sharp_crop = sharp_crop * 2 - 1
        
        return blur_tensor, sharp_tensor


def get_dataloaders(config):
    """
    Create train and test dataloaders
    """
    train_dataset = GoProDataset(
        config.DATASET_ROOT, 
        phase='train',
        crop_size=config.CROP_SIZE,
        augment=config.AUGMENT
    )
    
    test_dataset = GoProDataset(
        config.DATASET_ROOT,
        phase='test', 
        crop_size=0,  # Full images for testing
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, test_loader

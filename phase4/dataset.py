# dataset.py

import os
import random
import cv2
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class GoProDataset(Dataset):
    """
    PyTorch Dataset for GoPro deblurring.
    Supports an easy blur‐curriculum via gaussian_blur_prob.
    """
    def __init__(
        self,
        root_dir: str,
        phase: str = 'train',
        crop_size: int = 256,
        augment: bool = True,
        noise_std: float = 0.005,
        gaussian_blur_prob: float = 0.1
    ):
        self.blur_paths  = sorted(glob(os.path.join(root_dir, phase, 'blur',  '*.png')))
        self.sharp_paths = sorted(glob(os.path.join(root_dir, phase, 'sharp', '*.png')))
        self.crop_size   = crop_size
        self.augment     = augment
        self.noise_std   = noise_std
        self.gb_prob     = gaussian_blur_prob  # adjustable in training loop

        assert len(self.blur_paths) == len(self.sharp_paths), \
            f"Mismatch: {len(self.blur_paths)} blur vs {len(self.sharp_paths)} sharp"

    def __len__(self) -> int:
        return len(self.blur_paths)

    def __getitem__(self, idx: int):
        # load & normalize to [0,1]
        blur  = cv2.cvtColor(cv2.imread(self.blur_paths[idx]),  cv2.COLOR_BGR2RGB) / 255.0
        sharp = cv2.cvtColor(cv2.imread(self.sharp_paths[idx]), cv2.COLOR_BGR2RGB) / 255.0

        # random crop
        h, w, _ = blur.shape
        top  = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        blur  = blur[top:top+self.crop_size, left:left+self.crop_size]
        sharp = sharp[top:top+self.crop_size, left:left+self.crop_size]

        # synthetic Gaussian blur (curriculum)
        if self.augment and random.random() < self.gb_prob:
            k = random.choice([3,5,7])
            blur = cv2.GaussianBlur(blur, (k,k), 0)

        # to tensor
        blur  = torch.from_numpy(blur ).permute(2,0,1).float()
        sharp = torch.from_numpy(sharp).permute(2,0,1).float()

        if self.augment:
            # horizontal flip
            if random.random() < 0.5:
                blur, sharp = TF.hflip(blur), TF.hflip(sharp)
            # small rotation
            angle = random.uniform(-3, 3)
            blur, sharp = TF.rotate(blur, angle), TF.rotate(sharp, angle)
            # brightness & contrast jitter
            b, c = random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)
            blur = TF.adjust_brightness(blur, b)
            blur = TF.adjust_contrast(blur, c)
            # Gaussian noise
            noise = torch.randn_like(blur) * self.noise_std
            blur = (blur + noise).clamp(0.0, 1.0)

        return blur, sharp

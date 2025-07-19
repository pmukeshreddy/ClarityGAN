"""
ClarityGAN Configuration File
"""

import torch

class Config:
    # Dataset paths
    DATASET_ROOT = '/kaggle/input/gopro-image-deblurring-dataset/Gopro'
    
    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 200
    CROP_SIZE = 256
    
    # Learning rates
    GEN_LR = 1e-4
    DISC_LR = 1e-4
    
    # Loss weights
    LAMBDA_GAN = 1.0
    LAMBDA_L1 = 100.0
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model saving
    SAVE_INTERVAL = 10
    MODEL_SAVE_PATH = './checkpoints/'
    
    # Data loading
    NUM_WORKERS = 4
    
    # Augmentation
    AUGMENT = True

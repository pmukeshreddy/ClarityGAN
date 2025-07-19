"""
ClarityGAN Utility Functions
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_info(generator, discriminator):
    """Print model architecture information"""
    gen_total, gen_trainable = count_parameters(generator)
    disc_total, disc_trainable = count_parameters(discriminator)
    
    print("="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Generator:")
    print(f"  Total parameters: {gen_total:,}")
    print(f"  Trainable parameters: {gen_trainable:,}")
    print(f"Discriminator:")
    print(f"  Total parameters: {disc_total:,}")
    print(f"  Trainable parameters: {disc_trainable:,}")
    print(f"Total model parameters: {gen_total + disc_total:,}")
    print("="*60)


def visualize_results(blur_imgs, sharp_imgs, fake_imgs, num_samples=4, save_path=None):
    """
    Visualize comparison between blurry, sharp, and generated images
    
    Args:
        blur_imgs: Batch of blurry images (tensors)
        sharp_imgs: Batch of sharp ground truth images (tensors)
        fake_imgs: Batch of generated images (tensors)
        num_samples: Number of samples to show
        save_path: Path to save the visualization
    """
    # Ensure we don't try to show more samples than available
    num_samples = min(num_samples, blur_imgs.size(0))
    
    # Select subset of images
    blur_subset = blur_imgs[:num_samples].cpu()
    sharp_subset = sharp_imgs[:num_samples].cpu()
    fake_subset = fake_imgs[:num_samples].cpu()
    
    # Create figure
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_samples):
        # Convert tensors to numpy and transpose
        blur_np = blur_subset[i].numpy().transpose(1, 2, 0)
        sharp_np = sharp_subset[i].numpy().transpose(1, 2, 0)
        fake_np = fake_subset[i].numpy().transpose(1, 2, 0)
        
        # Clip values to [0, 1]
        blur_np = np.clip(blur_np, 0, 1)
        sharp_np = np.clip(sharp_np, 0, 1)
        fake_np = np.clip(fake_np, 0, 1)
        
        # Plot images
        axes[0, i].imshow(blur_np)
        axes[0, i].set_title(f'Blurry {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(fake_np)
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(sharp_np)
        axes[2, i].set_title(f'Ground Truth {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def save_sample_images(generator, test_loader, device, save_dir, num_samples=8):
    """
    Save sample deblurred images for visual inspection
    
    Args:
        generator: Trained generator model
        test_loader: Test data loader
        device: Device to run inference on
        save_dir: Directory to save sample images
        num_samples: Number of samples to save
    """
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    samples_saved = 0
    
    with torch.no_grad():
        for blur, sharp in test_loader:
            if samples_saved >= num_samples:
                break
                
            blur = blur.to(device)
            fake = generator(blur)
            
            # Save each image in the batch
            batch_size = blur.size(0)
            for i in range(min(batch_size, num_samples - samples_saved)):
                # Convert tensors to numpy
                blur_np = blur[i].cpu().numpy().transpose(1, 2, 0)
                fake_np = fake[i].cpu().numpy().transpose(1, 2, 0)
                sharp_np = sharp[i].numpy().transpose(1, 2, 0)
                
                # Clip and convert to uint8
                blur_np = np.clip(blur_np, 0, 1)
                fake_np = np.clip(fake_np, 0, 1)
                sharp_np = np.clip(sharp_np, 0, 1)
                
                blur_img = (blur_np * 255).astype(np.uint8)
                fake_img = (fake_np * 255).astype(np.uint8)
                sharp_img = (sharp_np * 255).astype(np.uint8)
                
                # Save images
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(blur_img)
                plt.title('Blurry Input')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(fake_img)
                plt.title('Deblurred Output')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(sharp_img)
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sample_{samples_saved+1}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                samples_saved += 1
                
                if samples_saved >= num_samples:
                    break
    
    print(f"Saved {samples_saved} sample images to {save_dir}")


def load_checkpoint(model_path, generator, discriminator=None, gen_optimizer=None, disc_optimizer=None):
    """
    Load model checkpoint with proper error handling
    
    Args:
        model_path: Path to checkpoint file
        generator: Generator model to load weights into
        discriminator: Optional discriminator model
        gen_optimizer: Optional generator optimizer
        disc_optimizer: Optional discriminator optimizer
        
    Returns:
        Dictionary with loaded information
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load generator
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        # Assume the checkpoint is just the generator state dict
        generator.load_state_dict(checkpoint)
    
    # Load discriminator if provided
    if discriminator is not None and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Load optimizers if provided
    if gen_optimizer is not None and 'gen_optimizer_state_dict' in checkpoint:
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    
    if disc_optimizer is not None and 'disc_optimizer_state_dict' in checkpoint:
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    losses = checkpoint.get('losses', {})
    
    print(f"Checkpoint loaded successfully from epoch {epoch}")
    
    return {
        'epoch': epoch,
        'losses': losses
    }


def check_gpu_memory():
    """Check and print GPU memory information"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("CUDA not available. Using CPU.")


def create_project_structure():
    """Create the project directory structure"""
    directories = [
        'checkpoints',
        'results',
        'samples',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

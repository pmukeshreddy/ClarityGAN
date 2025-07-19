"""
ClarityGAN Training Module
"""

import os
import torch
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from generator import create_generator
from discriminator import create_discriminator


class ClarityGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Create models
        self.generator = create_generator(self.device)
        self.discriminator = create_discriminator(self.device)
        
        # Create optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=config.GEN_LR, 
            betas=(0.5, 0.999)
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.DISC_LR, 
            betas=(0.5, 0.999)
        )
        
        # Create checkpoint directory
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        
    def train_discriminator(self, blur, sharp, fake):
        """Train discriminator for one step"""
        self.disc_optimizer.zero_grad()
        
        # Real pair loss
        d_real = self.discriminator(blur, sharp)
        loss_d_real = bce(d_real, torch.ones_like(d_real))
        
        # Fake pair loss
        d_fake = self.discriminator(blur, fake.detach())
        loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
        
        # Total discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.disc_optimizer.step()
        
        return loss_d.item()
    
    def train_generator(self, blur, sharp, fake):
        """Train generator for one step"""
        self.gen_optimizer.zero_grad()
        
        # Adversarial loss
        d_fake_gen = self.discriminator(blur, fake)
        loss_g_gan = bce(d_fake_gen, torch.ones_like(d_fake_gen))
        
        # L1 loss for pixel-level reconstruction
        loss_l1 = torch.mean(torch.abs(fake - sharp))
        
        # Total generator loss
        loss_g = (self.config.LAMBDA_GAN * loss_g_gan + 
                  self.config.LAMBDA_L1 * loss_l1)
        loss_g.backward()
        self.gen_optimizer.step()
        
        return loss_g.item(), loss_g_gan.item(), loss_l1.item()
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        total_d_loss = 0
        total_g_loss = 0
        total_gan_loss = 0
        total_l1_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for blur, sharp in pbar:
            blur, sharp = blur.to(self.device), sharp.to(self.device)
            
            # Generate fake images
            fake = self.generator(blur)
            
            # Train discriminator
            d_loss = self.train_discriminator(blur, sharp, fake)
            
            # Train generator
            g_loss, gan_loss, l1_loss = self.train_generator(blur, sharp, fake)
            
            # Update metrics
            total_d_loss += d_loss
            total_g_loss += g_loss
            total_gan_loss += gan_loss
            total_l1_loss += l1_loss
            
            # Update progress bar
            pbar.set_postfix({
                'D_Loss': f'{d_loss:.4f}',
                'G_Loss': f'{g_loss:.4f}',
                'L1_Loss': f'{l1_loss:.4f}'
            })
        
        # Calculate average losses
        num_batches = len(train_loader)
        return {
            'discriminator_loss': total_d_loss / num_batches,
            'generator_loss': total_g_loss / num_batches,
            'gan_loss': total_gan_loss / num_batches,
            'l1_loss': total_l1_loss / num_batches
        }
    
    def save_checkpoint(self, epoch, losses):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'losses': losses
        }
        
        checkpoint_path = os.path.join(
            self.config.MODEL_SAVE_PATH, 
            f'claritygan_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        train_loader, _ = get_dataloaders(self.config)
        
        print(f"Starting training on {self.device}")
        print(f"Dataset size: {len(train_loader.dataset)}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Number of epochs: {self.config.NUM_EPOCHS}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train for one epoch
            losses = self.train_epoch(train_loader, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Discriminator Loss: {losses['discriminator_loss']:.4f}")
            print(f"  Generator Loss: {losses['generator_loss']:.4f}")
            print(f"  GAN Loss: {losses['gan_loss']:.4f}")
            print(f"  L1 Loss: {losses['l1_loss']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, losses)
        
        # Save final model
        self.save_checkpoint(self.config.NUM_EPOCHS - 1, losses)
        print("Training completed!")


def main():
    """Main training function"""
    config = Config()
    trainer = ClarityGANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

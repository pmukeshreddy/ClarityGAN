"""
ClarityGAN Discriminator Module - PatchGAN Architecture
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=6):  # 3 (blurry) + 3 (sharp/generated)
        """
        PatchGAN Discriminator for image deblurring
        
        Args:
            input_channels: Number of input channels (6 for concatenated blurry+sharp)
        """
        super(PatchGANDiscriminator, self).__init__()

        # Discriminator layers
        self.conv1 = self._discriminator_block(input_channels, 64, norm=False)
        self.conv2 = self._discriminator_block(64, 128)
        self.conv3 = self._discriminator_block(128, 256)
        self.conv4 = self._discriminator_block(256, 512)
        
        # Final layer for PatchGAN (outputs a probability map)
        self.output = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def _discriminator_block(self, in_channels, out_channels, norm=True):
        """Create discriminator block"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, blurry, sharp_or_generated):
        """
        Forward pass through discriminator
        
        Args:
            blurry: Blurry input image
            sharp_or_generated: Either sharp ground truth or generated image
            
        Returns:
            Probability map indicating real/fake for each patch
        """
        # Concatenate inputs
        x = torch.cat([blurry, sharp_or_generated], dim=1)
        
        # Pass through discriminator layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Output probability map
        return self.output(x)


def create_discriminator(device):
    """Factory function to create and initialize discriminator"""
    discriminator = PatchGANDiscriminator()
    return discriminator.to(device)

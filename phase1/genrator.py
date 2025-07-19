"""
ClarityGAN Generator Module - U-Net Architecture
"""

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        """
        U-Net Generator for image deblurring
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            output_channels: Number of output channels (3 for RGB)
        """
        super(UNetGenerator, self).__init__()

        # Encoder layers
        self.enc1 = self._encoder_block(input_channels, 64, norm=False)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        self.enc5 = self._encoder_block(512, 512)

        # Decoder layers
        self.dec1 = self._decoder_block(512, 512)
        self.dec2 = self._decoder_block(1024, 256)  # 512 from dec + 512 from skip
        self.dec3 = self._decoder_block(512, 128)   # 256 + 256
        self.dec4 = self._decoder_block(256, 64)    # 128 + 128
        self.dec5 = self._decoder_block(128, 64)    # 64 + 64

        # Output layer
        self.output = nn.Conv2d(64, output_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def _encoder_block(self, in_channels, out_channels, norm=True):
        """Create encoder block"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels, out_channels):
        """Create decoder block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through U-Net"""
        # Encoder
        e1 = self.enc1(x)    # 64 channels
        e2 = self.enc2(e1)   # 128 channels
        e3 = self.enc3(e2)   # 256 channels
        e4 = self.enc4(e3)   # 512 channels
        bottleneck = self.enc5(e4)  # 512 channels

        # Decoder with skip connections
        d1 = self.dec1(bottleneck)  # 512 channels
        d1 = torch.cat([d1, e4], dim=1)  # Skip connection: 1024 channels

        d2 = self.dec2(d1)  # 256 channels
        d2 = torch.cat([d2, e3], dim=1)  # Skip connection: 512 channels

        d3 = self.dec3(d2)  # 128 channels
        d3 = torch.cat([d3, e2], dim=1)  # Skip connection: 256 channels

        d4 = self.dec4(d3)  # 64 channels
        d4 = torch.cat([d4, e1], dim=1)  # Skip connection: 128 channels

        d5 = self.dec5(d4)  # 64 channels

        # Output
        output = self.output(d5)
        return self.tanh(output)


def create_generator(device):
    """Factory function to create and initialize generator"""
    generator = UNetGenerator()
    return generator.to(device)

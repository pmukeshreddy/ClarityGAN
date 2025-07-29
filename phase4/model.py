import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class MultiStageDeblurNet(nn.Module):
    """
    Two-stage UNet: coarse deblur → refine via cross-stage skip.
    """
    def __init__(self, encoder="efficientnet-b0"):
        super().__init__()
        self.stage1 = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
        )
        self.stage2 = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=6,
            classes=3,
        )

    def forward(self, x):
        coarse = torch.sigmoid(self.stage1(x))
        combined = torch.cat([coarse, x], dim=1)
        refined = self.stage2(combined)
        return torch.sigmoid(x + refined)

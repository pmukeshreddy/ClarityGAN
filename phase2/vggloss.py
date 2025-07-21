from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.vgg_layers = nn.Sequential(*[vgg[i] for i in range(36)]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.loss = nn.L1Loss()

    def forward(self, generated, target):
        generated_norm = (generated + 1) / 2
        target_norm = (target + 1) / 2
        
        # Add ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        
        generated_norm = (generated_norm - mean) / std
        target_norm = (target_norm - mean) / std
        
        vgg_generated = self.vgg_layers(generated_norm)
        vgg_target = self.vgg_layers(target_norm)
        return self.loss(vgg_generated, vgg_target)

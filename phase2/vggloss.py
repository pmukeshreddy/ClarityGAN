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
        
        vgg_generated = self.vgg_layers(generated)
        vgg_target = self.vgg_layers(target)
        return self.loss(vgg_generated, vgg_target)

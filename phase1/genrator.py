import torch.nn as nn
import torch

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.bottleneck_res = nn.Sequential(ResidualBlock(512),ResidualBlock(512),ResidualBlock(512))
        self.bottleneck_attn = SelfAttention(512)
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec1_res = ResidualBlock(1024)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # 512 from dec + 512 from skip
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2_res = ResidualBlock(512) 
        self.dec2_attention = SelfAttention(512)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 256 + 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3_res = ResidualBlock(256)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 128 + 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec4_res = ResidualBlock(128)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 + 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2) #256
        e4 = self.enc4(e3) #512
        e5 = self.enc5(e4) #outputs 512 as well

        bott = self.bottleneck_res(e5)
        bott = self.bottleneck_attn(bott)
        

        d1 = self.dec1(bott)                    # 512 channels
        d1_skip = torch.cat([d1, e4], dim=1)    # 1024 channels
        d1_skip = self.dec1_res(d1_skip)        # Enhanced with residuals
        
        d2 = self.dec2(d1_skip)                 #256
        d2_skip = torch.cat([d2, e3], dim=1)    #256*2
        d2_skip = self.dec2_res(d2_skip)       
        d2_skip = self.dec2_attention(d2_skip)  
        
        d3 = self.dec3(d2_skip)                 # 128 channels
        d3_skip = torch.cat([d3, e2], dim=1)    # 128*2 channels
        d3_skip = self.dec3_res(d3_skip)        # Enhanced with residuals
        
        d4 = self.dec4(d3_skip)                 # 64 channels
        d4_skip = torch.cat([d4, e1], dim=1)    # 128 channels
        d4_skip = self.dec4_res(d4_skip)        # Enhanced with residuals
        
        d5 = self.dec5(d4_skip)                 # 64 channels
        
        out = self.out(d5)
        return self.tanh(out)
        

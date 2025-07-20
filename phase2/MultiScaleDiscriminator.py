class NLayerDiscrimnator(nn.Module):
    def __init__(self,input_nc=6,ndf=64,n_layers=3):
        super(NLayerDiscrimnator,self).__init__()
        layers = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult=1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3, num_d=2):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_d = num_d
        for i in range(num_d):
            net_d = NLayerDiscriminator(input_nc, ndf, n_layers)
            setattr(self, 'discriminator_' + str(i), net_d)
            
    def forward(self, blurry, sharp):
        x = torch.cat([blurry, sharp], dim=1)
        outputs = []
        for i in range(self.num_d):
            net_d = getattr(self, 'discriminator_' + str(i))
            outputs.append(net_d(x))
            # Downsample for the next discriminator
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
        return outputs


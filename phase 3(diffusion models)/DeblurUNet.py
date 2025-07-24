class DeblurUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, base_channels=64, channel_mult=(1, 2, 4, 8), 
                 num_res_blocks=2, attention_resolutions=(16,), dropout=0.1):
        super().__init__()
        
        self.time_emb = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_mult):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(now_channels, out_channels, base_channels * 4, dropout))
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1))
                channels.append(now_channels)
        
        # Middle
        self.mid_block1 = ResBlock(now_channels, now_channels, base_channels * 4, dropout)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResBlock(now_channels, now_channels, base_channels * 4, dropout)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mult)):
            out_channels = base_channels * mult
            for j in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock(channels.pop() + now_channels, out_channels, base_channels * 4, dropout)
                )
                now_channels = out_channels
            if i != len(channel_mult) - 1:
                self.up_blocks.append(nn.ConvTranspose2d(now_channels, now_channels, 4, stride=2, padding=1))
        
        self.norm_out = nn.GroupNorm(8, now_channels)
        self.conv_out = nn.Conv2d(now_channels, 3, 3, padding=1)
        
    def forward(self, x, t, blur_img):
        # Concatenate blurry image with noisy image
        x = torch.cat([x, blur_img], dim=1)
        
        # Time embedding
        t_emb = self.time_emb(t)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Downsampling
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
            hs.append(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)
        
        h = self.norm_out(h)
        h = F.relu(h)
        h = self.conv_out(h)
        
        return h

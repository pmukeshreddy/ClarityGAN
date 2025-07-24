class TimeEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
    def forward(self,t):
        device = t.device
        half_dim = self.dim//2
        embeddings = math.log(10000) /(half_dim-1)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

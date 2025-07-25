class DeblurDiffusion:
    def __init__(self, model, device, num_timesteps=1000):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x, t, blur_img):
        """Reverse diffusion process - single step"""
        with torch.no_grad():
            # Model prediction
            pred_noise = self.model(x, t, blur_img)
            
            # Get coefficients
            betas_t = self.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]
            
            # Equation for mean
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
            )
            
            if t[0] == 0:
                return model_mean
            else:
                posterior_variance_t = self.posterior_variance[t][:, None, None, None]
                noise = torch.randn_like(x)
                return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, blur_img, num_inference_steps=50):
        """Generate deblurred image"""
        batch_size = blur_img.shape[0]
        shape = blur_img.shape
        
        # Start from pure noise
        img = torch.randn(shape, device=self.device)
        
        # Inference with fewer steps
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t_batch, blur_img)
            
        return img
    
    def train_loss(self, sharp_img, blur_img):
        """Calculate training loss"""
        batch_size = sharp_img.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        # Sample noise
        noise = torch.randn_like(sharp_img)
        
        # Forward diffusion
        x_noisy = self.q_sample(sharp_img, t, noise=noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t, blur_img)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

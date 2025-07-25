import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

def gaussian(window_size, sigma):
    gauss = [math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)]
    return gauss / sum(gauss)

def ssim(img1, img2, window_size=11, sigma=1.5, channel=3, size_average=True):
    window = torch.tensor(gaussian(window_size, sigma), device=img1.device, dtype=img1.dtype)
    window = window.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
    window /= window.sum([2, 3], keepdim=True)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction='none').mean([1, 2, 3])
    return (20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))).mean()

def train_deblur_diffusion(model, diffusion, train_loader, val_loader, num_epochs=100, lr=1e-4):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for blur_img, sharp_img in progress_bar:
            blur_img = blur_img.to(diffusion.device)
            sharp_img = sharp_img.to(diffusion.device)
            
            optimizer.zero_grad()
            loss = diffusion.train_loss(sharp_img, blur_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        scheduler.step()
        
        # Validation every epoch
        model.eval()
        total_ssim = 0
        total_psnr = 0
        num_batches = 0
        with torch.no_grad():
            for val_blur, val_sharp in val_loader:
                val_blur = val_blur.to(diffusion.device)
                val_sharp = val_sharp.to(diffusion.device)
                deblurred = diffusion.sample(val_blur, num_inference_steps=50)
                # Assume images are in [0,1], RGB with channel=3
                current_ssim = ssim(deblurred, val_sharp, channel=3, size_average=False).mean()
                current_psnr = psnr(deblurred, val_sharp)
                total_ssim += current_ssim.item()
                total_psnr += current_psnr.item()
                num_batches += 1
        avg_ssim = total_ssim / num_batches
        avg_psnr = total_psnr / num_batches
        print(f'Epoch {epoch+1}, Val SSIM: {avg_ssim:.4f}, Val PSNR: {avg_psnr:.4f}')
        model.train()

# Initialize and train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeblurUNet().to(device)
diffusion = DeblurDiffusion(model, device)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Train the model
train_deblur_diffusion(model, diffusion, train_loader, val_loader, num_epochs=100)

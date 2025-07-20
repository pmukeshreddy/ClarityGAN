"""
ClarityGAN Evaluation Module
"""

import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import Config
from dataset import get_dataloaders
from generator import create_generator



class ClarityGANEvaluator:
    def __init__(self, config, model_path):
        self.config = config
        self.device = config.DEVICE
        
        # Load trained generator
        self.generator = create_generator(self.device)
        self.load_model(model_path)
        self.generator.eval()
        
    def load_model(self, model_path):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'generator_state_dict' in checkpoint:
            # Full checkpoint
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Just state dict
            self.generator.load_state_dict(checkpoint)
            print("Loaded model state dict")
    
    def calculate_metrics(self, fake_img, real_img):
        """Calculate PSNR and SSIM metrics"""
        # Convert tensors to numpy arrays and transpose to HWC format
        fake_np = fake_img.cpu().numpy()[0].transpose(1, 2, 0)
        real_np = real_img.cpu().numpy()[0].transpose(1, 2, 0)
        
        # Ensure values are in [0, 1] range
        fake_np = np.clip(fake_np, 0, 1)
        real_np = np.clip(real_np, 0, 1)
        
        # Calculate PSNR
        psnr_value = psnr(real_np, fake_np, data_range=1.0)
        
        # Calculate SSIM
        ssim_value = ssim(real_np, fake_np, 
                         multichannel=True, 
                         channel_axis=2,
                         data_range=1.0)
        
        return psnr_value, ssim_value
    
    def evaluate_dataset(self, test_loader):
        """Evaluate on entire test dataset"""
        psnr_scores = []
        ssim_scores = []
        
        print(f"Evaluating on {len(test_loader)} test images...")
        
        with torch.no_grad():
            for blur, sharp in tqdm(test_loader, desc="Evaluating"):
                blur = blur.to(self.device)
                
                # Generate deblurred image
                fake = self.generator(blur)
                
                # Calculate metrics
                psnr_val, ssim_val = self.calculate_metrics(fake, sharp)
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
        
        # Calculate average metrics
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        std_psnr = np.std(psnr_scores)
        std_ssim = np.std(ssim_scores)
        
        return {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'std_psnr': std_psnr,
            'std_ssim': std_ssim,
            'psnr_scores': psnr_scores,
            'ssim_scores': ssim_scores
        }
    
    def evaluate(self):
        """Main evaluation function"""
        _, test_loader = get_dataloaders(self.config)
        results = self.evaluate_dataset(test_loader)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Average PSNR: {results['avg_psnr']:.4f} ± {results['std_psnr']:.4f}")
        print(f"Average SSIM: {results['avg_ssim']:.4f} ± {results['std_ssim']:.4f}")
        print(f"Number of test images: {len(results['psnr_scores'])}")
        print("="*50)
        
        return results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ClarityGAN')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    config = Config()
    evaluator = ClarityGANEvaluator(config, args.model_path)
    results = evaluator.evaluate()
    
    return results


if __name__ == "__main__":
    main()

"""
ClarityGAN Inference Module
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image

from config import Config
from generator import create_generator


class ClarityGANInference:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
    
    def preprocess_image(self, image_path):
        """Preprocess input image for inference"""
        # Read image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array or PIL image
            img = np.array(image_path)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def postprocess_image(self, tensor_img):
        """Convert tensor back to numpy image"""
        # Remove batch dimension and move to CPU
        img = tensor_img.squeeze(0).cpu()
        
        # Convert from CHW to HWC
        img = img.numpy().transpose(1, 2, 0)
        
        # Clip values and convert to uint8
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def deblur_image(self, image_path):
        """
        Deblur a single image
        
        Args:
            image_path: Path to blurry image or numpy array
            
        Returns:
            Deblurred image as numpy array
        """
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_image(image_path)
            
            # Generate deblurred image
            output_tensor = self.generator(input_tensor)
            
            # Postprocess
            output_img = self.postprocess_image(output_tensor)
            
        return output_img
    
    def deblur_and_save(self, input_path, output_path):
        """
        Deblur image and save result
        
        Args:
            input_path: Path to input blurry image
            output_path: Path to save deblurred image
        """
        # Deblur image
        deblurred_img = self.deblur_image(input_path)
        
        # Save result
        # Convert RGB to BGR for OpenCV
        deblurred_bgr = cv2.cvtColor(deblurred_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, deblurred_bgr)
        
        print(f"Deblurred image saved to: {output_path}")
        
        return deblurred_img
    
    def batch_deblur(self, input_dir, output_dir, file_extensions=('.jpg', '.jpeg', '.png')):
        """
        Deblur all images in a directory
        
        Args:
            input_dir: Directory containing blurry images
            output_dir: Directory to save deblurred images
            file_extensions: Tuple of valid file extensions
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image files
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(file_extensions)]
        
        print(f"Processing {len(image_files)} images...")
        
        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"deblurred_{filename}")
            
            try:
                self.deblur_and_save(input_path, output_path)
                print(f"✓ Processed: {filename}")
            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}")
        
        print(f"Batch processing completed. Results saved in: {output_dir}")


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ClarityGAN Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input blurry image or directory')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save deblurred image or directory')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in input directory')
    
    args = parser.parse_args()
    
    # Create inference object
    inferencer = ClarityGANInference(args.model_path)
    
    if args.batch:
        # Batch processing
        inferencer.batch_deblur(args.input_path, args.output_path)
    else:
        # Single image processing
        inferencer.deblur_and_save(args.input_path, args.output_path)


if __name__ == "__main__":
    main()

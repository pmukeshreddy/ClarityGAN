"""
ClarityGAN Main Script
Image Deblurring using Generative Adversarial Networks

Usage:
    python main.py --mode train                    # Train the model
    python main.py --mode evaluate --model_path checkpoint.pth    # Evaluate model
    python main.py --mode infer --model_path checkpoint.pth --input_path image.jpg --output_path deblurred.jpg
"""

import argparse
import os
import sys

from config import Config
from train import ClarityGANTrainer
from evaluate import ClarityGANEvaluator
from inference import ClarityGANInference
from utils import print_model_info, check_gpu_memory, create_project_structure
from generator import create_generator
from discriminator import create_discriminator


def train_model(config):
    """Train ClarityGAN model"""
    print("="*60)
    print("CLARITYGAN TRAINING")
    print("="*60)
    
    # Check GPU
    check_gpu_memory()
    
    # Create project structure
    create_project_structure()
    
    # Create models for info display
    generator = create_generator(config.DEVICE)
    discriminator = create_discriminator(config.DEVICE)
    print_model_info(generator, discriminator)
    
    # Start training
    trainer = ClarityGANTrainer(config)
    trainer.train()


def evaluate_model(config, model_path):
    """Evaluate trained ClarityGAN model"""
    print("="*60)
    print("CLARITYGAN EVALUATION")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    evaluator = ClarityGANEvaluator(config, model_path)
    results = evaluator.evaluate()
    
    return results


def infer_single_image(model_path, input_path, output_path):
    """Deblur a single image"""
    print("="*60)
    print("CLARITYGAN INFERENCE")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not os.path.exists(input_path):
        print(f"Error: Input image not found: {input_path}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Perform inference
    inferencer = ClarityGANInference(model_path)
    inferencer.deblur_and_save(input_path, output_path)
    
    print(f"Deblurring completed!")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")


def batch_infer(model_path, input_dir, output_dir):
    """Deblur all images in a directory"""
    print("="*60)
    print("CLARITYGAN BATCH INFERENCE")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Perform batch inference
    inferencer = ClarityGANInference(model_path)
    inferencer.batch_deblur(input_dir, output_dir)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='ClarityGAN - Image Deblurring with GANs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train the model
    python main.py --mode train
    
    # Evaluate trained model
    python main.py --mode evaluate --model_path checkpoints/claritygan_epoch_50.pth
    
    # Deblur single image
    python main.py --mode infer --model_path checkpoints/claritygan_epoch_50.pth \\
                   --input_path blurry_image.jpg --output_path deblurred_image.jpg
    
    # Batch deblur directory
    python main.py --mode batch --model_path checkpoints/claritygan_epoch_50.pth \\
                   --input_path ./blurry_images/ --output_path ./deblurred_images/
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate', 'infer', 'batch'],
                       help='Mode to run: train, evaluate, infer, or batch')
    
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model checkpoint (required for evaluate/infer/batch)')
    
    parser.add_argument('--input_path', type=str,
                       help='Input image path (for infer) or directory (for batch)')
    
    parser.add_argument('--output_path', type=str,
                       help='Output image path (for infer) or directory (for batch)')
    
    parser.add_argument('--config', type=str,
                       help='Path to custom config file (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Validate arguments based on mode
    if args.mode == 'train':
        train_model(config)
        
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("Error: --model_path is required for evaluation")
            sys.exit(1)
        evaluate_model(config, args.model_path)
        
    elif args.mode == 'infer':
        if not all([args.model_path, args.input_path, args.output_path]):
            print("Error: --model_path, --input_path, and --output_path are required for inference")
            sys.exit(1)
        infer_single_image(args.model_path, args.input_path, args.output_path)
        
    elif args.mode == 'batch':
        if not all([args.model_path, args.input_path, args.output_path]):
            print("Error: --model_path, --input_path, and --output_path are required for batch inference")
            sys.exit(1)
        batch_infer(args.model_path, args.input_path, args.output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

"""
Generate CAPTCHAs using the trained V2 character-level flow model.
Uses the improved model architecture with EMA weights.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import string
import argparse

# Import from train_character_model_v2.py
import sys
sys.path.append(str(Path(__file__).parent))
from train_character_model_v2 import ImprovedConditionalUNet, EMA


def load_model(checkpoint_path, device):
    """Load the trained V2 model from checkpoint (with EMA)."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = ImprovedConditionalUNet(
        in_channels=3,
        out_channels=3,
        num_classes=36,
        channels=[96, 192, 384, 768]
    ).to(device)
    
    # Load EMA weights for best generation quality
    if 'ema_state_dict' in checkpoint:
        print("Loading EMA weights for generation...")
        model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    return model


def class_to_char(class_idx):
    """Convert class index to character (0-9, a-z)."""
    if class_idx < 10:
        return str(class_idx)
    else:
        return chr(ord('a') + class_idx - 10)


def char_to_class(char):
    """Convert character to class index."""
    if char.isdigit():
        return int(char)
    else:
        return ord(char.lower()) - ord('a') + 10


def generate_random_label(length=4):
    """Generate a random CAPTCHA label."""
    chars = string.digits + string.ascii_lowercase
    return ''.join(random.choice(chars) for _ in range(length))


@torch.no_grad()
def sample_character(model, class_label, device, num_steps=150, guidance_scale=3.0):
    """
    Sample a single character image using CFG.
    
    Args:
        model: The trained flow model
        class_label: Class index (0-35)
        device: torch device
        num_steps: Number of sampling steps (150 for V2)
        guidance_scale: CFG guidance scale (3.0 for V2)
    
    Returns:
        PIL Image of the generated character
    """
    model.eval()
    
    # Start from noise
    x = torch.randn(1, 3, 64, 64, device=device)
    
    # Create conditioning
    c = torch.tensor([class_label], device=device)
    c_null = torch.tensor([36], device=device)  # Null class
    
    # ODE solver (Euler method)
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t = step * dt
        t_tensor = torch.tensor([t], device=device)
        
        # Classifier-Free Guidance
        v_uncond = model(x, t_tensor, c_null)
        v_cond = model(x, t_tensor, c)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        # Euler step
        x = x + v * dt
    
    # Convert to image
    img = x[0].cpu()
    img = torch.clamp(img, -1, 1)
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = transforms.ToPILImage()(img)
    
    return img


def whiten_background(char_img, threshold=128):
    """
    Replace dark background pixels with white.
    
    Args:
        char_img: PIL Image of character
        threshold: Pixels brighter than this are considered background
    
    Returns:
        PIL Image with whitened background
    """
    import numpy as np
    
    # Convert to numpy array
    img_array = np.array(char_img)
    
    # Create mask for bright pixels (background)
    # A pixel is considered background if all RGB channels are > threshold
    background_mask = np.all(img_array > threshold, axis=2)
    
    # Replace background pixels with white
    img_array[background_mask] = [255, 255, 255]
    
    # Also replace very dark pixels (< 50) with white (the black spacing artifacts)
    dark_mask = np.all(img_array < 50, axis=2)
    img_array[dark_mask] = [255, 255, 255]
    
    return Image.fromarray(img_array)


def compose_captcha(char_images, spacing=10, background_color=255, whiten_backgrounds=True):
    """
    Compose individual character images into a full CAPTCHA.
    
    Args:
        char_images: List of PIL Images (characters)
        spacing: Pixels between characters
        background_color: Background color (0-255)
        whiten_backgrounds: If True, replace dark backgrounds with white
    
    Returns:
        PIL Image of the composed CAPTCHA
    """
    if not char_images:
        return None
    
    # Whiten backgrounds if requested
    if whiten_backgrounds:
        char_images = [whiten_background(img) for img in char_images]
    
    # Get dimensions
    char_width = char_images[0].width
    char_height = char_images[0].height
    char_height = char_images[0].height
    
    # Calculate total width
    total_width = len(char_images) * char_width + (len(char_images) - 1) * spacing
    
    # Create canvas
    captcha = Image.new('RGB', (total_width, char_height), 
                       color=(background_color, background_color, background_color))
    
    # Paste characters
    x_offset = 0
    for char_img in char_images:
        captcha.paste(char_img, (x_offset, 0))
        x_offset += char_width + spacing
    
    return captcha


def generate_captcha(model, label, device, num_steps=150, guidance_scale=3.0):
    """
    Generate a full CAPTCHA image for a given label.
    
    Args:
        model: The trained flow model
        label: String label (e.g., "ab3x")
        device: torch device
        num_steps: Number of sampling steps
        guidance_scale: CFG guidance scale
    
    Returns:
        PIL Image of the CAPTCHA
    """
    # Generate each character
    char_images = []
    for char in label.lower():
        class_idx = char_to_class(char)
        char_img = sample_character(model, class_idx, device, num_steps, guidance_scale)
        char_images.append(char_img)
    
    # Compose into CAPTCHA
    captcha = compose_captcha(char_images, spacing=10, background_color=255)
    return captcha


def main():
    parser = argparse.ArgumentParser(description='Generate CAPTCHAs using trained V2 flow model')
    parser.add_argument('--checkpoint', type=str, 
                       default='results/character_model_v2/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--num_captchas', type=int, default=100,
                       help='Number of CAPTCHAs to generate')
    parser.add_argument('--captcha_length', type=int, default=4,
                       help='Length of each CAPTCHA')
    parser.add_argument('--output_dir', type=str, default='generated_captchas_v2',
                       help='Output directory for generated CAPTCHAs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_steps', type=int, default=150,
                       help='Number of sampling steps (150 recommended for V2)')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                       help='CFG guidance scale (3.0 recommended for V2)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save labels file
    labels_file = output_dir / 'labels.txt'
    
    print(f"\nGenerating {args.num_captchas} CAPTCHAs...")
    print(f"CAPTCHA length: {args.captcha_length} characters")
    print(f"Sampling steps: {args.num_steps}")
    print(f"CFG guidance scale: {args.guidance_scale}")
    print(f"Output directory: {output_dir}")
    
    # Generate CAPTCHAs
    with open(labels_file, 'w') as f:
        for i in tqdm(range(args.num_captchas), desc="Generating"):
            # Generate random label
            label = generate_random_label(args.captcha_length)
            
            # Generate CAPTCHA
            captcha = generate_captcha(model, label, device, args.num_steps, args.guidance_scale)
            
            # Save
            filename = f"captcha_{i:05d}.png"
            captcha.save(output_dir / filename)
            
            # Write label
            f.write(f"{filename}\t{label}\n")
    
    print(f"\n✓ Generated {args.num_captchas} CAPTCHAs!")
    print(f"  Images: {output_dir}")
    print(f"  Labels: {labels_file}")


if __name__ == '__main__':
    main()

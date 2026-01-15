"""
Sample generation script for trained Rectified Flow model
Generates CAPTCHA images from trained model checkpoint
"""

import torch
from model import UNetFlowModel
from torchvision.utils import save_image
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_samples(model, num_samples=100, steps=20, device='cuda', img_size=(80, 640)):
    """
    Generate CAPTCHA samples using ODE solver
    Args:
        model: Trained flow model
        num_samples: Number of images to generate
        steps: Number of ODE integration steps (more = higher quality)
        device: Device to run on
        img_size: Output image size (H, W)
    Returns:
        Generated images tensor
    """
    model.eval()
    
    # Start from standard Gaussian noise (x_0)
    x = torch.randn(num_samples, 3, img_size[0], img_size[1], device=device)
    dt = 1.0 / steps
    
    logger.info(f"Generating {num_samples} samples with {steps} ODE steps...")
    
    # ODE integration: dx/dt = v(x, t)
    for i in range(steps):
        t = torch.ones(num_samples, device=device) * i * dt
        v = model(x, t)
        x = x + v * dt
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Step {i+1}/{steps}")
    
    # Denormalize from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    
    return x


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = UNetFlowModel(
        img_channels=3,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        num_res_blocks=args.num_res_blocks
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Generate samples
    samples = generate_samples(
        model,
        num_samples=args.num_samples,
        steps=args.ode_steps,
        device=device,
        img_size=(args.img_height, args.img_width)
    )
    
    # Save grid of samples
    grid_path = output_dir / 'generated_grid.png'
    save_image(samples, grid_path, nrow=10, padding=2)
    logger.info(f"Sample grid saved to {grid_path}")
    
    # Save individual samples
    if args.save_individual:
        individual_dir = output_dir / 'individual'
        individual_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(samples):
            save_image(img, individual_dir / f'sample_{i:04d}.png')
        
        logger.info(f"Individual samples saved to {individual_dir}")
    
    logger.info("Generation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CAPTCHA samples from trained model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='Output directory for generated images')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--ode_steps', type=int, default=20,
                        help='Number of ODE integration steps')
    parser.add_argument('--img_width', type=int, default=640,
                        help='Image width')
    parser.add_argument('--img_height', type=int, default=80,
                        help='Image height')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels (must match training)')
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Channel multipliers (must match training)')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='Number of residual blocks (must match training)')
    parser.add_argument('--save_individual', action='store_true',
                        help='Save individual images in addition to grid')
    
    args = parser.parse_args()
    main(args)

"""
Evaluation script for Rectified Flow CAPTCHA Generation
Computes test loss and generates evaluation metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
import numpy as np
from model import UNetFlowModel


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CAPTCHADataset(Dataset):
    """Dataset loader for CAPTCHA images"""
    def __init__(self, data_dir, img_size=(80, 640)):
        self.data_dir = Path(data_dir)
        self.img_paths = sorted(glob.glob(str(self.data_dir / "*.png")))
        self.img_size = img_size
        
        logger.info(f"Found {len(self.img_paths)} images in {data_dir}")
        
        # No augmentation for evaluation
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)


class RectifiedFlowLoss(nn.Module):
    """
    Rectified Flow Loss: Simple MSE on velocity prediction
    x_t = t * x_1 + (1 - t) * x_0
    v_t = x_1 - x_0 (constant velocity)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, model, x_1, device):
        """
        Args:
            model: The flow model
            x_1: Real data (target)
            device: torch device
        """
        batch_size = x_1.shape[0]
        
        # Sample random time t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1, device=device)
        
        # Interpolate: x_t = t * x_1 + (1 - t) * x_0
        t_expanded = t.view(batch_size, 1, 1, 1)
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        
        # True velocity (constant for straight paths)
        v_true = x_1 - x_0
        
        # Predict velocity
        v_pred = model(x_t, t)
        
        # MSE loss
        loss = self.mse(v_pred, v_true)
        
        return loss


@torch.no_grad()
def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    logger.info("Evaluating on test set...")
    
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(device)
        
        # Compute loss
        loss = criterion(model, batch, device)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def sample_images(model, num_samples=16, steps=50, device='cuda', img_size=(80, 640)):
    """Generate samples using ODE solver"""
    model.eval()
    
    # Start from noise
    x = torch.randn(num_samples, 3, img_size[0], img_size[1], device=device)
    
    # ODE solver (Euler method)
    dt = 1.0 / steps
    
    for i in tqdm(range(steps), desc="Sampling"):
        t = torch.ones(num_samples, device=device) * (i * dt)
        
        # Predict velocity
        v = model(x, t)
        
        # Update: x_{t+dt} = x_t + v * dt
        x = x + v * dt
    
    # Denormalize from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    
    return x


def compute_statistics(samples):
    """Compute basic statistics of generated samples"""
    # Convert to numpy
    samples_np = samples.cpu().numpy()
    
    stats = {
        'mean': samples_np.mean(),
        'std': samples_np.std(),
        'min': samples_np.min(),
        'max': samples_np.max(),
        'shape': samples_np.shape
    }
    
    return stats


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    img_size = (args.img_height, args.img_width)
    test_dataset = CAPTCHADataset(args.test_dir, img_size=img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = UNetFlowModel(
        img_channels=3,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        num_res_blocks=args.num_res_blocks,
        time_emb_dim=256,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        train_loss = checkpoint.get('loss', 'unknown')
        logger.info(f"Loaded checkpoint from epoch {epoch}, train loss: {train_loss}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded model weights")
    
    # Create loss function
    criterion = RectifiedFlowLoss()
    
    # ===== EVALUATE TEST LOSS =====
    test_loss = evaluate_model(model, test_loader, criterion, device)
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST LOSS: {test_loss:.6f}")
    logger.info(f"{'='*60}\n")
    
    # ===== GENERATE SAMPLES =====
    if args.generate_samples:
        logger.info(f"Generating {args.num_samples} samples with {args.sample_steps} steps...")
        samples = sample_images(
            model,
            num_samples=args.num_samples,
            steps=args.sample_steps,
            device=device,
            img_size=img_size
        )
        
        # Save samples
        from torchvision.utils import save_image
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample_path = output_dir / f'evaluation_samples_{args.num_samples}.png'
        save_image(samples, sample_path, nrow=4, padding=2)
        logger.info(f"Samples saved to: {sample_path}")
        
        # Compute statistics
        stats = compute_statistics(samples)
        logger.info(f"\nSample Statistics:")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Std:  {stats['std']:.4f}")
        logger.info(f"  Min:  {stats['min']:.4f}")
        logger.info(f"  Max:  {stats['max']:.4f}")
    
    # ===== SAVE RESULTS =====
    results = {
        'checkpoint': args.checkpoint,
        'test_loss': float(test_loss),
        'test_samples': len(test_dataset),
        'batch_size': args.batch_size,
        'image_size': img_size
    }
    
    results_path = Path(args.output_dir) / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Checkpoint: {results['checkpoint']}\n")
        f.write(f"Test Loss: {results['test_loss']:.6f}\n")
        f.write(f"Test Samples: {results['test_samples']}\n")
        f.write(f"Batch Size: {results['batch_size']}\n")
        f.write(f"Image Size: {results['image_size']}\n")
    
    logger.info(f"\nResults saved to: {results_path}")
    
    logger.info("\n✓ Evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Rectified Flow CAPTCHA Model')
    
    # Data parameters
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results/evaluation', help='Output directory')
    parser.add_argument('--img_width', type=int, default=640, help='Image width')
    parser.add_argument('--img_height', type=int, default=80, help='Image height')
    
    # Model parameters (must match training)
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels')
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Channel multipliers for each layer')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Sampling parameters
    parser.add_argument('--generate_samples', action='store_true', help='Generate sample images')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--sample_steps', type=int, default=50, help='Number of sampling steps')
    
    args = parser.parse_args()
    main(args)

"""
Training script for Rectified Flow CAPTCHA Generation
Implements Flow Matching with straight-line interpolation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
import json
from model import UNetFlowModel, count_parameters


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CAPTCHADataset(Dataset):
    """Dataset loader for CAPTCHA images"""
    def __init__(self, data_dir, img_size=(80, 640), augment=False):
        self.data_dir = Path(data_dir)
        self.img_paths = sorted(glob.glob(str(self.data_dir / "*.png")))
        self.img_size = img_size
        
        logger.info(f"Found {len(self.img_paths)} images in {data_dir}")
        
        # Base transforms
        transform_list = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ]
        
        # Add augmentation for training
        if augment:
            transform_list = [
                transforms.Resize(img_size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image


class RectifiedFlowLoss(nn.Module):
    """
    Rectified Flow Loss: MSE between predicted velocity and true velocity
    For straight-line interpolation: x_t = t * x_1 + (1-t) * x_0
    True velocity: v = x_1 - x_0
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, model, x1):
        """
        Args:
            model: The flow model
            x1: Real data samples (target)
        Returns:
            loss: Flow matching loss
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Sample x0 from standard Gaussian (source distribution)
        x0 = torch.randn_like(x1)
        
        # Sample random time t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Linear interpolation: x_t = t * x_1 + (1-t) * x_0
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = t_expanded * x1 + (1 - t_expanded) * x0
        
        # True velocity for straight line
        v_true = x1 - x0
        
        # Predict velocity with model
        v_pred = model(x_t, t)
        
        # MSE loss
        loss = self.mse(v_pred, v_true)
        
        return loss


@torch.no_grad()
def sample_images(model, num_samples=16, steps=10, device='cuda'):
    """
    Generate samples using ODE solver (Euler method)
    Args:
        model: Trained flow model
        num_samples: Number of images to generate
        steps: Number of ODE steps
        device: Device to run on
    Returns:
        Generated images tensor
    """
    model.eval()
    
    # Start from noise (x_0)
    x = torch.randn(num_samples, 3, 80, 640, device=device)
    dt = 1.0 / steps
    
    # ODE integration: dx/dt = v(x, t)
    for i in range(steps):
        t = torch.ones(num_samples, device=device) * i * dt
        v = model(x, t)
        x = x + v * dt
    
    # Denormalize from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    
    return x


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, x1 in enumerate(progress_bar):
        x1 = x1.to(device)
        
        # Forward pass
        loss = criterion(model, x1)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataset and dataloader
    train_dataset = CAPTCHADataset(
        args.train_dir,
        img_size=(args.img_height, args.img_width),
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Steps per epoch: {len(train_loader)}")
    
    # Create model
    model = UNetFlowModel(
        img_channels=3,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model created with {count_parameters(model):,} parameters")
    
    # Loss and optimizer
    criterion = RectifiedFlowLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint {args.resume} not found, starting from scratch")
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1)
        logger.info(f"Average loss: {avg_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = checkpoints_dir / f'checkpoint_epoch_{epoch + 1:04d}.pt'
            save_checkpoint(model, optimizer, epoch, avg_loss, save_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = checkpoints_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, epoch, avg_loss, save_path)
            logger.info(f"New best model saved! Loss: {best_loss:.4f}")
        
        # Generate samples
        if (epoch + 1) % args.sample_every == 0:
            logger.info("Generating samples...")
            samples = sample_images(model, num_samples=16, steps=args.sample_steps, device=device)
            
            # Save samples
            from torchvision.utils import save_image
            save_image(
                samples,
                samples_dir / f'samples_epoch_{epoch + 1:04d}.png',
                nrow=4,
                padding=2
            )
            logger.info(f"Samples saved to {samples_dir}")
    
    # Save final model
    final_path = checkpoints_dir / 'final_model.pt'
    save_checkpoint(model, optimizer, args.epochs - 1, avg_loss, final_path)
    logger.info("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Rectified Flow for CAPTCHA Generation')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--img_width', type=int, default=640, help='Image width')
    parser.add_argument('--img_height', type=int, default=80, help='Image height')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels')
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Channel multipliers for each layer')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Sampling parameters
    parser.add_argument('--sample_steps', type=int, default=10, help='Number of sampling steps')
    parser.add_argument('--sample_every', type=int, default=5, help='Sample every N epochs')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Resume training
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)

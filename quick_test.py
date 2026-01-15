"""
Quick test script - Train for 1 epoch and generate samples
For rapid iteration and testing before full training
"""

import torch
import sys
from pathlib import Path
from train import CAPTCHADataset, RectifiedFlowLoss, train_epoch, save_checkpoint, sample_images
from model import UNetFlowModel, count_parameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_test(train_dir, output_dir='./test_output'):
    """
    Quick test: 1 epoch + sample generation
    
    Args:
        train_dir: Path to training data
        output_dir: Output directory
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create smaller model for testing
    logger.info("Creating model...")
    model = UNetFlowModel(
        img_channels=3,
        base_channels=32,  # Smaller for quick testing
        channel_multipliers=(1, 2, 4),  # Fewer layers
        num_res_blocks=1
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Small dataset for testing
    logger.info("Loading dataset...")
    from torch.utils.data import DataLoader, Subset
    
    dataset = CAPTCHADataset(train_dir, img_size=(80, 640), augment=True)
    
    # Use only first 100 samples for quick test
    subset = Subset(dataset, range(min(100, len(dataset))))
    loader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=2)
    
    logger.info(f"Training on {len(subset)} samples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = RectifiedFlowLoss()
    
    # Train 1 epoch
    logger.info("Training for 1 epoch...")
    loss = train_epoch(model, loader, optimizer, criterion, device, epoch=1)
    logger.info(f"Epoch 1 loss: {loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = output_dir / 'test_checkpoint.pt'
    save_checkpoint(model, optimizer, 0, loss, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Generate samples
    logger.info("Generating samples...")
    samples = sample_images(model, num_samples=16, steps=10, device=device)
    
    # Save samples
    from torchvision.utils import save_image
    sample_path = output_dir / 'test_samples.png'
    save_image(samples, sample_path, nrow=4, padding=2)
    logger.info(f"Samples saved to {sample_path}")
    
    logger.info("✓ Quick test complete!")
    return checkpoint_path, sample_path


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_dir = sys.argv[1]
    else:
        train_dir = './data/preprocessed/train'
    
    quick_test(train_dir)

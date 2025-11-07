"""
Training script for character recognition CNN.

Loads CAPTCHA images, applies preprocessing and tokenization,
then trains a CNN to recognize individual alphanumeric characters.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
from tqdm import tqdm

# Import preprocessing and tokenization
from preprocessing.hairline_removal import color_voting_propagation
from tokenization.color_region_tokenizer import ColorRegionTokenizer
from model_structure import CharacterCNN


class CaptchaCharacterDataset(Dataset):
    """
    Dataset that loads CAPTCHA images, tokenizes them, and extracts individual characters.
    Only includes samples where the number of tokens matches the ground truth label length.
    """

    def __init__(
        self,
        image_dir: str,
        preprocessing_method: str = "color_voting",
        target_size: Tuple[int, int] = (32, 32),
        split: str = "train",
        train_ratio: float = 0.8,
        random_seed: int = 42,
        augment: bool = True
    ):
        """
        Args:
            image_dir: Directory containing CAPTCHA images
            preprocessing_method: Preprocessing method to use
            target_size: Target size for character images (width, height)
            split: "train" or "val"
            train_ratio: Ratio of training samples
            random_seed: Random seed for splitting
            augment: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.preprocessing_method = preprocessing_method
        self.target_size = target_size
        self.augment = augment

        # Character mapping
        self.idx_to_char = string.digits + string.ascii_uppercase
        self.char_to_idx = {char: idx for idx, char in enumerate(self.idx_to_char)}

        # Initialize tokenizer
        self.tokenizer = ColorRegionTokenizer(
            white_threshold=200,
            black_threshold=50,
            min_region_area=25,
            max_region_area=10000,
            target_height=80,
            padding=2,
            min_saturation=15,
            max_aspect_ratio=8.0,
            split_wide_regions=True
        )

        # Load and prepare dataset
        print(f"Loading {split} dataset from {image_dir}...")
        self.samples = self._prepare_dataset(split, train_ratio, random_seed)
        print(f"Loaded {len(self.samples)} character samples for {split}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to remove hairlines."""
        if self.preprocessing_method == "color_voting":
            return color_voting_propagation(image, target_black_only=True, iterations=2)
        else:
            return image

    def _prepare_dataset(
        self,
        split: str,
        train_ratio: float,
        random_seed: int
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Load all CAPTCHA images, tokenize them, and extract individual characters.
        Only keep samples where token count matches label length.
        """
        all_image_paths = sorted(list(self.image_dir.glob("*.png")))

        if len(all_image_paths) == 0:
            raise ValueError(f"No PNG images found in {self.image_dir}")

        # Split into train/val
        random.seed(random_seed)
        random.shuffle(all_image_paths)
        split_idx = int(len(all_image_paths) * train_ratio)

        if split == "train":
            image_paths = all_image_paths[:split_idx]
        else:
            image_paths = all_image_paths[split_idx:]

        samples = []
        skipped = 0
        token_mismatch = 0

        print(f"Processing {len(image_paths)} CAPTCHA images for {split}...")

        for img_path in tqdm(image_paths, desc=f"Tokenizing {split} images"):
            # Extract label from filename (e.g., "ed57g-0.png" -> "ed57g")
            label = img_path.stem.split('-')[0].upper()

            # Validate label (only alphanumeric)
            if not all(c in self.char_to_idx for c in label):
                skipped += 1
                continue

            # Load image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                skipped += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Preprocess
            preprocessed = self._preprocess_image(img_rgb)

            # Tokenize
            try:
                tokens = self.tokenizer.tokenize(preprocessed)
            except Exception as e:
                print(f"Error tokenizing {img_path.name}: {e}")
                skipped += 1
                continue

            # Only use if token count matches label length
            if len(tokens) != len(label):
                token_mismatch += 1
                continue

            # Add each character token with its label
            for token, char in zip(tokens, label):
                # Convert token to grayscale
                token_gray = cv2.cvtColor(token, cv2.COLOR_RGB2GRAY)
                samples.append((token_gray, char))

        print(f"Skipped {skipped} images (invalid/failed)")
        print(f"Skipped {token_mismatch} images (token count mismatch)")
        print(f"Total character samples: {len(samples)}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_img, char_label = self.samples[idx]

        # Resize to target size
        token_resized = cv2.resize(token_img, self.target_size)

        # Apply augmentation if training
        if self.augment:
            token_resized = self._augment(token_resized)

        # Convert to tensor and normalize
        token_tensor = torch.from_numpy(token_resized).float() / 255.0
        token_tensor = token_tensor.unsqueeze(0)  # Add channel dimension

        # Normalize to [-1, 1]
        token_tensor = (token_tensor - 0.5) / 0.5

        # Get label index
        label_idx = self.char_to_idx[char_label]

        return token_tensor, label_idx

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderValue=255)

        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # Random small translation
        if random.random() > 0.5:
            tx = random.randint(-2, 2)
            ty = random.randint(-2, 2)
            h, w = image.shape[:2]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h), borderValue=255)

        return image


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    # Configuration
    TRAIN_DIR = "data/raw/train"
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0  # Set to 0 for Windows
    RANDOM_SEED = 42
    MODEL_SAVE_PATH = "char_recognizer.pth"

    # Set random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("Character Recognition Training")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Training directory: {TRAIN_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print("=" * 70)
    print()

    # Create datasets
    train_dataset = CaptchaCharacterDataset(
        image_dir=TRAIN_DIR,
        preprocessing_method="color_voting",
        target_size=(32, 32),
        split="train",
        train_ratio=0.8,
        random_seed=RANDOM_SEED,
        augment=True
    )

    val_dataset = CaptchaCharacterDataset(
        image_dir=TRAIN_DIR,
        preprocessing_method="color_voting",
        target_size=(32, 32),
        split="val",
        train_ratio=0.8,
        random_seed=RANDOM_SEED,
        augment=False
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    model = CharacterCNN(input_channels=1, num_classes=36)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_acc)

        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, MODEL_SAVE_PATH)
            print(f"  >>> Saved best model! (Val Acc: {val_acc*100:.2f}%)")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional
import string


class CharacterCNN(nn.Module):
    """
    Convolutional Neural Network for character recognition.
    Outputs probabilities for 36 alphanumeric characters (A-Z, 0-9).
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 36):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes (36 for alphanumeric)
        """
        super(CharacterCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv block 1: Extract low-level features
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions by half

            # Conv block 2: Extract mid-level features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 3: Extract high-level features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 4: Deep feature extraction
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Character mapping (index to character)
        self.idx_to_char = string.digits + string.ascii_uppercase  # '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.char_to_idx = {char: idx for idx, char in enumerate(self.idx_to_char)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

    def predict_char(self, x: torch.Tensor, return_confidence: bool = True) -> Tuple[str, float]:
        """
        Predict character from input image with confidence score.

        Args:
            x: Input tensor of shape (1, channels, height, width)
            return_confidence: Whether to return confidence score

        Returns:
            Tuple of (predicted_character, confidence_score)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_char = self.idx_to_char[predicted_idx.item()]

            if return_confidence:
                return predicted_char, confidence.item()
            return predicted_char


class CharacterDataset(Dataset):
    """
    Dataset class for character images.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: List or array of images
            labels: List or array of character labels
            transform: Optional transforms to apply to images
        """
        self.images = images
        self.labels = labels
        self.transform = transform

        # Character mapping
        self.idx_to_char = string.digits + string.ascii_uppercase
        self.char_to_idx = {char: idx for idx, char in enumerate(self.idx_to_char)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert character label to index
        if isinstance(label, str):
            label = self.char_to_idx[label.upper()]

        return image, label


def get_data_transforms(image_size: Tuple[int, int] = (32, 32)):
    """
    Get data augmentation transforms for training and validation.

    Args:
        image_size: Target image size (height, width)

    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomRotation(10),  # Slight rotation for robustness
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_transform


def create_dataloaders(
    train_images,
    train_labels,
    val_images,
    val_labels,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        train_images: Training images
        train_labels: Training labels
        val_images: Validation images
        val_labels: Validation labels
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform, val_transform = get_data_transforms()

    train_dataset = CharacterDataset(train_images, train_labels, transform=train_transform)
    val_dataset = CharacterDataset(val_images, val_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None
):
    """
    Main training loop (to be implemented when ready to train).

    Args:
        model: The model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (defaults to GPU if available)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_val_acc = 0.0

    print(f"Training on device: {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_char_cnn.pth')
            print(f"  >>> Saved best model (Val Acc: {val_acc:.4f})")
        print("-" * 60)

    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    # Example usage (when ready to train)

    # Initialize model
    model = CharacterCNN(input_channels=1, num_classes=36)

    # Print model architecture
    print("Character CNN Architecture:")
    print("=" * 60)
    print(model)
    print("=" * 60)

    # Print character mapping
    print("\nCharacter Mapping:")
    print(f"Classes: {model.idx_to_char}")
    print(f"Total classes: {len(model.idx_to_char)}")

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 1, 32, 32)  # (batch_size, channels, height, width)
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output.shape}")

    # Test prediction
    char, confidence = model.predict_char(dummy_input)
    print(f"\nPrediction: '{char}' with confidence {confidence:.4f}")

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice available: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    print("\n" + "=" * 60)
    print("Network is ready for training!")
    print("To train, prepare your dataset and call train_model()")
    print("=" * 60)
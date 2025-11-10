from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from src.config import IDX_TO_CHAR

from torch.utils.data import DataLoader


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

    def predict_char(self, x: torch.Tensor, return_confidence: bool = True) -> Tuple[str, float] | str:
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
            predicted_char = IDX_TO_CHAR[int(predicted_idx.item())]

            if return_confidence:
                return predicted_char, confidence.item()
            return predicted_char

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
    model_save_path: Path,
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
        model_save_path: Path to save the model
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (defaults to GPU if available)
    """
    if not model_save_path.exists():
        model_save_path.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler (dynamic learning rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
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
            model_save_name = f"best_model_epoch{epoch+1:03d}_val_acc{val_acc:.4f}.pth"
            torch.save(model.state_dict(), model_save_path / model_save_name)
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
    print(f"Classes: {IDX_TO_CHAR}")
    print(f"Total classes: {len(IDX_TO_CHAR)}")

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

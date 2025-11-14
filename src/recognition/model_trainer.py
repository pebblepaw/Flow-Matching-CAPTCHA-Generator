import random
from pathlib import Path
import numpy as np
import torch
from src.config import SEED
from src.recognition.data_loader import create_dataloaders
from src.recognition.model.character_cnn import CharacterCNN, train_model


def main(model_name: str):
    # Configuration
    TRAIN_IMG_DIR = Path("data/raw/train")
    TRAIN_IMG_CACHE_DIR = Path("data/processed/train_cache")
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TRAIN_VAL_RATIO = 0.8
    NUM_WORKERS = 0  # Set to 0 for Windows
    MODEL_SAVE_PATH = Path("checkpoints") / model_name

    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("Character Recognition Training")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Training directory: {TRAIN_IMG_DIR} and cache directory: {TRAIN_IMG_CACHE_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print("=" * 70)
    print()

    # Create datasets and dataloaders
    train_loader, val_loader = create_dataloaders(
        img_dir=TRAIN_IMG_DIR,
        img_cache_dir=TRAIN_IMG_CACHE_DIR,
        train_val_ratio=TRAIN_VAL_RATIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Create model
    model = CharacterCNN(input_channels=1, num_classes=36)
    model = model.to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_save_path=MODEL_SAVE_PATH,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train character recognition model")
    parser.add_argument("--model-name", type=str, help="Name of the trained model")
    args = parser.parse_args()

    if args.model_name is None:
        parser.error("Model name is required")
    else:
        model_name = args.model_name
    main(model_name=model_name)

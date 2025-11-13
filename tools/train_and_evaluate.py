"""
Train CNN model and evaluate with confusion matrix.

This script:
1. Trains the Character CNN model on tokenized CAPTCHA data
2. Evaluates both character-level and word-level accuracy
3. Generates a 36x36 confusion matrix showing which characters are confused
4. Identifies character pairs that are commonly misclassified
"""

import sys
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SEED, IDX_TO_CHAR, CHAR_TO_IDX
from src.recognition.data_loader import create_dataloaders, CaptchaWordDataset
from src.recognition.model.character_cnn import CharacterCNN, train_model


# Configuration
TRAIN_IMG_DIR = Path("data/filtered/train")
TRAIN_IMG_CACHE_DIR = Path("data/processed/filtered_train_cache")
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_VAL_RATIO = 0.8
NUM_WORKERS = 0
MODEL_SAVE_PATH = Path("checkpoints/character_cnn")


def train_cnn_model():
    """Train the Character CNN model."""
    print("=" * 80)
    print("STEP 1: Training Character Recognition CNN")
    print("=" * 80)

    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training directory: {TRAIN_IMG_DIR}")
    print(f"Cache directory: {TRAIN_IMG_CACHE_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
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

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_save_path=MODEL_SAVE_PATH,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80 + "\n")


def load_best_model(device: torch.device):
    """Load the best trained model."""
    model_files = list(MODEL_SAVE_PATH.glob("best_model_*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No trained model found in {MODEL_SAVE_PATH}")

    # Sort by validation accuracy (extract from filename)
    model_files.sort(key=lambda x: float(x.stem.split('_val_acc')[-1]), reverse=True)
    best_model_path = model_files[0]

    print(f"Loading best model: {best_model_path.name}")

    model = CharacterCNN(input_channels=1, num_classes=36)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    model = model.to(device)

    return model


def evaluate_character_level(model: CharacterCNN, val_loader, device: torch.device):
    """Evaluate character-level accuracy and build confusion matrix."""
    print("\n" + "=" * 80)
    print("STEP 2: Character-Level Evaluation")
    print("=" * 80)

    model.eval()

    # Initialize confusion matrix (36x36)
    confusion_matrix = np.zeros((36, 36), dtype=int)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Update confusion matrix
            for true_label, pred_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[true_label, pred_label] += 1
                total += 1
                if true_label == pred_label:
                    correct += 1

    char_accuracy = correct / total
    print(f"Character-Level Accuracy: {correct}/{total} = {100*char_accuracy:.2f}%")
    print()

    return confusion_matrix, char_accuracy


def evaluate_word_level(model: CharacterCNN, device: torch.device, dataset_type: str = "train"):
    """Evaluate word-level accuracy on train or validation set."""
    print("=" * 80)
    print(f"STEP 3: Word-Level Evaluation ({dataset_type.upper()})")
    print("=" * 80)

    # Load word dataset
    from torch.utils.data import random_split
    full_dataset = CaptchaWordDataset(
        image_dir=TRAIN_IMG_DIR,
        cache_dir=TRAIN_IMG_CACHE_DIR
    )

    # Split the same way as character-level
    train_len = int(len(full_dataset) * TRAIN_VAL_RATIO)
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Select dataset
    dataset = train_dataset if dataset_type == "train" else val_dataset

    model.eval()

    word_correct = 0
    total_words = 0
    char_correct = 0
    total_chars = 0

    # Track misclassified character pairs
    char_confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for i in range(len(dataset)):
            tokens_tensors, label_indices = dataset[i]

            # Ground truth word
            gt_word = ''.join(IDX_TO_CHAR[idx] for idx in label_indices)
            total_words += 1
            total_chars += len(gt_word)

            # Predict each character
            pred_chars = []
            for token in tokens_tensors:
                token = torch.stack([token]).to(device)
                outputs = model(token)
                _, predicted_idx = torch.max(outputs, 1)
                pred_char = IDX_TO_CHAR[int(predicted_idx.item())]
                pred_chars.append(pred_char)

            pred_word = ''.join(pred_chars)

            # Word-level accuracy (all characters must match)
            if pred_word == gt_word:
                word_correct += 1

            # Character-level accuracy and confusion tracking
            for gt_char, pred_char in zip(gt_word, pred_word):
                if gt_char == pred_char:
                    char_correct += 1
                else:
                    char_confusion[gt_char][pred_char] += 1

    word_accuracy = word_correct / total_words if total_words > 0 else 0
    char_accuracy = char_correct / total_chars if total_chars > 0 else 0

    print(f"Word-Level Accuracy:      {word_correct}/{total_words} = {100*word_accuracy:.2f}%")
    print(f"Character-Level Accuracy: {char_correct}/{total_chars} = {100*char_accuracy:.2f}%")
    print()

    return word_accuracy, char_accuracy, char_confusion


def plot_confusion_matrix(confusion_matrix: np.ndarray, output_file: str = "confusion_matrix.png"):
    """Plot and save the confusion matrix."""
    print("=" * 80)
    print("STEP 4: Generating Confusion Matrix Visualization")
    print("=" * 80)

    # Create character labels
    chars = [IDX_TO_CHAR[i] for i in range(36)]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Plot heatmap
    sns.heatmap(
        confusion_matrix,
        xticklabels=chars,
        yticklabels=chars,
        annot=False,  # Too many cells to annotate
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        ax=ax,
        square=True
    )

    ax.set_xlabel('Predicted Character', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Character', fontweight='bold', fontsize=12)
    ax.set_title('36x36 Character Confusion Matrix', fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix: {output_file}")
    print()


def analyze_confusion_patterns(confusion_matrix: np.ndarray, char_confusion: dict):
    """Analyze and report common character confusions."""
    print("=" * 80)
    print("STEP 5: Character Confusion Analysis")
    print("=" * 80)

    # Find most confused character pairs
    confused_pairs = []
    for i in range(36):
        for j in range(36):
            if i != j and confusion_matrix[i, j] > 0:
                true_char = IDX_TO_CHAR[i]
                pred_char = IDX_TO_CHAR[j]
                count = confusion_matrix[i, j]
                total = np.sum(confusion_matrix[i, :])
                error_rate = count / total if total > 0 else 0
                confused_pairs.append((true_char, pred_char, count, error_rate))

    # Sort by count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 20 Most Confused Character Pairs:")
    print("-" * 80)
    print(f"{'True':<6} {'Predicted':<10} {'Count':<8} {'Error Rate':<12}")
    print("-" * 80)

    for true_char, pred_char, count, error_rate in confused_pairs[:20]:
        print(f"'{true_char}' -> '{pred_char}' {count:>10} {error_rate*100:>10.2f}%")

    # Find characters with highest error rates
    print("\n\nCharacters with Highest Overall Error Rates:")
    print("-" * 80)
    print(f"{'Character':<12} {'Correct':<10} {'Wrong':<10} {'Error Rate':<12}")
    print("-" * 80)

    char_errors = []
    for i in range(36):
        char = IDX_TO_CHAR[i]
        correct = confusion_matrix[i, i]
        wrong = np.sum(confusion_matrix[i, :]) - correct
        total = correct + wrong
        error_rate = wrong / total if total > 0 else 0
        char_errors.append((char, correct, wrong, error_rate))

    char_errors.sort(key=lambda x: x[3], reverse=True)

    for char, correct, wrong, error_rate in char_errors[:15]:
        print(f"'{char}' {correct:>15} {wrong:>10} {error_rate*100:>10.2f}%")

    print("\n" + "=" * 80)


def main():
    print("\n" + "=" * 80)
    print("CAPTCHA CNN Training and Evaluation Pipeline")
    print("=" * 80 + "\n")

    # Step 1: Train model
    train_cnn_model()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best model
    model = load_best_model(device)

    # Step 2: Evaluate character-level (on validation set)
    _, val_loader = create_dataloaders(
        img_dir=TRAIN_IMG_DIR,
        img_cache_dir=TRAIN_IMG_CACHE_DIR,
        train_val_ratio=TRAIN_VAL_RATIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    confusion_matrix, char_acc_val = evaluate_character_level(model, val_loader, device)

    # Step 3: Evaluate word-level on training set
    word_acc_train, char_acc_train, char_confusion_train = evaluate_word_level(model, device, dataset_type="train")

    # Step 4: Evaluate word-level on validation set
    word_acc_val, char_acc_val_word, char_confusion_val = evaluate_word_level(model, device, dataset_type="val")

    # Step 5: Visualize confusion matrix
    plot_confusion_matrix(confusion_matrix, output_file="confusion_matrix.png")

    # Step 6: Analyze confusion patterns
    analyze_confusion_patterns(confusion_matrix, char_confusion_train)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Character-Level Accuracy (validation): {100*char_acc_val:.2f}%")
    print(f"Character-Level Accuracy (training):   {100*char_acc_train:.2f}%")
    print(f"Word-Level Accuracy (validation):      {100*word_acc_val:.2f}%")
    print(f"Word-Level Accuracy (training):        {100*word_acc_train:.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

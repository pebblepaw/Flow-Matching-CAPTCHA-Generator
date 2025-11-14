"""
Generate confusion matrix heatmap for top 10 most common characters.

This script evaluates the trained character recognition model and generates
a confusion matrix visualization showing the top 10 most frequently occurring
characters in the test set.
"""

import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import IDX_TO_CHAR, CHAR_TO_IDX, SEED
from src.recognition.data_loader import CaptchaWordDataset
from src.recognition.model.character_cnn import CharacterCNN


def load_trained_model(model_path: str, device: torch.device) -> CharacterCNN:
    """Load the trained character recognition model from checkpoint."""
    model = CharacterCNN(input_channels=1, num_classes=36)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (epoch: {checkpoint.get('epoch', 'N/A')}, "
              f"val_acc: {checkpoint.get('val_acc', 'N/A'):.4f})")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model from: {model_path}")

    model.eval()
    model = model.to(device)
    return model


def predict_characters(
    model: CharacterCNN,
    tokens: List[torch.Tensor],
    device: torch.device
) -> List[Tuple[str, float]]:
    """
    Predict characters from token tensors.

    Args:
        model: Trained CharacterCNN model
        tokens: List of token tensors
        device: torch device

    Returns:
        List of tuples (predicted_char, confidence)
    """
    predictions = []

    for token in tokens:
        token = torch.stack([token]).to(device)
        pred_char, conf = model.predict_char(token, return_confidence=True)
        predictions.append((pred_char, conf))

    return predictions


def collect_predictions(
    model: CharacterCNN,
    dataset: CaptchaWordDataset,
    device: torch.device
) -> Tuple[List[str], List[str]]:
    """
    Collect all ground truth and predicted characters from the dataset.

    Args:
        model: Trained model
        dataset: CaptchaWordDataset
        device: torch device

    Returns:
        Tuple of (all_ground_truth_chars, all_predicted_chars)
    """
    all_gt_chars = []
    all_pred_chars = []

    print(f"Processing {len(dataset)} samples...")

    for sample_idx in range(len(dataset)):
        tokens_tensors, label_indices = dataset[sample_idx]

        # Convert label indices to characters
        gt_chars = [IDX_TO_CHAR[idx] for idx in label_indices]

        # Predict characters
        predictions = predict_characters(model, tokens_tensors, device)
        pred_chars = [pred_char for pred_char, _ in predictions]

        # Add to lists
        all_gt_chars.extend(gt_chars)
        all_pred_chars.extend(pred_chars)

        if (sample_idx + 1) % 100 == 0:
            print(f"  Processed {sample_idx + 1}/{len(dataset)} samples...")

    print(f"Total characters: {len(all_gt_chars)}")
    return all_gt_chars, all_pred_chars


def get_top_n_classes(ground_truth: List[str], n: int = 10, most_common: bool = True) -> List[str]:
    """
    Get the top N most or least frequent classes from ground truth labels.

    Args:
        ground_truth: List of ground truth characters
        n: Number of top classes to return
        most_common: If True, return most common; if False, return least common

    Returns:
        List of top N most/least frequent characters
    """
    counter = Counter(ground_truth)

    if most_common:
        top_n = [char for char, count in counter.most_common(n)]
        print(f"\nTop {n} most common characters:")
        for i, (char, count) in enumerate(counter.most_common(n), 1):
            print(f"  {i}. '{char}': {count} occurrences")
    else:
        # Get least common by reversing the most_common list
        all_common = counter.most_common()
        least_common = all_common[-n:][::-1]  # Get last n and reverse
        top_n = [char for char, count in least_common]
        print(f"\nTop {n} least common characters:")
        for i, (char, count) in enumerate(least_common, 1):
            print(f"  {i}. '{char}': {count} occurrences")

    return top_n


def get_top_n_by_accuracy(
    ground_truth: List[str],
    predictions: List[str],
    n: int = 10,
    highest: bool = True
) -> List[str]:
    """
    Get the top N characters by accuracy (highest or lowest).

    Args:
        ground_truth: List of ground truth labels
        predictions: List of predicted labels
        n: Number of top classes to return
        highest: If True, return highest accuracy; if False, return lowest

    Returns:
        List of top N characters by accuracy
    """
    # Calculate per-class accuracy
    char_stats = {}
    for gt, pred in zip(ground_truth, predictions):
        if gt not in char_stats:
            char_stats[gt] = {'correct': 0, 'total': 0}
        char_stats[gt]['total'] += 1
        if gt == pred:
            char_stats[gt]['correct'] += 1

    # Calculate accuracy for each character
    char_accuracy = {}
    for char, stats in char_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        char_accuracy[char] = (accuracy, stats['correct'], stats['total'])

    # Sort by accuracy
    sorted_chars = sorted(char_accuracy.items(), key=lambda x: x[1][0], reverse=highest)
    top_n_chars = sorted_chars[:n]

    if highest:
        print(f"\nTop {n} highest accuracy characters:")
    else:
        print(f"\nTop {n} lowest accuracy characters:")

    for i, (char, (acc, correct, total)) in enumerate(top_n_chars, 1):
        print(f"  {i}. '{char}': {acc:.2%} ({correct}/{total})")

    return [char for char, _ in top_n_chars]


def create_confusion_matrix(
    ground_truth: List[str],
    predictions: List[str],
    classes: List[str]
) -> np.ndarray:
    """
    Create confusion matrix for specified classes.

    Args:
        ground_truth: List of ground truth labels
        predictions: List of predicted labels
        classes: List of classes to include in confusion matrix

    Returns:
        Confusion matrix as numpy array
    """
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    class_to_idx = {char: idx for idx, char in enumerate(classes)}

    for gt, pred in zip(ground_truth, predictions):
        if gt in class_to_idx and pred in class_to_idx:
            matrix[class_to_idx[gt], class_to_idx[pred]] += 1

    return matrix


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    output_path: str,
    normalize: bool = True,
    dpi: int = 120,
    title_suffix: str = ""
):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        classes: List of class labels
        output_path: Path to save the figure
        normalize: Whether to normalize by row (true labels)
        dpi: DPI for saved figure
    """
    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
        cmap = 'Blues'
    else:
        cm_display = cm
        fmt = 'd'
        cmap = 'Blues'

    # Dynamically adjust figure size based on number of classes
    n_classes = len(classes)
    if n_classes > 20:
        # For large matrices (like 36x36), use bigger figure and smaller font
        figsize = (20, 18)
        fontsize = 6
        title_fontsize = 16
    elif n_classes > 15:
        figsize = (16, 14)
        fontsize = 7
        title_fontsize = 14
    else:
        figsize = (12, 10)
        fontsize = 9
        title_fontsize = 14

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                # For large matrices, only show percentage if significant
                if n_classes > 20:
                    # Only show text for values > 5% or diagonal
                    if cm_display[i, j] > 0.05 or i == j:
                        if cm_display[i, j] > 0.995:
                            text = f'{cm_display[i, j]:.0%}'
                        else:
                            text = f'{cm_display[i, j]:.0%}'
                    else:
                        text = ''
                else:
                    # Show percentage for normalized matrix
                    text = f'{cm_display[i, j]:.1%}\n({cm[i, j]})'
            else:
                # Show count for non-normalized matrix
                text = f'{cm[i, j]}'

            if text:  # Only add text if not empty
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="white" if cm_display[i, j] > thresh else "black",
                       fontsize=fontsize)

    # Add title
    if title_suffix:
        title = f'Confusion Matrix - {title_suffix}'
    else:
        title = 'Confusion Matrix - Top 10 Characters'
    if normalize:
        title += ' (Normalized by Row)'
    ax.set_title(title, fontsize=title_fontsize, weight='bold', pad=20)

    # Calculate overall accuracy for top 10
    correct = np.trace(cm)
    total = cm.sum()
    accuracy = correct / total if total > 0 else 0

    # Add accuracy text
    fig.text(0.5, 0.02,
             f'Accuracy on Top 10: {correct}/{total} = {accuracy:.2%}',
             ha='center', fontsize=11, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")
    print(f"Accuracy on top 10 characters: {accuracy:.2%}")


def main(
    model_path: str = "char_recognizer.pth",
    image_dir: str = "data/raw/test",
    cache_dir: str = "data/processed/test_cache",
    output_file: str = "confusion_matrix_top10.png",
    top_n: int = 10,
    normalize: bool = True,
    dpi: int = 120,
    generate_both: bool = False,
    generate_all: bool = False,
    by_accuracy: bool = False
):
    """
    Main function to generate confusion matrix heatmap.

    Args:
        model_path: Path to trained model checkpoint
        image_dir: Directory containing test images
        cache_dir: Directory for caching preprocessed data
        output_file: Output filename for confusion matrix
        top_n: Number of top classes to include (default: 10)
        normalize: Whether to normalize confusion matrix by row
        dpi: DPI for saved figure
        generate_both: Generate both most and least common matrices
        generate_all: Generate confusion matrix for all 36 characters
        by_accuracy: Sort by accuracy instead of frequency
    """
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Confusion Matrix Generator - Top 10 Characters")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print(f"Image directory: {image_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Output file: {output_file}")
    print(f"Top N classes: {top_n}")
    print(f"Normalize: {normalize}")
    print(f"Generate both (most + least): {generate_both}")
    print("=" * 70)
    print()

    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path, device)
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = CaptchaWordDataset(
        image_dir=Path(image_dir),
        cache_dir=Path(cache_dir)
    )
    print(f"Dataset loaded: {len(dataset)} valid samples, {dataset.skipped} failed")
    print()

    # Collect predictions
    print("Collecting predictions...")
    all_gt_chars, all_pred_chars = collect_predictions(model, dataset, device)
    print()

    # Generate all 36 characters confusion matrix if requested
    if generate_all:
        print("\n" + "=" * 70)
        print("ALL 36 CHARACTERS")
        print("=" * 70)

        # Get all unique characters and sort them
        all_classes = sorted(list(set(all_gt_chars)))
        print(f"\nAll {len(all_classes)} characters: {all_classes}")
        print()

        print("Creating confusion matrix for all characters...")
        cm_all = create_confusion_matrix(all_gt_chars, all_pred_chars, all_classes)
        print(f"Confusion matrix shape: {cm_all.shape}")
        print()

        all_output = output_file.replace('.png', '_all36.png')
        print(f"Plotting confusion matrix for all 36 -> {all_output}...")
        plot_confusion_matrix(
            cm_all, all_classes, all_output,
            normalize=normalize, dpi=150,  # Higher DPI for larger matrix
            title_suffix="All 36 Characters"
        )
        print()

    # Generate by accuracy if requested
    if by_accuracy:
        # Highest accuracy
        print("\n" + "=" * 70)
        print("HIGHEST ACCURACY CHARACTERS")
        print("=" * 70)
        high_acc_classes = get_top_n_by_accuracy(
            all_gt_chars, all_pred_chars, n=top_n, highest=True
        )
        print()

        print("Creating confusion matrix for highest accuracy...")
        cm_high_acc = create_confusion_matrix(all_gt_chars, all_pred_chars, high_acc_classes)
        print(f"Confusion matrix shape: {cm_high_acc.shape}")
        print()

        high_acc_output = output_file.replace('.png', '_highest_accuracy.png')
        print(f"Plotting confusion matrix for highest accuracy -> {high_acc_output}...")
        plot_confusion_matrix(
            cm_high_acc, high_acc_classes, high_acc_output,
            normalize=normalize, dpi=dpi,
            title_suffix="Top 10 Highest Accuracy Characters"
        )
        print()

        # Lowest accuracy
        print("\n" + "=" * 70)
        print("LOWEST ACCURACY CHARACTERS")
        print("=" * 70)
        low_acc_classes = get_top_n_by_accuracy(
            all_gt_chars, all_pred_chars, n=top_n, highest=False
        )
        print()

        print("Creating confusion matrix for lowest accuracy...")
        cm_low_acc = create_confusion_matrix(all_gt_chars, all_pred_chars, low_acc_classes)
        print(f"Confusion matrix shape: {cm_low_acc.shape}")
        print()

        low_acc_output = output_file.replace('.png', '_lowest_accuracy.png')
        print(f"Plotting confusion matrix for lowest accuracy -> {low_acc_output}...")
        plot_confusion_matrix(
            cm_low_acc, low_acc_classes, low_acc_output,
            normalize=normalize, dpi=dpi,
            title_suffix="Top 10 Lowest Accuracy Characters"
        )
        print()

    # Generate by frequency if requested (original behavior)
    if generate_both:
        # Most common
        print("\n" + "=" * 70)
        print("MOST COMMON CHARACTERS")
        print("=" * 70)
        top_classes = get_top_n_classes(all_gt_chars, n=top_n, most_common=True)
        print()

        print("Creating confusion matrix for most common...")
        cm_most = create_confusion_matrix(all_gt_chars, all_pred_chars, top_classes)
        print(f"Confusion matrix shape: {cm_most.shape}")
        print()

        most_output = output_file.replace('.png', '_most_common.png')
        print(f"Plotting confusion matrix for most common -> {most_output}...")
        plot_confusion_matrix(
            cm_most, top_classes, most_output,
            normalize=normalize, dpi=dpi,
            title_suffix="Top 10 Most Common Characters"
        )
        print()

        # Least common
        print("\n" + "=" * 70)
        print("LEAST COMMON CHARACTERS")
        print("=" * 70)
        bottom_classes = get_top_n_classes(all_gt_chars, n=top_n, most_common=False)
        print()

        print("Creating confusion matrix for least common...")
        cm_least = create_confusion_matrix(all_gt_chars, all_pred_chars, bottom_classes)
        print(f"Confusion matrix shape: {cm_least.shape}")
        print()

        least_output = output_file.replace('.png', '_least_common.png')
        print(f"Plotting confusion matrix for least common -> {least_output}...")
        plot_confusion_matrix(
            cm_least, bottom_classes, least_output,
            normalize=normalize, dpi=dpi,
            title_suffix="Top 10 Least Common Characters"
        )
        print()

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate confusion matrix heatmap for top N characters"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="char_recognizer.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/raw/test",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/processed/test_cache",
        help="Directory for caching preprocessed data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="confusion_matrix_top10.png",
        help="Output filename for confusion matrix"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top classes to include (default: 10)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize confusion matrix"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="DPI for saved figure (default: 120)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate both most and least common confusion matrices"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate confusion matrix for all 36 characters"
    )
    parser.add_argument(
        "--by-accuracy",
        action="store_true",
        help="Generate confusion matrices sorted by accuracy (highest and lowest)"
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        output_file=args.output,
        top_n=args.top_n,
        normalize=not args.no_normalize,
        dpi=args.dpi,
        generate_both=args.both,
        generate_all=args.all,
        by_accuracy=args.by_accuracy
    )

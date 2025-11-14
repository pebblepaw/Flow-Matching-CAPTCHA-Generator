"""
Class Balance Analysis and Theoretical Optimum Calculator

Analyzes the class distribution in the dataset and calculates:
1. Class balance metrics
2. Theoretical accuracy upper bounds
3. Impact analysis for improving specific characters
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import IDX_TO_CHAR, SEED
from src.recognition.data_loader import CaptchaWordDataset
from src.recognition.model.character_cnn import CharacterCNN


def load_trained_model(model_path: str, device: torch.device) -> CharacterCNN:
    """Load the trained character recognition model from checkpoint."""
    model = CharacterCNN(input_channels=1, num_classes=36)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model = model.to(device)
    return model


def collect_predictions(
    model: CharacterCNN,
    dataset: CaptchaWordDataset,
    device: torch.device
) -> Tuple[List[str], List[str]]:
    """Collect all ground truth and predicted characters."""
    all_gt_chars = []
    all_pred_chars = []

    for sample_idx in range(len(dataset)):
        tokens_tensors, label_indices = dataset[sample_idx]
        gt_chars = [IDX_TO_CHAR[idx] for idx in label_indices]

        # Predict
        predictions = []
        for token in tokens_tensors:
            token = torch.stack([token]).to(device)
            pred_char, _ = model.predict_char(token, return_confidence=True)
            predictions.append(pred_char)

        all_gt_chars.extend(gt_chars)
        all_pred_chars.extend(predictions)

    return all_gt_chars, all_pred_chars


def analyze_class_balance(ground_truth: List[str]) -> Dict:
    """
    Analyze class balance in the dataset.

    Returns:
        Dictionary with balance metrics
    """
    counter = Counter(ground_truth)
    total = len(ground_truth)

    # Calculate statistics
    counts = list(counter.values())
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    min_count = min(counts)
    max_count = max(counts)

    # Balance ratio (max/min)
    imbalance_ratio = max_count / min_count

    # Coefficient of variation
    cv = std_count / mean_count if mean_count > 0 else 0

    return {
        'total_chars': total,
        'num_classes': len(counter),
        'mean_count': mean_count,
        'std_count': std_count,
        'min_count': min_count,
        'max_count': max_count,
        'imbalance_ratio': imbalance_ratio,
        'cv': cv,
        'distribution': counter
    }


def calculate_per_char_accuracy(
    ground_truth: List[str],
    predictions: List[str]
) -> Dict[str, Dict]:
    """
    Calculate per-character accuracy statistics.

    Returns:
        Dictionary mapping char -> {accuracy, correct, total, weight}
    """
    char_stats = {}
    total_chars = len(ground_truth)

    for gt, pred in zip(ground_truth, predictions):
        if gt not in char_stats:
            char_stats[gt] = {'correct': 0, 'total': 0}
        char_stats[gt]['total'] += 1
        if gt == pred:
            char_stats[gt]['correct'] += 1

    # Calculate accuracy and weight for each character
    for char, stats in char_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        stats['weight'] = stats['total'] / total_chars

    return char_stats


def calculate_theoretical_limits(char_stats: Dict[str, Dict]) -> Dict:
    """
    Calculate theoretical accuracy limits under different scenarios.

    Returns:
        Dictionary with various theoretical limits
    """
    # Current weighted accuracy
    current_accuracy = sum(
        stats['accuracy'] * stats['weight']
        for stats in char_stats.values()
    )

    # Scenario 1: Perfect recognition (100% for all)
    perfect_accuracy = 1.0

    # Scenario 2: All chars achieve best current accuracy
    best_accuracy = max(stats['accuracy'] for stats in char_stats.values())
    all_best_accuracy = best_accuracy  # Would be 100% if applied to all

    # Scenario 3: Fix only the worst performing chars to average
    avg_accuracy = sum(stats['accuracy'] for stats in char_stats.values()) / len(char_stats)

    # Calculate impact of fixing worst chars to average
    sorted_chars = sorted(
        char_stats.items(),
        key=lambda x: x[1]['accuracy']
    )

    # Fix bottom 25% to average
    n_to_fix = len(sorted_chars) // 4
    fixed_bottom_25_accuracy = current_accuracy
    for char, stats in sorted_chars[:n_to_fix]:
        improvement = (avg_accuracy - stats['accuracy']) * stats['weight']
        fixed_bottom_25_accuracy += improvement

    # Fix bottom 50% to average
    n_to_fix = len(sorted_chars) // 2
    fixed_bottom_50_accuracy = current_accuracy
    for char, stats in sorted_chars[:n_to_fix]:
        improvement = (avg_accuracy - stats['accuracy']) * stats['weight']
        fixed_bottom_50_accuracy += improvement

    return {
        'current_accuracy': current_accuracy,
        'perfect_accuracy': perfect_accuracy,
        'all_best_accuracy': all_best_accuracy,
        'avg_char_accuracy': avg_accuracy,
        'fixed_bottom_25': fixed_bottom_25_accuracy,
        'fixed_bottom_50': fixed_bottom_50_accuracy,
        'gap_to_perfect': perfect_accuracy - current_accuracy,
        'best_char_accuracy': best_accuracy
    }


def calculate_improvement_impact(char_stats: Dict[str, Dict]) -> List[Tuple]:
    """
    Calculate impact of improving each character by 1%.

    Returns:
        List of (char, current_acc, impact, total_count) sorted by impact
    """
    impacts = []

    for char, stats in char_stats.items():
        # Impact of 1% improvement on this character
        impact = 0.01 * stats['weight']
        impacts.append((
            char,
            stats['accuracy'],
            impact,
            stats['total']
        ))

    # Sort by impact (descending)
    impacts.sort(key=lambda x: x[2], reverse=True)

    return impacts


def plot_analysis(
    balance_metrics: Dict,
    char_stats_test: Dict,
    avg_char_accuracy_test: float,
    output_file: str = "class_balance_analysis.png",
    dpi: int = 120
):
    """Create comprehensive visualization of class balance analysis."""

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # 1. Class distribution (bar chart) - from entire filtered dataset
    ax1 = fig.add_subplot(gs[0, 0])
    chars = sorted(balance_metrics['distribution'].keys())
    counts = [balance_metrics['distribution'][c] for c in chars]
    colors = ['#1f77b4' if c.isdigit() else '#ff7f0e' for c in chars]
    ax1.bar(chars, counts, color=colors)
    ax1.set_xlabel('Character', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Class Distribution (Blue=Digits, Orange=Letters)', fontsize=14, weight='bold')
    ax1.axhline(y=balance_metrics['mean_count'], color='r', linestyle='--',
                label=f"Mean: {balance_metrics['mean_count']:.1f}", linewidth=2)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Per-character accuracy (top 10 lowest) - from test set only
    ax2 = fig.add_subplot(gs[0, 1])
    chars_by_acc = sorted(char_stats_test.items(), key=lambda x: x[1]['accuracy'])
    # Take bottom 10 (lowest accuracy)
    bottom_10 = chars_by_acc[:10]
    chars = [c for c, _ in bottom_10]
    accs = [s['accuracy'] for _, s in bottom_10]
    colors = ['#d62728' if acc < 0.8 else '#ff7f0e' for acc in accs]
    ax2.barh(chars, accs, color=colors)
    ax2.set_xlabel('Accuracy', fontsize=12)
    ax2.set_ylabel('Character', fontsize=12)
    ax2.set_title('Bottom 10 Characters by Accuracy (Test Set)', fontsize=14, weight='bold')
    ax2.axvline(x=avg_char_accuracy_test, color='b', linestyle='--',
                label=f"Mean: {avg_char_accuracy_test:.2%}", linewidth=2)
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 1.05)

    # Main title
    fig.suptitle('Character-Level Analysis',
                 fontsize=16, weight='bold', y=0.995)

    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\nAnalysis visualization saved to: {output_file}")


def main(
    model_path: str = "char_recognizer.pth",
    image_dirs: List[str] = None,
    cache_dirs: List[str] = None,
    output_file: str = "class_balance_analysis.png"
):
    """Main analysis function."""

    # Default to using both filtered/train and filtered/test
    if image_dirs is None:
        image_dirs = ["data/filtered/train", "data/filtered/test"]
    if cache_dirs is None:
        cache_dirs = ["data/processed/filtered_train_cache", "data/processed/filtered_test_cache"]

    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Class Balance and Theoretical Optimum Analysis")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Image directories: {image_dirs}")
    print("=" * 70)
    print()

    # Load model
    print("Loading model...")
    model = load_trained_model(model_path, device)

    # Load datasets separately
    print("Loading datasets...")
    all_gt_chars = []  # For class distribution (all datasets)
    test_gt_chars = []  # For accuracy analysis (test only)
    test_pred_chars = []  # For accuracy analysis (test only)

    for image_dir, cache_dir in zip(image_dirs, cache_dirs):
        print(f"\n  Processing: {image_dir}")
        dataset = CaptchaWordDataset(
            image_dir=Path(image_dir),
            cache_dir=Path(cache_dir)
        )
        print(f"  Loaded: {len(dataset)} samples")

        # Collect predictions for this dataset
        gt_chars, pred_chars = collect_predictions(model, dataset, device)

        # Add to overall class distribution
        all_gt_chars.extend(gt_chars)

        # Only add to test accuracy if this is test set
        if "test" in str(image_dir):
            test_gt_chars.extend(gt_chars)
            test_pred_chars.extend(pred_chars)
            print(f"  Added to test accuracy analysis: {len(gt_chars)} characters")

    print(f"\nTotal characters for class distribution: {len(all_gt_chars)}")
    print(f"Total characters for accuracy analysis (test only): {len(test_gt_chars)}")
    print()

    # Analyze class balance on all data
    print("Analyzing class balance (entire filtered dataset)...")
    balance_metrics = analyze_class_balance(all_gt_chars)
    print()

    # Calculate per-character accuracy on test data only
    print("Calculating per-character accuracy (test set only)...")
    char_stats_test = calculate_per_char_accuracy(test_gt_chars, test_pred_chars)

    # Calculate average test accuracy
    avg_char_accuracy_test = sum(
        stats['accuracy'] * stats['weight']
        for stats in char_stats_test.values()
    )
    print()

    # Print detailed results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("Class Balance Metrics (Entire Filtered Dataset):")
    print(f"  Total Characters: {balance_metrics['total_chars']:,}")
    print(f"  Imbalance Ratio: {balance_metrics['imbalance_ratio']:.2f}x")
    print(f"  Most Common: {max(balance_metrics['distribution'].items(), key=lambda x: x[1])}")
    print(f"  Least Common: {min(balance_metrics['distribution'].items(), key=lambda x: x[1])}")
    print()

    print("Test Set Accuracy Metrics:")
    print(f"  Total Test Characters: {len(test_gt_chars):,}")
    print(f"  Average Character Accuracy: {avg_char_accuracy_test:.4f} ({avg_char_accuracy_test:.2%})")
    print()

    print("Bottom 10 Characters by Accuracy (Test Set):")
    chars_by_acc = sorted(char_stats_test.items(), key=lambda x: x[1]['accuracy'])
    for i, (char, stats) in enumerate(chars_by_acc[:10], 1):
        print(f"  {i}. '{char}': {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print()

    # Generate visualization
    print("Generating visualization...")
    plot_analysis(balance_metrics, char_stats_test, avg_char_accuracy_test, output_file)
    print()

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze class balance and calculate theoretical optimum"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="char_recognizer.pth",
        help="Path to trained model"
    )
    parser.add_argument(
        "--image-dirs",
        type=str,
        nargs='+',
        default=None,
        help="Image directories (default: data/filtered/train data/filtered/test)"
    )
    parser.add_argument(
        "--cache-dirs",
        type=str,
        nargs='+',
        default=None,
        help="Cache directories (default: data/processed/filtered_train_cache data/processed/filtered_test_cache)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="class_balance_analysis.png",
        help="Output filename"
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        image_dirs=args.image_dirs,
        cache_dirs=args.cache_dirs,
        output_file=args.output
    )

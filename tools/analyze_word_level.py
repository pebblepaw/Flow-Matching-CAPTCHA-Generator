"""
Word-Level Analysis for CAPTCHA Recognition

Analyzes:
1. Word length distribution
2. Word-level accuracy (all characters must be correct)
3. Theoretical word-level accuracy limits
"""

import random
from pathlib import Path
from typing import List, Tuple, Dict
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


def evaluate_word_level(
    model: CharacterCNN,
    dataset: CaptchaWordDataset,
    device: torch.device
) -> Tuple[List[int], List[bool], List[float]]:
    """
    Evaluate model at word level.

    Returns:
        - List of word lengths
        - List of word correctness (True if all chars correct)
        - List of per-word character accuracy
    """
    word_lengths = []
    word_correct = []
    word_char_accuracies = []

    for sample_idx in range(len(dataset)):
        tokens_tensors, label_indices = dataset[sample_idx]

        # Ground truth
        gt_chars = [IDX_TO_CHAR[idx] for idx in label_indices]
        word_length = len(gt_chars)

        # Predict
        pred_chars = []
        for token in tokens_tensors:
            token = torch.stack([token]).to(device)
            pred_char, _ = model.predict_char(token, return_confidence=True)
            pred_chars.append(pred_char)

        # Check if word is correct (all characters must match)
        is_correct = all(gt == pred for gt, pred in zip(gt_chars, pred_chars))

        # Calculate character-level accuracy for this word
        char_correct = sum(1 for gt, pred in zip(gt_chars, pred_chars) if gt == pred)
        char_accuracy = char_correct / word_length if word_length > 0 else 0

        word_lengths.append(word_length)
        word_correct.append(is_correct)
        word_char_accuracies.append(char_accuracy)

    return word_lengths, word_correct, word_char_accuracies


def analyze_word_length_distribution(word_lengths: List[int]) -> Dict:
    """Analyze distribution of word lengths."""
    counter = Counter(word_lengths)

    return {
        'distribution': counter,
        'mean': np.mean(word_lengths),
        'std': np.std(word_lengths),
        'min': min(word_lengths),
        'max': max(word_lengths),
        'median': np.median(word_lengths),
        'mode': counter.most_common(1)[0][0] if counter else 0
    }


def calculate_theoretical_word_accuracy(
    word_lengths: List[int],
    char_accuracy: float
) -> Dict:
    """
    Calculate theoretical word-level accuracy given character accuracy.

    Word is correct only if ALL characters are correct.
    For a word of length n with char accuracy p:
        P(word correct) = p^n
    """
    theoretical_accuracies = {}

    # Group by word length
    length_counter = Counter(word_lengths)

    for length, count in length_counter.items():
        # Theoretical word accuracy for this length
        word_acc = char_accuracy ** length
        theoretical_accuracies[length] = {
            'count': count,
            'weight': count / len(word_lengths),
            'theoretical_accuracy': word_acc
        }

    # Weighted average theoretical word accuracy
    weighted_avg = sum(
        stats['theoretical_accuracy'] * stats['weight']
        for stats in theoretical_accuracies.values()
    )

    return {
        'by_length': theoretical_accuracies,
        'weighted_average': weighted_avg
    }


def calculate_word_level_limits(
    word_lengths: List[int],
    word_correct: List[bool]
) -> Dict:
    """Calculate word-level accuracy under different scenarios."""

    # Current word-level accuracy
    current_word_acc = sum(word_correct) / len(word_correct)

    # Theoretical limits with different character accuracies
    char_accuracies = [0.90, 0.92, 0.95, 0.98, 0.99, 1.00]
    theoretical_limits = {}

    for char_acc in char_accuracies:
        theory = calculate_theoretical_word_accuracy(word_lengths, char_acc)
        theoretical_limits[char_acc] = theory['weighted_average']

    return {
        'current_word_accuracy': current_word_acc,
        'theoretical_at_char_accuracy': theoretical_limits
    }


def analyze_by_word_length(
    word_lengths: List[int],
    word_correct: List[bool],
    word_char_accuracies: List[float]
) -> Dict:
    """Analyze accuracy grouped by word length."""

    length_stats = {}

    for length, correct, char_acc in zip(word_lengths, word_correct, word_char_accuracies):
        if length not in length_stats:
            length_stats[length] = {
                'word_correct': 0,
                'word_total': 0,
                'char_accuracies': []
            }

        length_stats[length]['word_total'] += 1
        if correct:
            length_stats[length]['word_correct'] += 1
        length_stats[length]['char_accuracies'].append(char_acc)

    # Calculate averages
    for length, stats in length_stats.items():
        stats['word_accuracy'] = stats['word_correct'] / stats['word_total']
        stats['avg_char_accuracy'] = np.mean(stats['char_accuracies'])

    return length_stats


def plot_word_analysis(
    length_dist: Dict,
    length_stats: Dict,
    word_limits: Dict,
    output_file: str = "word_level_analysis.png",
    dpi: int = 120
):
    """Create word-level analysis visualization."""

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # 1. Word accuracy by word length
    ax1 = fig.add_subplot(gs[0, 0])
    lengths_sorted = sorted(length_stats.keys())
    word_accs = [length_stats[l]['word_accuracy'] for l in lengths_sorted]

    bars = ax1.bar(lengths_sorted, word_accs, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Word Length (# of characters)', fontsize=12, labelpad=15)
    ax1.set_ylabel('Word Accuracy', fontsize=12)
    ax1.set_title('Word Accuracy by Word Length', fontsize=14, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(lengths_sorted)

    # Add overall average word accuracy line
    overall_avg = word_limits['current_word_accuracy']
    ax1.axhline(y=overall_avg, color='r', linestyle='--',
                label=f"Overall Avg: {overall_avg:.1%}", linewidth=2)
    ax1.legend(fontsize=10)

    # Add accuracy labels on bars
    for length, acc, bar in zip(lengths_sorted, word_accs, bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10, weight='bold')
        # Add sample count below x-axis label
        count = length_stats[length]['word_total']
        ax1.text(bar.get_x() + bar.get_width()/2., -0.08,
                f'n={count}', ha='center', va='top', fontsize=9, style='italic')

    # 2. Theoretical limits table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    limits_text = (
        f"Theoretical Limits\n"
        f"{'='*35}\n\n"
        f"Char Acc -> Word Acc\n"
        f"{'-'*35}\n"
    )

    for ca in [0.90, 0.92, 0.95, 0.98, 0.99, 1.00]:
        wa = word_limits['theoretical_at_char_accuracy'][ca]
        limits_text += f"{ca:.0%}      ->  {wa:.2%}\n"

    limits_text += f"\n{'-'*35}\n"
    limits_text += f"Current Word Acc:\n{word_limits['current_word_accuracy']:.2%}\n\n"
    limits_text += f"To reach 95% word acc,\n"
    limits_text += f"need ~{0.95**(1/length_dist['mean']):.1%} char acc\n\n"
    limits_text += f"{'-'*35}\n"
    limits_text += f"Mean word length:\n{length_dist['mean']:.2f} characters"

    ax2.text(0.05, 0.95, limits_text, fontsize=11, family='monospace',
             verticalalignment='top', transform=ax2.transAxes)

    # Main title
    fig.suptitle('Word-Level Analysis',
                 fontsize=16, weight='bold', y=0.995)

    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\nWord-level visualization saved to: {output_file}")


def main(
    model_path: str = "char_recognizer.pth",
    image_dirs: List[str] = None,
    cache_dirs: List[str] = None,
    output_file: str = "word_level_analysis.png"
):
    """Main analysis function."""

    # Default to using only filtered/test
    if image_dirs is None:
        image_dirs = ["data/filtered/test"]
    if cache_dirs is None:
        cache_dirs = ["data/processed/filtered_test_cache"]

    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Word-Level Distribution and Limits Analysis")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Image directories: {image_dirs}")
    print("=" * 70)
    print()

    # Load model
    print("Loading model...")
    model = load_trained_model(model_path, device)

    # Load and combine datasets
    print("Loading datasets...")
    all_word_lengths = []
    all_word_correct = []
    all_word_char_accuracies = []

    for image_dir, cache_dir in zip(image_dirs, cache_dirs):
        print(f"\n  Processing: {image_dir}")
        dataset = CaptchaWordDataset(
            image_dir=Path(image_dir),
            cache_dir=Path(cache_dir)
        )
        print(f"  Loaded: {len(dataset)} words")

        # Evaluate at word level for this dataset
        print(f"  Evaluating word-level accuracy...")
        word_lengths, word_correct, word_char_accuracies = evaluate_word_level(
            model, dataset, device
        )
        all_word_lengths.extend(word_lengths)
        all_word_correct.extend(word_correct)
        all_word_char_accuracies.extend(word_char_accuracies)

    print(f"\nTotal words across all datasets: {len(all_word_lengths)}")
    print()

    # Analyze word length distribution
    print("Analyzing word length distribution...")
    length_dist = analyze_word_length_distribution(all_word_lengths)
    print()

    # Calculate theoretical limits
    print("Calculating theoretical limits...")
    word_limits = calculate_word_level_limits(all_word_lengths, all_word_correct)
    print()

    # Analyze by word length
    print("Analyzing accuracy by word length...")
    length_stats = analyze_by_word_length(
        all_word_lengths, all_word_correct, all_word_char_accuracies
    )
    print()

    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("Word Length Distribution:")
    print(f"  Mean: {length_dist['mean']:.2f} characters")
    print(f"  Std Dev: {length_dist['std']:.2f}")
    print(f"  Range: {length_dist['min']} - {length_dist['max']}")
    print(f"  Mode: {length_dist['mode']} characters")
    print()

    print("Distribution by length:")
    for length in sorted(length_dist['distribution'].keys()):
        count = length_dist['distribution'][length]
        pct = count / len(all_word_lengths) * 100
        print(f"  Length {length}: {count:4d} words ({pct:5.1f}%)")
    print()

    print("Current Performance:")
    print(f"  Word-Level Accuracy: {word_limits['current_word_accuracy']:.4f} ({word_limits['current_word_accuracy']:.2%})")
    print()

    print("Accuracy by Word Length:")
    for length in sorted(length_stats.keys()):
        stats = length_stats[length]
        print(f"  Length {length}: Word Acc = {stats['word_accuracy']:.2%}, "
              f"Char Acc = {stats['avg_char_accuracy']:.2%} "
              f"({stats['word_correct']}/{stats['word_total']})")
    print()

    print("Theoretical Word Accuracy at Different Character Accuracies:")
    for char_acc in [0.90, 0.92, 0.95, 0.98, 0.99, 1.00]:
        word_acc = word_limits['theoretical_at_char_accuracy'][char_acc]
        print(f"  Char Acc {char_acc:.0%} -> Word Acc {word_acc:.4f} ({word_acc:.2%})")
    print()

    # Calculate required char accuracy for target word accuracy
    target_word_acc = 0.95
    required_char_acc = target_word_acc ** (1 / length_dist['mean'])
    print(f"To achieve {target_word_acc:.0%} word accuracy:")
    print(f"  Required character accuracy: {required_char_acc:.4f} ({required_char_acc:.2%})")
    print()

    # Generate visualization
    print("Generating visualization...")
    plot_word_analysis(length_dist, length_stats, word_limits, output_file)
    print()

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze word-level distribution and theoretical limits"
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
        help="Image directories (default: data/filtered/test)"
    )
    parser.add_argument(
        "--cache-dirs",
        type=str,
        nargs='+',
        default=None,
        help="Cache directories (default: data/processed/filtered_test_cache)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="word_level_analysis.png",
        help="Output filename"
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        image_dirs=args.image_dirs,
        cache_dirs=args.cache_dirs,
        output_file=args.output
    )

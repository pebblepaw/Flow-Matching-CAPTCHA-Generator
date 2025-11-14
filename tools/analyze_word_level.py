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
    """Create comprehensive visualization of word-level analysis."""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Word length distribution
    ax1 = fig.add_subplot(gs[0, :2])
    lengths = sorted(length_dist['distribution'].keys())
    counts = [length_dist['distribution'][l] for l in lengths]
    ax1.bar(lengths, counts, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Word Length (characters)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Word Length Distribution', fontsize=14, weight='bold')
    ax1.axvline(x=length_dist['mean'], color='r', linestyle='--',
                label=f"Mean: {length_dist['mean']:.1f}", linewidth=2)
    ax1.axvline(x=length_dist['mode'], color='g', linestyle='--',
                label=f"Mode: {length_dist['mode']}", linewidth=2)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(lengths)

    # Add count labels on bars
    for i, (length, count) in enumerate(zip(lengths, counts)):
        ax1.text(length, count + max(counts)*0.02, str(count),
                ha='center', va='bottom', fontsize=10, weight='bold')

    # 2. Length statistics (text box)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = (
        f"Word Length Statistics\n"
        f"{'='*35}\n"
        f"Total Words: {len(word_limits):,}\n"
        f"Mean Length: {length_dist['mean']:.2f}\n"
        f"Std Dev: {length_dist['std']:.2f}\n"
        f"Min Length: {length_dist['min']}\n"
        f"Max Length: {length_dist['max']}\n"
        f"Median: {length_dist['median']:.1f}\n"
        f"Mode: {length_dist['mode']}\n\n"
        f"Current Word Accuracy:\n"
        f"{'='*35}\n"
        f"{word_limits['current_word_accuracy']:.2%}\n\n"
        f"(Word correct only if\n"
        f"ALL characters correct)"
    )
    ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    # 3. Word vs Character accuracy by length
    ax3 = fig.add_subplot(gs[1, :2])
    lengths_sorted = sorted(length_stats.keys())
    word_accs = [length_stats[l]['word_accuracy'] for l in lengths_sorted]
    char_accs = [length_stats[l]['avg_char_accuracy'] for l in lengths_sorted]

    x = np.arange(len(lengths_sorted))
    width = 0.35

    bars1 = ax3.bar(x - width/2, word_accs, width, label='Word Accuracy',
                    color='#d62728', alpha=0.8)
    bars2 = ax3.bar(x + width/2, char_accs, width, label='Char Accuracy',
                    color='#2ca02c', alpha=0.8)

    ax3.set_xlabel('Word Length', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Word vs Character Accuracy by Length', fontsize=14, weight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(lengths_sorted)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=9)

    # 4. Sample counts by length
    ax4 = fig.add_subplot(gs[1, 2])
    sample_counts = [length_stats[l]['word_total'] for l in lengths_sorted]
    correct_counts = [length_stats[l]['word_correct'] for l in lengths_sorted]

    ax4.barh(range(len(lengths_sorted)), sample_counts,
             color='#1f77b4', alpha=0.3, label='Total')
    ax4.barh(range(len(lengths_sorted)), correct_counts,
             color='#2ca02c', alpha=0.8, label='Correct')
    ax4.set_yticks(range(len(lengths_sorted)))
    ax4.set_yticklabels([f'Length {l}' for l in lengths_sorted])
    ax4.set_xlabel('Number of Words', fontsize=12)
    ax4.set_title('Sample Distribution', fontsize=14, weight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(axis='x', alpha=0.3)

    # Add count labels
    for i, (total, correct) in enumerate(zip(sample_counts, correct_counts)):
        ax4.text(total + max(sample_counts)*0.02, i,
                f'{correct}/{total}', va='center', fontsize=9)

    # 5. Theoretical limits curve
    ax5 = fig.add_subplot(gs[2, :2])
    char_accs = sorted(word_limits['theoretical_at_char_accuracy'].keys())
    word_accs_theoretical = [word_limits['theoretical_at_char_accuracy'][ca] for ca in char_accs]

    ax5.plot(char_accs, word_accs_theoretical, 'b-o', linewidth=2, markersize=8,
            label='Theoretical Word Accuracy')
    ax5.axhline(y=word_limits['current_word_accuracy'], color='r', linestyle='--',
                label=f"Current: {word_limits['current_word_accuracy']:.2%}", linewidth=2)
    ax5.set_xlabel('Character-Level Accuracy', fontsize=12)
    ax5.set_ylabel('Word-Level Accuracy', fontsize=12)
    ax5.set_title('Theoretical Word Accuracy vs Character Accuracy',
                 fontsize=14, weight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(alpha=0.3)
    ax5.set_xlim(0.88, 1.02)

    # Add value labels on curve
    for ca, wa in zip(char_accs, word_accs_theoretical):
        ax5.text(ca, wa + 0.02, f'{wa:.1%}', ha='center', fontsize=9)

    # 6. Theoretical limits table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

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
    limits_text += f"need ~{0.95**(1/length_dist['mean']):.1%} char acc"

    ax6.text(0.05, 0.95, limits_text, fontsize=11, family='monospace',
             verticalalignment='top', transform=ax6.transAxes)

    # Main title
    fig.suptitle('Word-Level Distribution and Theoretical Limits Analysis',
                 fontsize=16, weight='bold', y=0.995)

    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\nWord-level visualization saved to: {output_file}")


def main(
    model_path: str = "char_recognizer.pth",
    image_dir: str = "data/raw/test",
    cache_dir: str = "data/processed/test_cache",
    output_file: str = "word_level_analysis.png"
):
    """Main analysis function."""

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
    print("=" * 70)
    print()

    # Load model
    print("Loading model...")
    model = load_trained_model(model_path, device)

    # Load dataset
    print("Loading dataset...")
    dataset = CaptchaWordDataset(
        image_dir=Path(image_dir),
        cache_dir=Path(cache_dir)
    )
    print(f"Dataset: {len(dataset)} words")
    print()

    # Evaluate at word level
    print("Evaluating word-level accuracy...")
    word_lengths, word_correct, word_char_accuracies = evaluate_word_level(
        model, dataset, device
    )
    print(f"Total words evaluated: {len(word_lengths)}")
    print()

    # Analyze word length distribution
    print("Analyzing word length distribution...")
    length_dist = analyze_word_length_distribution(word_lengths)
    print()

    # Calculate theoretical limits
    print("Calculating theoretical limits...")
    word_limits = calculate_word_level_limits(word_lengths, word_correct)
    print()

    # Analyze by word length
    print("Analyzing accuracy by word length...")
    length_stats = analyze_by_word_length(
        word_lengths, word_correct, word_char_accuracies
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
        pct = count / len(word_lengths) * 100
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
        "--image-dir",
        type=str,
        default="data/raw/test",
        help="Test images directory"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/processed/test_cache",
        help="Cache directory"
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
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        output_file=args.output
    )

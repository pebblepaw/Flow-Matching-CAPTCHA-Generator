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
    char_stats: Dict,
    theoretical_limits: Dict,
    output_file: str = "class_balance_analysis.png",
    dpi: int = 120
):
    """Create comprehensive visualization of class balance analysis."""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Class distribution (bar chart)
    ax1 = fig.add_subplot(gs[0, :2])
    chars = sorted(balance_metrics['distribution'].keys())
    counts = [balance_metrics['distribution'][c] for c in chars]
    colors = ['#1f77b4' if c.isdigit() else '#ff7f0e' for c in chars]
    ax1.bar(chars, counts, color=colors)
    ax1.set_xlabel('Character', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Class Distribution (Blue=Digits, Orange=Letters)', fontsize=14, weight='bold')
    ax1.axhline(y=balance_metrics['mean_count'], color='r', linestyle='--',
                label=f"Mean: {balance_metrics['mean_count']:.1f}")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Balance metrics (text box)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = (
        f"Class Balance Metrics\n"
        f"{'='*30}\n"
        f"Total Characters: {balance_metrics['total_chars']:,}\n"
        f"Number of Classes: {balance_metrics['num_classes']}\n"
        f"Mean Count: {balance_metrics['mean_count']:.1f}\n"
        f"Std Dev: {balance_metrics['std_count']:.1f}\n"
        f"Min Count: {balance_metrics['min_count']}\n"
        f"Max Count: {balance_metrics['max_count']}\n"
        f"Imbalance Ratio: {balance_metrics['imbalance_ratio']:.2f}x\n"
        f"Coeff. of Variation: {balance_metrics['cv']:.2%}\n\n"
        f"Interpretation:\n"
        f"{'='*30}\n"
        f"{'Balanced' if balance_metrics['imbalance_ratio'] < 1.5 else 'Imbalanced'}\n"
        f"({balance_metrics['imbalance_ratio']:.2f}x ratio)"
    )
    ax2.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center')

    # 3. Per-character accuracy
    ax3 = fig.add_subplot(gs[1, :2])
    chars_by_acc = sorted(char_stats.items(), key=lambda x: x[1]['accuracy'])
    chars = [c for c, _ in chars_by_acc]
    accs = [s['accuracy'] for _, s in chars_by_acc]
    colors = ['#d62728' if acc < 0.8 else '#2ca02c' if acc > 0.95 else '#ff7f0e'
              for acc in accs]
    ax3.barh(chars, accs, color=colors)
    ax3.set_xlabel('Accuracy', fontsize=12)
    ax3.set_ylabel('Character', fontsize=12)
    ax3.set_title('Per-Character Accuracy (Red<80%, Orange=80-95%, Green>95%)',
                  fontsize=14, weight='bold')
    ax3.axvline(x=theoretical_limits['avg_char_accuracy'], color='b', linestyle='--',
                label=f"Mean: {theoretical_limits['avg_char_accuracy']:.2%}")
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)

    # 4. Theoretical limits (bar chart)
    ax4 = fig.add_subplot(gs[1, 2])
    scenarios = [
        'Current',
        'Fix Bottom 25%',
        'Fix Bottom 50%',
        'All to Best',
        'Perfect'
    ]
    values = [
        theoretical_limits['current_accuracy'],
        theoretical_limits['fixed_bottom_25'],
        theoretical_limits['fixed_bottom_50'],
        theoretical_limits['all_best_accuracy'],
        theoretical_limits['perfect_accuracy']
    ]
    colors_scenarios = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax4.barh(scenarios, values, color=colors_scenarios)
    ax4.set_xlabel('Accuracy', fontsize=12)
    ax4.set_title('Theoretical Accuracy Limits', fontsize=14, weight='bold')
    ax4.set_xlim(0.8, 1.0)
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax4.text(val + 0.005, i, f'{val:.2%}', va='center', fontsize=10)
    ax4.grid(axis='x', alpha=0.3)

    # 5. Improvement impact (top 10)
    ax5 = fig.add_subplot(gs[2, :2])
    impacts = calculate_improvement_impact(char_stats)[:10]
    chars_impact = [c for c, _, _, _ in impacts]
    impact_vals = [imp * 100 for _, _, imp, _ in impacts]  # Convert to percentage points
    ax5.barh(chars_impact, impact_vals, color='#17becf')
    ax5.set_xlabel('Impact on Overall Accuracy (% points)', fontsize=12)
    ax5.set_ylabel('Character', fontsize=12)
    ax5.set_title('Top 10: Impact of 1% Improvement per Character',
                  fontsize=14, weight='bold')
    ax5.grid(axis='x', alpha=0.3)
    for i, (char, curr_acc, imp, count) in enumerate(impacts[:10]):
        ax5.text(impact_vals[i] + 0.001, i,
                f'{curr_acc:.1%} ({count})',
                va='center', fontsize=9)

    # 6. Summary and recommendations (text box)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    # Find worst performing high-frequency chars
    high_freq_chars = sorted(
        [(c, s) for c, s in char_stats.items()],
        key=lambda x: x[1]['total'],
        reverse=True
    )[:10]
    worst_of_common = sorted(
        high_freq_chars,
        key=lambda x: x[1]['accuracy']
    )[:3]

    summary_text = (
        f"Key Insights\n"
        f"{'='*30}\n\n"
        f"Current: {theoretical_limits['current_accuracy']:.2%}\n"
        f"Gap to Perfect: {theoretical_limits['gap_to_perfect']:.2%}\n\n"
        f"Quick Wins:\n"
        f"{'='*30}\n"
        f"Fix bottom 25% chars:\n"
        f"  → {theoretical_limits['fixed_bottom_25']:.2%}\n"
        f"  (Gain: {(theoretical_limits['fixed_bottom_25']-theoretical_limits['current_accuracy']):.2%})\n\n"
        f"Priority Targets:\n"
        f"{'='*30}\n"
    )

    for i, (char, stats) in enumerate(worst_of_common, 1):
        summary_text += f"{i}. '{char}': {stats['accuracy']:.1%}\n"
        summary_text += f"   ({stats['total']} samples)\n"

    ax6.text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax6.transAxes)

    # Main title
    fig.suptitle('Class Balance and Theoretical Optimum Analysis',
                 fontsize=16, weight='bold', y=0.995)

    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\nAnalysis visualization saved to: {output_file}")


def main(
    model_path: str = "char_recognizer.pth",
    image_dir: str = "data/raw/test",
    cache_dir: str = "data/processed/test_cache",
    output_file: str = "class_balance_analysis.png"
):
    """Main analysis function."""

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
    print(f"Dataset: {len(dataset)} samples")
    print()

    # Collect predictions
    print("Collecting predictions...")
    all_gt_chars, all_pred_chars = collect_predictions(model, dataset, device)
    print(f"Total characters: {len(all_gt_chars)}")
    print()

    # Analyze class balance
    print("Analyzing class balance...")
    balance_metrics = analyze_class_balance(all_gt_chars)
    print()

    # Calculate per-character accuracy
    print("Calculating per-character accuracy...")
    char_stats = calculate_per_char_accuracy(all_gt_chars, all_pred_chars)
    print()

    # Calculate theoretical limits
    print("Calculating theoretical limits...")
    theoretical_limits = calculate_theoretical_limits(char_stats)
    print()

    # Print detailed results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("Class Balance Metrics:")
    print(f"  Imbalance Ratio: {balance_metrics['imbalance_ratio']:.2f}x")
    print(f"  Most Common: {max(balance_metrics['distribution'].items(), key=lambda x: x[1])}")
    print(f"  Least Common: {min(balance_metrics['distribution'].items(), key=lambda x: x[1])}")
    print()

    print("Theoretical Accuracy Limits:")
    print(f"  Current Accuracy: {theoretical_limits['current_accuracy']:.4f} ({theoretical_limits['current_accuracy']:.2%})")
    print(f"  Fix Bottom 25%:   {theoretical_limits['fixed_bottom_25']:.4f} ({theoretical_limits['fixed_bottom_25']:.2%})")
    print(f"  Fix Bottom 50%:   {theoretical_limits['fixed_bottom_50']:.4f} ({theoretical_limits['fixed_bottom_50']:.2%})")
    print(f"  All to Best:      {theoretical_limits['all_best_accuracy']:.4f} ({theoretical_limits['all_best_accuracy']:.2%})")
    print(f"  Perfect (100%):   {theoretical_limits['perfect_accuracy']:.4f} ({theoretical_limits['perfect_accuracy']:.2%})")
    print()

    print(f"  Gap to Perfect:   {theoretical_limits['gap_to_perfect']:.4f} ({theoretical_limits['gap_to_perfect']:.2%})")
    print()

    print("Top 5 Impact Characters (1% improvement):")
    impacts = calculate_improvement_impact(char_stats)
    for i, (char, curr_acc, impact, count) in enumerate(impacts[:5], 1):
        print(f"  {i}. '{char}': +{impact:.4f} (+{impact*100:.2f}% points) "
              f"[Current: {curr_acc:.2%}, Freq: {count}]")
    print()

    # Generate visualization
    print("Generating visualization...")
    plot_analysis(balance_metrics, char_stats, theoretical_limits, output_file)
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
        default="class_balance_analysis.png",
        help="Output filename"
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        output_file=args.output
    )

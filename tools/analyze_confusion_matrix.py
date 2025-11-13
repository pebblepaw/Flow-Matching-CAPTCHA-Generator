"""
Generate confusion matrix for character recognition errors.

This script analyzes the tokenization results to identify which character pairs
are commonly confused, helping identify limitations in the CAPTCHA solving pipeline.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.hairline_removal import (
    color_voting_remove_black_hairline,
    inpainting_remove_black_hairline
)
from src.tokenization.color_region_tokenizer import ColorRegionTokenizer


# Configuration
TRAIN_DIR = 'data/raw/train'
PREPROCESSING_METHOD = 'combined'
INPAINT_RADIUS = 2
BLACK_THRESHOLD = 50
INPAINT_THRESHOLD = 5
REMOVE_SECONDARY_COLORS = True
HUE_THRESHOLD = 1
VALUE_THRESHOLD = 50

# Valid CAPTCHA characters (36 characters: 0-9, a-z)
VALID_CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'


def extract_ground_truth(filename: str) -> str:
    """Extract ground truth label from filename (format: label-number.png)."""
    return filename.split('-')[0]


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply hairline removal preprocessing."""
    color_voted = color_voting_remove_black_hairline(image, black_threshold=BLACK_THRESHOLD)
    preprocessed = inpainting_remove_black_hairline(
        color_voted,
        black_threshold=INPAINT_THRESHOLD,
        inpaint_radius=INPAINT_RADIUS
    )
    return preprocessed


def analyze_tokenization_errors(train_dir: str):
    """
    Analyze all training images and build confusion matrix.

    Returns:
        confusion_matrix: 36x36 matrix where [i, j] = count of times char i was tokenized as j tokens
        char_to_idx: Mapping from character to matrix index
        error_cases: List of (image_name, expected_chars, num_tokens) for error cases
    """
    train_path = Path(train_dir)
    if not train_path.exists():
        print(f"Error: {train_dir} not found")
        sys.exit(1)

    tokenizer = ColorRegionTokenizer()

    # Initialize confusion tracking
    # We'll track: confusion[expected_char] = {num_tokens: count}
    confusion = defaultdict(lambda: defaultdict(int))
    error_cases = []

    total_images = 0
    total_chars = 0
    total_tokens = 0
    correct_images = 0

    # Process all images
    image_files = sorted(train_path.glob('*.png'))

    print(f"Analyzing {len(image_files)} images...")
    print("=" * 80)

    for img_path in image_files:
        total_images += 1
        ground_truth = extract_ground_truth(img_path.name)

        # Load and preprocess
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(img_rgb)

        # Tokenize
        tokens = tokenizer.tokenize(preprocessed)

        if REMOVE_SECONDARY_COLORS:
            tokens = [
                tokenizer.remove_secondary_colors(
                    token,
                    hue_threshold=HUE_THRESHOLD,
                    value_threshold=VALUE_THRESHOLD
                )
                for token in tokens
            ]

        num_tokens = len(tokens)
        expected_num = len(ground_truth)

        total_chars += expected_num
        total_tokens += num_tokens

        # Track confusion for each character
        if num_tokens == expected_num:
            correct_images += 1
            # Correct tokenization - each char maps to 1 token
            for char in ground_truth:
                confusion[char][1] += 1
        else:
            # Error case
            error_cases.append((img_path.name, ground_truth, num_tokens))

            # Estimate which character caused the error
            # Strategy: if we have more/fewer tokens, attribute error to all chars
            token_ratio = num_tokens / expected_num if expected_num > 0 else 0

            for char in ground_truth:
                confusion[char][num_tokens] += 1

    print(f"\nProcessed {total_images} images")
    print(f"  Total characters: {total_chars}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Correct images: {correct_images}/{total_images} ({100*correct_images/total_images:.1f}%)")
    print(f"  Error cases: {len(error_cases)}")

    return confusion, error_cases


def build_confusion_matrix(confusion: dict):
    """
    Build a 36x36 confusion matrix from confusion data.

    Matrix[i, j] represents:
    - j=0: Character was merged (under-tokenized, produced 0 tokens)
    - j=1: Character was correctly tokenized (1 token)
    - j=2+: Character was split (over-tokenized, produced multiple tokens)

    We'll simplify to: [correct, under-tokenized, over-tokenized]
    """
    # Create mapping
    char_to_idx = {char: idx for idx, char in enumerate(VALID_CHARS)}

    # Initialize 36x3 matrix: [correct, merged, split]
    matrix = np.zeros((36, 3), dtype=int)

    for char, token_counts in confusion.items():
        if char not in char_to_idx:
            continue

        char_idx = char_to_idx[char]

        for num_tokens, count in token_counts.items():
            if num_tokens == 1:
                matrix[char_idx, 0] += count  # Correct
            elif num_tokens == 0:
                matrix[char_idx, 1] += count  # Merged (under-tokenized)
            else:  # num_tokens >= 2
                matrix[char_idx, 2] += count  # Split (over-tokenized)

    return matrix, char_to_idx


def plot_confusion_matrix(matrix: np.ndarray, char_to_idx: dict, output_file: str):
    """Plot and save the confusion matrix."""
    # Create character labels
    chars = [char for char, _ in sorted(char_to_idx.items(), key=lambda x: x[1])]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 14))

    # Plot heatmap
    sns.heatmap(
        matrix,
        xticklabels=['Correct (1 token)', 'Under-tokenized\n(merged)', 'Over-tokenized\n(split)'],
        yticklabels=chars,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_xlabel('Tokenization Result', fontweight='bold', fontsize=11)
    ax.set_ylabel('Ground Truth Character', fontweight='bold', fontsize=11)
    ax.set_title('Character-Level Tokenization Confusion Matrix', fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix: {output_file}")

    plt.show()


def analyze_error_patterns(matrix: np.ndarray, char_to_idx: dict, error_cases: list):
    """Analyze and print error patterns."""
    chars = [char for char, _ in sorted(char_to_idx.items(), key=lambda x: x[1])]

    print("\n" + "=" * 80)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 80)

    # Characters most likely to be merged (under-tokenized)
    merged_counts = matrix[:, 1]
    merged_chars = [(chars[i], merged_counts[i]) for i in range(len(chars)) if merged_counts[i] > 0]
    merged_chars.sort(key=lambda x: x[1], reverse=True)

    print("\n1. Characters Most Likely to be MERGED (under-tokenized):")
    print("-" * 60)
    if merged_chars:
        for char, count in merged_chars[:10]:
            total = np.sum(matrix[char_to_idx[char], :])
            pct = 100 * count / total if total > 0 else 0
            print(f"   '{char}': {count:3d} times ({pct:5.1f}% of occurrences)")
    else:
        print("   None!")

    # Characters most likely to be split (over-tokenized)
    split_counts = matrix[:, 2]
    split_chars = [(chars[i], split_counts[i]) for i in range(len(chars)) if split_counts[i] > 0]
    split_chars.sort(key=lambda x: x[1], reverse=True)

    print("\n2. Characters Most Likely to be SPLIT (over-tokenized):")
    print("-" * 60)
    if split_chars:
        for char, count in split_chars[:10]:
            total = np.sum(matrix[char_to_idx[char], :])
            pct = 100 * count / total if total > 0 else 0
            print(f"   '{char}': {count:3d} times ({pct:5.1f}% of occurrences)")
    else:
        print("   None!")

    # Overall statistics
    total_chars = np.sum(matrix)
    correct = np.sum(matrix[:, 0])
    merged = np.sum(matrix[:, 1])
    split = np.sum(matrix[:, 2])

    print("\n3. Overall Statistics:")
    print("-" * 60)
    print(f"   Total character instances: {total_chars}")
    print(f"   Correctly tokenized: {correct} ({100*correct/total_chars:.1f}%)")
    print(f"   Under-tokenized (merged): {merged} ({100*merged/total_chars:.1f}%)")
    print(f"   Over-tokenized (split): {split} ({100*split/total_chars:.1f}%)")

    # Print some error cases
    print("\n4. Sample Error Cases:")
    print("-" * 60)
    for i, (img_name, ground_truth, num_tokens) in enumerate(error_cases[:15]):
        expected = len(ground_truth)
        error_type = "OVER" if num_tokens > expected else "UNDER"
        print(f"   {img_name:20s} | Expected: {expected} ({ground_truth}) | Got: {num_tokens} | {error_type}")

    if len(error_cases) > 15:
        print(f"   ... and {len(error_cases) - 15} more error cases")


def main():
    print("Character-Level Confusion Matrix Analysis")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Training directory: {TRAIN_DIR}")
    print(f"  Preprocessing: {PREPROCESSING_METHOD}")
    print(f"  Inpaint radius: {INPAINT_RADIUS}")
    print(f"  Inpaint threshold: {INPAINT_THRESHOLD}")
    print(f"  Remove secondary colors: {REMOVE_SECONDARY_COLORS}")
    print("=" * 80 + "\n")

    # Analyze tokenization
    confusion, error_cases = analyze_tokenization_errors(TRAIN_DIR)

    # Build confusion matrix
    matrix, char_to_idx = build_confusion_matrix(confusion)

    # Plot
    plot_confusion_matrix(matrix, char_to_idx, 'confusion_matrix.png')

    # Analyze patterns
    analyze_error_patterns(matrix, char_to_idx, error_cases)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

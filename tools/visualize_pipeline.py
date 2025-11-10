"""
Comprehensive visualization script for CAPTCHA preprocessing and tokenization pipeline.

This script provides the most verbose and detailed visualization of the entire pipeline:
1. Original image
2. Preprocessed image (hairline removal)
3. Foreground mask
4. Individual tokens (up to 6 displayed)
5. Combined token canvas with bounding boxes

Features:
- Red borders and red titles for mismatched token counts
- Detailed statistics for each image
- Multiple preprocessing method support
- Both individual and combined token displays
- Comprehensive console output
"""

import random
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.hairline_removal import (
    color_voting_remove_black_hairline,
    inpainting_remove_black_hairline
)
from src.tokenization.color_region_tokenizer import ColorRegionTokenizer


# Configuration
IMAGE_COUNT = 12
PREPROCESSING_METHOD = 'combined'  # 'color_voting', 'inpainting', or 'combined'
INPAINT_RADIUS = 2  # Smaller = weaker inpainting (1-3 recommended, default: 2)
BLACK_THRESHOLD = 50  # Pixels darker than this are considered black (for color voting)
INPAINT_THRESHOLD = 5  # Threshold for inpainting - only pixels darker than this get inpainted
TRAIN_DIR = 'data/raw/train'
OUTPUT_FILE = 'pipeline_visualizatio.png'
RANDOM_SEED = 142
DPI = 120
MAX_INDIVIDUAL_TOKENS = 6
SHOW_COMBINED_TOKENS = True


def get_random_images(train_dir: str, count: int, seed: int = 42) -> List[Path]:
    """
    Select random images from training directory.

    Args:
        train_dir: Path to training directory
        count: Number of images to select
        seed: Random seed for reproducibility

    Returns:
        List of selected image paths
    """
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    all_images = list(train_path.glob("*.png"))
    if len(all_images) == 0:
        raise ValueError(f"No PNG images found in {train_dir}")

    random.seed(seed)
    count = min(count, len(all_images))
    selected = random.sample(all_images, count)

    print(f"Selected {len(selected)} images from {len(all_images)} total images")
    return sorted(selected)


def preprocess_image(
    image: np.ndarray,
    method: str = "color_voting",
    inpaint_radius: int = 2,
    black_threshold: int = 50,
    inpaint_threshold: int = None
) -> np.ndarray:
    """
    Apply hairline removal preprocessing.

    Args:
        image: Input RGB image
        method: Preprocessing method ('color_voting', 'inpainting', or 'combined')
        inpaint_radius: Radius for inpainting (smaller = weaker effect)
        black_threshold: Pixels darker than this are considered black
        inpaint_threshold: Threshold for inpainting (only for 'inpainting' and 'combined' methods)

    Returns:
        Preprocessed RGB image
    """
    if method == "color_voting":
        return color_voting_remove_black_hairline(image, black_threshold=black_threshold)
    elif method == "inpainting":
        thresh = inpaint_threshold if inpaint_threshold is not None else black_threshold
        return inpainting_remove_black_hairline(image, black_threshold=thresh, inpaint_radius=inpaint_radius)
    elif method == "combined":
        # Color voting first, then inpainting with custom threshold
        result = color_voting_remove_black_hairline(image, black_threshold=black_threshold)
        thresh = inpaint_threshold if inpaint_threshold is not None else black_threshold
        return inpainting_remove_black_hairline(result, black_threshold=thresh, inpaint_radius=inpaint_radius)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def create_combined_token_canvas(tokens: List[np.ndarray]) -> np.ndarray:
    """
    Create a canvas with all tokens side-by-side with bounding boxes.

    Args:
        tokens: List of token images

    Returns:
        Combined canvas image with boxes around each token
    """
    if len(tokens) == 0:
        return np.ones((50, 200, 3), dtype=np.uint8) * 255

    max_height = max(token.shape[0] for token in tokens)
    spacing = 10
    total_width = sum(token.shape[1] for token in tokens) + spacing * (len(tokens) - 1)
    canvas = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

    x_offset = 0
    for token in tokens:
        h, w = token.shape[:2]
        y_offset = (max_height - h) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = token

        # Draw bounding box
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset+w-1, y_offset+h-1),
                     (0, 0, 0), 2)

        x_offset += w + spacing

    return canvas


def visualize_pipeline(
    images: List[Path],
    tokenizer: ColorRegionTokenizer,
    preprocessing_method: str,
    output_file: str,
    dpi: int = 120,
    inpaint_radius: int = 2,
    black_threshold: int = 50,
    inpaint_threshold: int = None
):
    """
    Visualize the complete pipeline for multiple images with maximum verbosity.

    Each row shows:
    - Original image
    - Color voted (if combined method)
    - Final preprocessed
    - Foreground mask
    - Individual tokens (up to MAX_INDIVIDUAL_TOKENS)
    - Combined token canvas (optional)

    Args:
        images: List of image paths
        tokenizer: ColorRegionTokenizer instance
        preprocessing_method: Name of preprocessing method
        output_file: Output file path
        dpi: DPI for saved figure
        inpaint_radius: Radius for inpainting
        black_threshold: Threshold for color voting
        inpaint_threshold: Threshold for inpainting (only for combined/inpainting methods)
    """
    num_images = len(images)

    # Calculate columns
    # For combined method: original + color_voted + final + mask + tokens + combined
    # For other methods: original + preprocessed + mask + tokens + combined
    show_intermediate = preprocessing_method == 'combined'
    base_cols = 4 if show_intermediate else 3
    token_cols = MAX_INDIVIDUAL_TOKENS
    combined_col = 1 if SHOW_COMBINED_TOKENS else 0
    num_cols = base_cols + token_cols + combined_col

    # Calculate figure dimensions
    fig_width = num_cols * 2.2
    fig_height = num_images * 2.0

    fig, axes = plt.subplots(
        num_images,
        num_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={'wspace': 0.15, 'hspace': 0.4}
    )

    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)

    print("\n" + "=" * 80)
    print("PROCESSING IMAGES")
    print("=" * 80)

    total_correct = 0
    total_tokens_expected = 0
    total_tokens_extracted = 0

    for idx, img_path in enumerate(images):
        print(f"\n[{idx+1}/{num_images}] Processing: {img_path.name}")
        print("-" * 80)

        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  ⚠️  ERROR: Failed to load image!")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        label = img_path.stem.split('-')[0]

        print(f"  Ground Truth Label: '{label}' ({len(label)} characters)")
        print(f"  Image Shape: {img_rgb.shape[1]}x{img_rgb.shape[0]} (WxH)")

        # Step 1: Preprocess
        if preprocessing_method == 'combined':
            # Combined method: color voting -> inpaint with custom threshold
            inpaint_thresh = inpaint_threshold if inpaint_threshold is not None else black_threshold

            color_voted = color_voting_remove_black_hairline(img_rgb, black_threshold=black_threshold)
            preprocessed = inpainting_remove_black_hairline(color_voted, black_threshold=inpaint_thresh, inpaint_radius=inpaint_radius)
            print(f"  [OK] Preprocessing (combined): Color voting + Inpainting(thresh={inpaint_thresh}) complete")
        else:
            color_voted = None
            preprocessed = preprocess_image(
                img_rgb,
                method=preprocessing_method,
                inpaint_radius=inpaint_radius,
                black_threshold=black_threshold,
                inpaint_threshold=inpaint_threshold
            )
            print(f"  [OK] Preprocessing ({preprocessing_method}): Complete")

        # Step 2: Tokenize with mask
        tokens, fg_mask = tokenizer.tokenize(
            preprocessed,
            return_mask=True
        )

        num_tokens = len(tokens)
        token_match = num_tokens == len(label)
        total_tokens_expected += len(label)
        total_tokens_extracted += num_tokens
        if token_match:
            total_correct += 1

        fg_pixels = np.sum(fg_mask)
        fg_percentage = (fg_pixels / (fg_mask.shape[0] * fg_mask.shape[1])) * 100

        print(f"  [OK] Tokenization: Extracted {num_tokens} tokens")
        print(f"    Expected: {len(label)} | Got: {num_tokens} | Match: {token_match}")
        print(f"    Foreground pixels: {fg_pixels:,} ({fg_percentage:.1f}%)")
        print(f"    Token shapes: {[f'{t.shape[1]}x{t.shape[0]}' for t in tokens]}")

        # Determine title color
        title_color = 'green' if token_match else 'red'
        border_color = 'green' if token_match else 'red'
        border_width = 3

        col_idx = 0

        # Column 0: Display original
        ax = axes[idx, col_idx]
        ax.imshow(img_rgb)
        ax.set_title(f'Original\n"{label}"', fontsize=9, color=title_color, weight='bold')
        ax.axis('off')
        if not token_match:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
                spine.set_visible(True)
        col_idx += 1

        # Column 1: Display color voted (for combined/brighten methods)
        if show_intermediate:
            ax = axes[idx, col_idx]
            ax.imshow(color_voted)
            ax.set_title(f'Color Voting\n(step 1)', fontsize=9, color=title_color, weight='bold')
            ax.axis('off')
            if not token_match:
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(border_width)
                    spine.set_visible(True)
            col_idx += 1

        # Column: Display final preprocessed
        ax = axes[idx, col_idx]
        ax.imshow(preprocessed)
        if show_intermediate:
            ax.set_title(f'+ Inpainting\n(step 2)', fontsize=9, color=title_color, weight='bold')
        else:
            ax.set_title(f'Preprocessed\n({preprocessing_method})', fontsize=9, color=title_color, weight='bold')
        ax.axis('off')
        if not token_match:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
                spine.set_visible(True)
        col_idx += 1

        # Column: Display foreground mask
        ax = axes[idx, col_idx]
        ax.imshow(fg_mask, cmap='gray')
        ax.set_title(f'Foreground Mask\n{fg_pixels:,} pixels', fontsize=9, color=title_color, weight='bold')
        ax.axis('off')
        if not token_match:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
                spine.set_visible(True)
        col_idx += 1

        # Columns: Display individual tokens
        for t_idx in range(MAX_INDIVIDUAL_TOKENS):
            ax = axes[idx, col_idx + t_idx]
            if t_idx < num_tokens:
                token = tokens[t_idx]
                ax.imshow(token)
                ax.set_title(f'Token {t_idx+1}\n{token.shape[1]}x{token.shape[0]}',
                           fontsize=8, color=title_color)
            else:
                ax.text(0.5, 0.5, 'N/A',
                       ha='center', va='center',
                       fontsize=9, color='gray')
                ax.set_title(f'Token {t_idx+1}\n—', fontsize=8, color='gray')
            ax.axis('off')
            if not token_match and t_idx < num_tokens:
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(border_width)
                    spine.set_visible(True)

        # Last column: Combined token canvas
        if SHOW_COMBINED_TOKENS:
            ax = axes[idx, col_idx + token_cols]
            combined = create_combined_token_canvas(tokens)
            ax.imshow(combined)
            ax.set_title(f'All Tokens\n{num_tokens}/{len(label)}', fontsize=9, color=title_color, weight='bold')
            ax.axis('off')
            if not token_match:
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(border_width)
                    spine.set_visible(True)

        # Add row status indicator
        status_symbol = "[OK]" if token_match else "[X]"
        status_text = f'{status_symbol} {idx+1}. Expected: {len(label)} | Got: {num_tokens}'
        fig.text(
            0.005,
            1 - (idx + 0.5) / num_images,
            status_text,
            va='center',
            fontsize=9,
            color=title_color,
            weight='bold'
        )

    # Add main title
    accuracy = (total_correct / num_images * 100) if num_images > 0 else 0
    fig.suptitle(
        f'CAPTCHA Processing Pipeline - Verbose Visualization\n'
        f'Method: {preprocessing_method} | Accuracy: {total_correct}/{num_images} ({accuracy:.1f}%)',
        fontsize=14,
        weight='bold',
        y=0.998
    )

    # Add footer with statistics
    footer_text = (
        f'Statistics: Total Expected Tokens: {total_tokens_expected} | '
        f'Total Extracted: {total_tokens_extracted} | '
        f'Green = Correct | Red = Incorrect'
    )
    fig.text(
        0.5, 0.002,
        footer_text,
        ha='center',
        fontsize=9,
        style='italic',
        weight='bold'
    )

    plt.tight_layout(rect=[0.015, 0.008, 1, 0.992])
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total Images Processed: {num_images}")
    print(f"Correct Token Counts: {total_correct}/{num_images} ({accuracy:.1f}%)")
    print(f"Total Expected Tokens: {total_tokens_expected}")
    print(f"Total Extracted Tokens: {total_tokens_extracted}")
    print(f"Token Extraction Rate: {(total_tokens_extracted/total_tokens_expected*100):.1f}%")
    print(f"\nVisualization saved to: {output_file}")
    print("=" * 80)

    plt.close()


def main():
    """Main execution function with comprehensive output."""
    print("\n" + "=" * 80)
    print("CAPTCHA PIPELINE VISUALIZATION - VERBOSE MODE")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Images to process: {IMAGE_COUNT}")
    print(f"  Training directory: {TRAIN_DIR}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Preprocessing method: {PREPROCESSING_METHOD}")
    if PREPROCESSING_METHOD in ['inpainting', 'combined']:
        print(f"  Inpaint radius: {INPAINT_RADIUS} (smaller = weaker)")
        print(f"  Inpaint threshold: {INPAINT_THRESHOLD} (pixels darker than this get inpainted)")
    print(f"  Black threshold: {BLACK_THRESHOLD}")
    print(f"  Max individual tokens displayed: {MAX_INDIVIDUAL_TOKENS}")
    print(f"  Show combined tokens: {SHOW_COMBINED_TOKENS}")
    print(f"  DPI: {DPI}")
    print("=" * 80)

    # Select random images
    images = get_random_images(TRAIN_DIR, IMAGE_COUNT, RANDOM_SEED)

    # Initialize tokenizer with optimized parameters
    print("\nInitializing ColorRegionTokenizer...")
    tokenizer = ColorRegionTokenizer(
        white_threshold=200,
        black_threshold=50,
        min_region_area=45,
        max_region_area=10000,
        target_height=80,
        padding=5,
        min_saturation=15,
        max_aspect_ratio=8.0,
        split_wide_regions=True,
        normalize_regions=True
    )
    print("  [OK] Tokenizer initialized")
    print(f"    - min_region_area: {tokenizer.min_region_area}")
    print(f"    - max_region_area: {tokenizer.max_region_area}")
    print(f"    - target_height: {tokenizer.target_height}")
    print(f"    - padding: {tokenizer.padding}")
    print(f"    - max_aspect_ratio: {tokenizer.max_aspect_ratio}")

    # Generate visualization
    visualize_pipeline(
        images,
        tokenizer,
        PREPROCESSING_METHOD,
        OUTPUT_FILE,
        DPI,
        inpaint_radius=INPAINT_RADIUS,
        black_threshold=BLACK_THRESHOLD,
        inpaint_threshold=INPAINT_THRESHOLD
    )

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

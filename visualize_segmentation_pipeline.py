"""
Visualize the complete segmentation pipeline from input to final tokens.

Generates visualization images showing:
1. Original input image
2. After preprocessing (hairline removal)
3. Foreground mask
4. Initial connected components
5. After color-based splitting & merging
6. Final token extraction
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from preprocessing.hairline_removal import color_voting_inpaint_dilate
from tokenization.color_region_tokenizer import ColorRegionTokenizer
from scipy.ndimage import label as scipy_label


def visualize_bboxes(image, bboxes, title, ax):
    """Draw bounding boxes on image."""
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10, pad=5)
    ax.axis('off')

    for i, (x, y, w, h) in enumerate(bboxes):
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        # Add box number
        ax.text(x, y - 2, str(i + 1),
                color='red', fontsize=8,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def visualize_tokens(tokens, ax, max_display=10):
    """Display extracted character tokens."""
    ax.axis('off')
    ax.set_title('Final Extracted Tokens', fontsize=10, pad=5)

    if not tokens:
        ax.text(0.5, 0.5, 'No tokens extracted',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Display up to max_display tokens
    display_tokens = tokens[:max_display]
    n = len(display_tokens)

    # Concatenate tokens horizontally with spacing
    max_height = max(t.shape[0] for t in display_tokens)
    spacing = 10
    total_width = sum(t.shape[1] for t in display_tokens) + (n - 1) * spacing

    canvas = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

    x_offset = 0
    for i, token in enumerate(display_tokens):
        h, w = token.shape[:2]
        y_offset = (max_height - h) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = token

        # Draw token number
        cv2.putText(canvas, str(i + 1),
                   (x_offset + 2, y_offset + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        x_offset += w

        # Draw vertical separator line between tokens (except after the last one)
        if i < n - 1:
            # Draw a vertical line in the middle of the spacing
            line_x = x_offset + spacing // 2
            cv2.line(canvas, (line_x, 0), (line_x, max_height - 1), (0, 0, 255), 2)

        x_offset += spacing

    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def get_initial_components(tokenizer, fg_mask):
    """Get initial connected components before color splitting."""
    bboxes = tokenizer.extract_connected_components(fg_mask)
    return bboxes


def get_after_color_processing(tokenizer, image, fg_mask):
    """Get bboxes after color-based splitting and merging."""
    bboxes = tokenizer.extract_connected_components(fg_mask)
    bboxes = tokenizer._split_different_color_tokens(image, bboxes, fg_mask)
    bboxes = tokenizer._merge_same_color_tokens(image, bboxes)
    return bboxes


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize segmentation pipeline')
    parser.add_argument('input_image', type=str, help='Path to input CAPTCHA image')
    parser.add_argument('--output', '-o', type=str, default='segmentation_pipeline.png',
                       help='Output visualization filename')
    parser.add_argument('--output-dir', '-d', type=str, default='figures',
                       help='Output directory (default: figures)')
    args = parser.parse_args()

    # Load image
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found")
        return

    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Error: Failed to load image '{args.input_image}'")
        return

    print(f"Processing: {args.input_image}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize components
    tokenizer = ColorRegionTokenizer()

    # Step 1: Preprocessing
    print("Step 1: Preprocessing (hairline removal)...")
    preprocessed = color_voting_inpaint_dilate(image)

    # Step 2: Foreground mask
    print("Step 2: Creating foreground mask...")
    fg_mask = tokenizer.create_foreground_mask(preprocessed)

    # Step 3: Initial connected components
    print("Step 3: Extracting initial connected components...")
    initial_bboxes = get_initial_components(tokenizer, fg_mask)
    print(f"  Found {len(initial_bboxes)} initial components")

    # Step 4: After color-based processing
    print("Step 4: Applying color-based splitting & merging...")
    final_bboxes = get_after_color_processing(tokenizer, preprocessed, fg_mask)
    print(f"  Final count: {len(final_bboxes)} regions")

    # Step 5: Extract tokens
    print("Step 5: Extracting final tokens...")
    tokens = tokenizer.extract_region_images(preprocessed, final_bboxes)
    print(f"  Extracted {len(tokens)} tokens")

    # Create visualization
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(16, 10))

    # Layout: 3 rows x 2 columns
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)

    # Row 1: Original and Preprocessed
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('1. Original Image', fontsize=10, pad=5)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
    ax2.set_title('2. After Preprocessing\n(Hairline Removal)', fontsize=10, pad=5)
    ax2.axis('off')

    # Row 2: Foreground mask and Initial components
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(fg_mask, cmap='gray')
    ax3.set_title('3. Foreground Mask\n(HSV filtering + morphology)', fontsize=10, pad=5)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 1])
    visualize_bboxes(preprocessed, initial_bboxes,
                     f'4. Initial Connected Components\n({len(initial_bboxes)} regions)', ax4)

    # Row 3: Final segmentation and Tokens
    ax5 = fig.add_subplot(gs[2, 0])
    visualize_bboxes(preprocessed, final_bboxes,
                     f'5. After Color Processing\n({len(final_bboxes)} regions)', ax5)

    ax6 = fig.add_subplot(gs[2, 1])
    visualize_tokens(tokens, ax6)

    # Add overall title
    fig.suptitle('CAPTCHA Segmentation Pipeline', fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    output_path = output_dir / args.output
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")

    # Also save individual steps for the poster
    print("\nSaving individual step images...")

    # Individual saves
    individual_dir = output_dir / 'individual_steps'
    individual_dir.mkdir(exist_ok=True)

    # 1. Original
    cv2.imwrite(str(individual_dir / 'step1_original.png'), image)

    # 2. Preprocessed
    cv2.imwrite(str(individual_dir / 'step2_preprocessed.png'), preprocessed)

    # 3. Foreground mask
    cv2.imwrite(str(individual_dir / 'step3_foreground_mask.png'),
                (fg_mask * 255).astype(np.uint8))

    # 4. Initial components with boxes
    img_initial = preprocessed.copy()
    for x, y, w, h in initial_bboxes:
        cv2.rectangle(img_initial, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(str(individual_dir / 'step4_initial_components.png'), img_initial)

    # 5. Final segmentation with boxes
    img_final = preprocessed.copy()
    for i, (x, y, w, h) in enumerate(final_bboxes):
        cv2.rectangle(img_final, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img_final, str(i + 1), (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(str(individual_dir / 'step5_final_segmentation.png'), img_final)

    # 6. Tokens concatenated
    if tokens:
        max_height = max(t.shape[0] for t in tokens)
        spacing = 10
        total_width = sum(t.shape[1] for t in tokens) + (len(tokens) - 1) * spacing
        token_canvas = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

        x_offset = 0
        for i, token in enumerate(tokens):
            h, w = token.shape[:2]
            y_offset = (max_height - h) // 2
            token_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = token
            x_offset += w

            # Draw red vertical separator between tokens (except after the last one)
            if i < len(tokens) - 1:
                line_x = x_offset + spacing // 2
                cv2.line(token_canvas, (line_x, 0), (line_x, max_height - 1), (0, 0, 255), 2)

            x_offset += spacing

        cv2.imwrite(str(individual_dir / 'step6_extracted_tokens.png'), token_canvas)

    print(f"Individual step images saved to: {individual_dir}")
    print(f"\nTotal: {len(tokens)} characters extracted")
    print("Done!")


if __name__ == '__main__':
    main()

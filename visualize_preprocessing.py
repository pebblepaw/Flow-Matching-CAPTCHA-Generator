"""
Visualize the preprocessing (hairline removal) pipeline step by step.

Shows the detailed breakdown of color voting propagation and inpainting.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing.hairline_removal import (
    color_voting_remove_black_hairline,
    inpainting_remove_black_hairline,
    color_preserving_dilation
)


def detect_black_hairlines(image):
    """Create a visualization of detected black hairlines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = (gray == 0).astype(np.uint8) * 255
    return black_mask


def visualize_preprocessing_steps(image, output_dir):
    """Generate step-by-step preprocessing visualizations."""

    # Step 1: Original
    original = image.copy()

    # Step 1.5: Detect black hairlines
    black_mask = detect_black_hairlines(image)

    # Step 2: Color voting propagation
    after_voting = color_voting_remove_black_hairline(image)

    # Step 3: Inpainting (on the voted result)
    after_inpainting = inpainting_remove_black_hairline(after_voting)

    # Step 4: Color-preserving dilation
    after_dilation = color_preserving_dilation(after_inpainting)

    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)

    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title('1. Original Image', fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(black_mask, cmap='gray')
    ax2.set_title('2. Black Hairline Detection\n(Pure black pixels, value = 0)', fontsize=11, fontweight='bold', pad=8)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(cv2.cvtColor(after_voting, cv2.COLOR_BGR2RGB))
    ax3.set_title('3. After Color Voting Propagation\n(2 iterations, ≥5 neighbor consensus)', fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')

    # Row 2
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(cv2.cvtColor(after_inpainting, cv2.COLOR_BGR2RGB))
    ax4.set_title('4. After Inpainting\n(Fill remaining black pixels)', fontsize=11, fontweight='bold', pad=8)
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(cv2.cvtColor(after_dilation, cv2.COLOR_BGR2RGB))
    ax5.set_title('5. After Color-Preserving Dilation\n(Final result)', fontsize=11, fontweight='bold', pad=8)
    ax5.axis('off')

    # Comparison: Before vs After
    ax6 = fig.add_subplot(gs[1, 2])
    # Create side-by-side comparison
    h, w = original.shape[:2]
    comparison = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:w+10] = 255  # White separator
    comparison[:, w+10:] = after_dilation

    ax6.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    ax6.set_title('6. Before ← | → After\nComparison', fontsize=11, fontweight='bold', pad=8)
    ax6.axis('off')

    # Add overall title
    fig.suptitle('Hairline Removal Preprocessing Pipeline', fontsize=15, fontweight='bold', y=0.98)

    # Save
    output_path = output_dir / 'preprocessing_pipeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    return {
        'original': original,
        'black_mask': black_mask,
        'after_voting': after_voting,
        'after_inpainting': after_inpainting,
        'after_dilation': after_dilation,
        'comparison': comparison
    }


def save_individual_steps(steps, output_dir):
    """Save individual preprocessing step images."""
    individual_dir = output_dir / 'preprocessing_steps'
    individual_dir.mkdir(exist_ok=True)

    # Save each step
    cv2.imwrite(str(individual_dir / 'step1_original.png'), steps['original'])
    cv2.imwrite(str(individual_dir / 'step2_black_detection.png'), steps['black_mask'])
    cv2.imwrite(str(individual_dir / 'step3_color_voting.png'), steps['after_voting'])
    cv2.imwrite(str(individual_dir / 'step4_inpainting.png'), steps['after_inpainting'])
    cv2.imwrite(str(individual_dir / 'step5_final.png'), steps['after_dilation'])
    cv2.imwrite(str(individual_dir / 'step6_comparison.png'), steps['comparison'])

    print(f"Individual steps saved to: {individual_dir}")


def create_detailed_comparison(image, output_dir):
    """Create a detailed comparison focusing on color voting and inpainting."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color voting only
    after_voting = color_voting_remove_black_hairline(image)
    axes[0].imshow(cv2.cvtColor(after_voting, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Color Voting Propagation\n(Assign black pixel to color if ≥5 neighbors agree)',
                     fontsize=12, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Inpainting (on voted result)
    after_inpainting = inpainting_remove_black_hairline(after_voting)
    axes[1].imshow(cv2.cvtColor(after_inpainting, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Inpainting\n(Fill remaining black pixels with surrounding colors)',
                     fontsize=12, fontweight='bold', pad=10)
    axes[1].axis('off')

    plt.suptitle('Preprocessing Methods Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = output_dir / 'preprocessing_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize preprocessing pipeline')
    parser.add_argument('input_image', type=str, help='Path to input CAPTCHA image')
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

    print("\nGenerating preprocessing visualizations...")

    # Generate main pipeline visualization
    steps = visualize_preprocessing_steps(image, output_dir)

    # Save individual steps
    save_individual_steps(steps, output_dir)

    # Create detailed comparison
    create_detailed_comparison(image, output_dir)

    print("\nDone! Generated files:")
    print(f"  - {output_dir}/preprocessing_pipeline.png (complete 6-step view)")
    print(f"  - {output_dir}/preprocessing_comparison.png (voting vs inpainting)")
    print(f"  - {output_dir}/preprocessing_steps/*.png (individual steps)")


if __name__ == '__main__':
    main()

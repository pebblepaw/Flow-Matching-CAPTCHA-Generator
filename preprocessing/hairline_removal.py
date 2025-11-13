"""
Hairline removal preprocessing for CAPTCHA images.

Provides utilities to remove thin black noise lines that connect characters,
making them easier to segment during tokenization.

Two main approaches:
1. Color voting propagation - Recolor pixels based on neighbor consensus
2. Inpainting - Fill black regions with surrounding colors
"""

import cv2
import numpy as np
from collections import Counter


def color_voting_propagation(
    image: np.ndarray,
    target_black_only: bool = False,
    iterations: int = 2
) -> np.ndarray:
    """
    Remove hairlines using color voting propagation.

    For each pixel, if 5+ of its 8 neighbors share the same color,
    recolor this pixel with that color. This effectively removes
    thin black noise lines by propagating character colors over them.

    Args:
        image: RGB image
        target_black_only: If True, only recolor black pixels.
                          If False, recolor all non-white pixels.
        iterations: Number of voting iterations (default: 2)

    Returns:
        Image with hairlines removed
    """
    result = image.copy()

    for _ in range(iterations):
        result = _single_voting_pass(result, target_black_only)

    return result


def _single_voting_pass(image: np.ndarray, target_black_only: bool) -> np.ndarray:
    """
    Single pass of color voting propagation.

    Args:
        image: RGB image
        target_black_only: Whether to only recolor black pixels

    Returns:
        Image after one voting pass
    """
    h, w = image.shape[:2]
    result = image.copy()

    # Detect white pixels (always skip these)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    is_white = (v > 200) & (s < 20)

    # Optionally detect black pixels
    if target_black_only:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        is_black = (gray < 50) & (r < 60) & (g < 60) & (b < 60)

    # Process each pixel
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Always skip white pixels
            if is_white[y, x]:
                continue

            # If black-only mode, skip non-black pixels
            if target_black_only and not is_black[y, x]:
                continue

            # Collect 8-neighbor colors
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    neighbors.append(tuple(result[ny, nx]))

            # Vote: if 5+ neighbors agree on a color, adopt it
            if len(neighbors) == 8:
                color_counts = Counter(neighbors)
                most_common_color, count = color_counts.most_common(1)[0]
                if count >= 5:
                    result[y, x] = np.array(most_common_color, dtype=np.uint8)

    return result


def inpainting_removal(
    image: np.ndarray,
    black_threshold: int = 50,
    inpaint_radius: int = 3
) -> np.ndarray:
    """
    Remove hairlines using OpenCV inpainting.

    Detects black pixels and fills them by interpolating from surrounding colors.

    WARNING: This approach removes ALL black pixels, including legitimate black
    characters. Use with caution or only on images without black characters.

    Args:
        image: RGB image
        black_threshold: Pixels darker than this are considered black
        inpaint_radius: Radius for inpainting algorithm

    Returns:
        Image with black regions inpainted
    """
    # Convert to grayscale to detect black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create mask of black pixels
    black_mask = (gray < black_threshold).astype(np.uint8) * 255

    # Apply inpainting
    result = cv2.inpaint(image, black_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return result


def adaptive_hairline_removal(image: np.ndarray) -> np.ndarray:
    """
    Automatically choose the best hairline removal approach.

    Uses color voting by default as it's safer (preserves black characters).

    Args:
        image: RGB image

    Returns:
        Image with hairlines removed
    """
    # Use color voting with 2 iterations (proven effective)
    return color_voting_propagation(image, target_black_only=False, iterations=2)


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("Hairline Removal Demo")
    print("=" * 70)

    # Test image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "data/raw/train/00wi5-0.png"

    img_path = Path(img_path)

    if not img_path.exists():
        print(f"Error: {img_path} not found")
        sys.exit(1)

    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Processing: {img_path.name}")

    # Apply different methods
    result_voting = color_voting_propagation(img, target_black_only=False, iterations=2)
    result_voting_black = color_voting_propagation(img, target_black_only=True, iterations=2)
    result_inpaint = inpainting_removal(img, black_threshold=50, inpaint_radius=3)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result_voting)
    axes[0, 1].set_title("Color Voting (All Non-White)", fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(result_voting_black)
    axes[1, 0].set_title("Color Voting (Black Only)", fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result_inpaint)
    axes[1, 1].set_title("Inpainting", fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle(f"Hairline Removal: {img_path.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = f"hairline_removal_{img_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")

    plt.show()

"""
Hairline removal preprocessing for CAPTCHA images.

Provides utilities to remove thin black noise lines that connect characters,
making them easier to segment during tokenization.

Two main approaches:
1. Color voting propagation - Recolor pixels based on neighbor consensus
2. Inpainting - Fill black regions with surrounding colors
"""

from cv2.typing import MatLike
import cv2
import numpy as np

def color_voting_propagation(
    image: MatLike,
    mask: MatLike,
    min_neighbors: int = 5,
    iterations: int = 2
) -> MatLike:
    """
    Remove hairlines using color voting propagation.

    For each pixel marked in the mask, if enough of its neighbors share the
    same color, recolor this pixel with that color. This helps propagate
    character colors over thin hairlines.

    Args:
        image: RGB image
        mask: Binary mask indicating pixels eligible for voting (255 = vote)
        connectivity: Neighbor connectivity (4 or 8)
        min_neighbors: Minimum agreeing neighbors required to recolor
        iterations: Number of voting iterations (default: 2)

    Returns:
        Image with hairlines reduced
    """
    
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        raise ValueError("mask must have the same height and width as image")

    for _ in range(iterations):
        image = _single_voting_pass(image, mask, min_neighbors)

    return image


def _single_voting_pass(
    image: MatLike,
    mask: MatLike,
    min_neighbors: int = 5,
) -> MatLike:
    """
    Single pass of color voting propagation.

    Args:
        image: RGB image
        mask: Binary mask where 255 marks pixels eligible for voting

    Returns:
        Image after one voting pass
    """
    h, w = image.shape[:2]
    result = image.copy()

    voting_mask = (mask == 255)      # pixels we are allowed to update
    fixed_mask  = ~voting_mask       # pixels we may sample colors from

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if not voting_mask[y, x]:
                continue

            colors = []
            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and fixed_mask[ny, nx]:
                    colors.append(image[ny, nx])

            if len(colors) < min_neighbors:
                continue

            unique, counts = np.unique(np.array(colors), axis=0, return_counts=True)
            most_idx = np.argmax(counts)
            if counts[most_idx] >= min_neighbors:
                winner = unique[most_idx]
                result[y, x] = winner

    return result
    

def inpainting_remove_black_hairline(
    image: MatLike,
    black_threshold: int = 50,
    inpaint_radius: int = 3
) -> MatLike:
    """
    Remove hairlines using OpenCV inpainting.

    Detects black pixels and fills them by interpolating from surrounding colors.

    WARNING: This approach removes ALL black pixels, including legitimate black
    characters. Use with caution or only on images without black characters.

    Args:
        image: BGR image
        black_threshold: Pixels darker than this are considered black
        inpaint_radius: Radius for inpainting algorithm

    Returns:
        Image with black regions inpainted
    """
    # Convert to grayscale to detect black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create mask of black pixels
    black_mask = (gray < black_threshold).astype(np.uint8) * 255

    # Apply inpainting
    result = cv2.inpaint(image, black_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return result


def color_voting_remove_black_hairline(image: MatLike, black_threshold: int = 50) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < black_threshold).astype(np.uint8) * 255
    # Use color voting with 2 iterations (proven effective)
    return color_voting_propagation(image, mask=black_mask, iterations=2)

def color_voting_then_inpainting(image: MatLike, black_threshold: int = 50) -> MatLike:
    result_voting = color_voting_remove_black_hairline(image, black_threshold=black_threshold)
    result_inpainting = inpainting_remove_black_hairline(result_voting, black_threshold=black_threshold)
    return result_inpainting

# Smoke Test for the hairline removal pipeline
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("Hairline Removal Smoke Test")
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
    img = cv2.imread(str(img_path)) # BGR

    print(f"Processing: {img_path.name}")

    # Apply different methods
    result_voting = color_voting_remove_black_hairline(img)
    result_inpaint = inpainting_remove_black_hairline(img)
    result_voting_then_inpainting = inpainting_remove_black_hairline(result_voting)

    # Visualize
    fig, axes = plt.subplots(2, 2)

    axes[0,0].imshow(img)
    axes[0,0].set_title("Original", fontweight='bold')
    axes[0,0].axis('off')

    axes[0,1].imshow(result_voting)
    axes[0,1].set_title("Color Voting (Black Only)", fontweight='bold')
    axes[0,1].axis('off')

    axes[1,0].imshow(result_inpaint)
    axes[1,0].set_title("Inpainting", fontweight='bold')
    axes[1,0].axis('off')

    axes[1,1].imshow(result_voting_then_inpainting)
    axes[1,1].set_title("Color Voting then Inpainting", fontweight='bold')
    axes[1,1].axis('off')

    plt.suptitle(f"Hairline Removal: {img_path.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = f"hairline_removal_{img_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")

    plt.show()

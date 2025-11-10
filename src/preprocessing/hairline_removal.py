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

def brighten_non_black_pixels(image: MatLike, black_threshold: int = 50, brightness_boost: int = 30) -> MatLike:
    """
    Increase the brightness (Value in HSV) of non-black pixels.

    This can help protect dark-but-not-black characters from being treated
    as hairlines during inpainting.

    Args:
        image: BGR image
        black_threshold: Pixels darker than this are considered black
        brightness_boost: Amount to increase Value (0-255, recommended: 20-40)

    Returns:
        Image with brightened non-black pixels
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Identify non-black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_black_mask = gray >= black_threshold

    # Boost the Value channel for non-black pixels
    v_boosted = v.copy().astype(np.int16)
    v_boosted[non_black_mask] = np.clip(v_boosted[non_black_mask] + brightness_boost, 0, 255)
    v_boosted = v_boosted.astype(np.uint8)

    # Merge back and convert to BGR
    hsv_boosted = cv2.merge([h, s, v_boosted])
    result = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    return result


def should_use_inpainting(image: MatLike, black_threshold: int = 50) -> bool:
    """
    Determine if inpainting is safe to use on this image.

    Inpainting should be avoided when there are legitimate black/dark characters,
    as it will smear them. It's safe when:
    1. Very few black pixels remain (mostly colored characters)
    2. Black pixels are thin and scattered (hairlines, not characters)

    Args:
        image: BGR image
        black_threshold: Pixels darker than this are considered black

    Returns:
        True if inpainting is safe, False otherwise
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < black_threshold)

    total_pixels = gray.shape[0] * gray.shape[1]
    black_pixel_count = np.sum(black_mask)
    black_percentage = (black_pixel_count / total_pixels) * 100

    # If more than 5% of pixels are black, there might be black characters
    if black_percentage > 5.0:
        return False

    # Check if black pixels form large connected components (characters)
    # vs small scattered ones (hairlines)
    from scipy.ndimage import label as scipy_label
    labeled, num_components = scipy_label(black_mask)

    if num_components == 0:
        return True  # No black pixels, safe to inpaint

    # Check size of largest black component
    component_sizes = []
    for i in range(1, num_components + 1):
        size = np.sum(labeled == i)
        component_sizes.append(size)

    if len(component_sizes) > 0:
        largest_component = max(component_sizes)
        # If largest black component is more than 100 pixels, it's likely a character
        if largest_component > 100:
            return False

    return True


def color_voting_then_inpainting(
    image: MatLike,
    black_threshold: int = 50,
    inpaint_radius: int = 2,
    adaptive: bool = True
) -> MatLike:
    """
    Apply color voting followed by inpainting for hairline removal.

    With adaptive=True, automatically skips inpainting if it would likely
    smear legitimate black characters.

    Args:
        image: BGR image
        black_threshold: Pixels darker than this are considered black
        inpaint_radius: Radius for inpainting (smaller = weaker effect, default: 2)
        adaptive: If True, only apply inpainting when safe (recommended)

    Returns:
        Image with hairlines removed using both methods (or just color voting if unsafe)
    """
    result_voting = color_voting_remove_black_hairline(image, black_threshold=black_threshold)

    # Check if inpainting is safe to apply
    if adaptive and not should_use_inpainting(result_voting, black_threshold=black_threshold):
        # Skip inpainting, just return color voting result
        return result_voting

    result_inpainting = inpainting_remove_black_hairline(
        result_voting,
        black_threshold=black_threshold,
        inpaint_radius=inpaint_radius
    )
    return result_inpainting


def color_voting_brighten_then_inpaint(
    image: MatLike,
    black_threshold: int = 50,
    inpaint_radius: int = 2,
    brightness_boost: int = 30,
    brighten_threshold: int = None,
    inpaint_threshold: int = None
) -> MatLike:
    """
    Apply color voting, brighten non-black pixels, then inpaint.

    This approach protects dark characters by brightening them before inpainting,
    so they won't be treated as black hairlines.

    Args:
        image: BGR image
        black_threshold: Default threshold for both brightening and inpainting (default: 50)
        inpaint_radius: Radius for inpainting (smaller = weaker effect, default: 2)
        brightness_boost: Amount to increase brightness of non-black pixels (default: 30)
        brighten_threshold: Threshold for identifying pixels to brighten (if None, uses black_threshold)
                           Higher values = more pixels get brightened
        inpaint_threshold: Threshold for inpainting (if None, uses black_threshold)
                          Lower values = fewer pixels get inpainted (more conservative)

    Returns:
        Image with hairlines removed using brightening + inpainting
    """
    # Use separate thresholds if provided, otherwise fall back to black_threshold
    brighten_thresh = brighten_threshold if brighten_threshold is not None else black_threshold
    inpaint_thresh = inpaint_threshold if inpaint_threshold is not None else black_threshold

    # Step 1: Color voting to remove hairlines
    result_voting = color_voting_remove_black_hairline(image, black_threshold=black_threshold)

    # Step 2: Brighten non-black pixels to protect them from inpainting
    # Use higher threshold to catch more dark pixels
    result_brightened = brighten_non_black_pixels(
        result_voting,
        black_threshold=brighten_thresh,
        brightness_boost=brightness_boost
    )

    # Step 3: Inpaint (only truly black pixels will be affected)
    # Use lower threshold to be more conservative
    result_inpainting = inpainting_remove_black_hairline(
        result_brightened,
        black_threshold=inpaint_thresh,
        inpaint_radius=inpaint_radius
    )

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

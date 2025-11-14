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
    black_mask = (gray == 0).astype(np.uint8) * 255

    # Apply inpainting
    result = cv2.inpaint(image, black_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return result


def color_voting_remove_black_hairline(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = (gray == 0).astype(np.uint8) * 255
    # Use color voting with 2 iterations (proven effective)
    return color_voting_propagation(image, mask=black_mask, iterations=2)

def brighten_non_black_pixels(image: MatLike, brightness_boost: int = 30) -> MatLike:
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
    non_black_mask = gray != 0

    # Boost the Value channel for non-black pixels
    v_boosted = v.copy().astype(np.int16)
    v_boosted[non_black_mask] = np.clip(v_boosted[non_black_mask] + brightness_boost, 0, 255)
    v_boosted = v_boosted.astype(np.uint8)

    # Merge back and convert to BGR
    hsv_boosted = cv2.merge([h, s, v_boosted])
    result = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    return result

def color_voting_inpaint(
    image: MatLike,
    inpaint_radius: int = 2
) -> MatLike:
    """
    Apply color voting followed by inpainting for hairline removal.

    With adaptive=True, automatically skips inpainting if it would likely
    smear legitimate black characters.

    Args:
        image: BGR image
        inpaint_radius: Radius for inpainting (smaller = weaker effect, default: 2)

    Returns:
        Image with hairlines removed using both methods (or just color voting if unsafe)
    """
    result_voting = color_voting_remove_black_hairline(image)

    result_inpainting = inpainting_remove_black_hairline(
        result_voting,
        inpaint_radius=inpaint_radius
    )
    return result_inpainting


def color_voting_inpaint_dilate(
    image: MatLike,
    inpaint_radius: int = 2,
    brightness_boost: int = 30,
) -> MatLike:
    """
    Apply color voting, then inpaint and dilate.

    This approach protects dark characters by brightening them before inpainting,
    so they won't be treated as black hairlines.

    Args:
        image: BGR image
        inpaint_radius: Radius for inpainting (smaller = weaker effect, default: 2)
        brightness_boost: Amount to increase brightness of non-black pixels (default: 30)

    Returns:
        Image with hairlines removed using brightening + inpainting
    """

    # Step 1: Color voting to remove hairlines
    result_voting = color_voting_remove_black_hairline(image)

    # Step 2: Inpaint (only truly black pixels will be affected)
    result_inpainting = inpainting_remove_black_hairline(
        result_voting,
        inpaint_radius=inpaint_radius
    )

    # Step 3: Dilate
    result_dilated = color_preserving_dilation(result_inpainting)
    return result_dilated

def color_voting_brighten_inpaint_dilate(
    image: MatLike,
    inpaint_radius: int = 2,
    brightness_boost: int = 30,
) -> MatLike:
    """
    Apply color voting, brighten non-black pixels, then inpaint and dilate.
    """
    result_voting = color_voting_remove_black_hairline(image)
    result_brightened = brighten_non_black_pixels(result_voting, brightness_boost=brightness_boost)
    result_inpainting = inpainting_remove_black_hairline(result_brightened, inpaint_radius=inpaint_radius)
    result_dilated = color_preserving_dilation(result_inpainting)
    return result_dilated

def color_preserving_dilation(
    image: MatLike,
    kernel_size: int = 3,
    white_threshold: int = 255, # in this case, we only have pure white as the background
    core_threshold: int = 200,
    require_no_white_neighbors: bool = False
) -> MatLike:
    """
    Apply selective dilation that expands from character cores outward.

    This strengthens characters by dilating only from strong/saturated pixels
    (the "core" of characters), preventing expansion of faint pixels in gaps
    or artifacts.

    Algorithm:
    1. Identify "core" pixels (dark, saturated - clearly part of characters)
    2. Optionally filter to only dilate pixels with NO white neighbors (prevents gap-filling)
    3. Dilate ONLY these qualified pixels into white areas
    4. Leave faint/weak pixels and gap pixels untouched

    Args:
        image: RGB or BGR image
        kernel_size: Size of dilation kernel (default: 3)
        white_threshold: Pixels brighter than this are considered white (default: 255)
        core_threshold: Pixels darker than this are considered "core" (default: 200)
        require_no_white_neighbors: Only dilate pixels with no white neighbors (default: True)
                                     This prevents filling gaps in characters like "0"

    Returns:
        Image with characters strengthened via selective dilation
    """
    # Convert to grayscale and HSV to identify pixel types
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
    else:
        gray = image.copy()
        saturation = np.zeros_like(gray)

    # Identify pixel types:
    # - White: >= white_threshold (background)
    # - Core: < core_threshold OR high saturation (strong character pixels)
    # - Weak: between core and white (faint pixels, don't dilate these)
    white_mask = (gray >= white_threshold).astype(np.uint8)
    core_mask = ((gray < core_threshold) | (saturation > 50)).astype(np.uint8)

    # If required, only dilate core pixels that are NOT inside gaps
    # A gap is a white region that is surrounded by foreground (like inside "0", "O", "8")
    if require_no_white_neighbors:
        # Find "interior" white pixels (gaps) by:
        # 1. Eroding white mask to remove thin white lines and small gaps
        # 2. Dilating back to restore size of large white regions
        # 3. Remaining white after this is either true background or large gaps
        kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Erode white mask - this removes thin white spaces (gaps in characters)
        white_eroded = cv2.erode(white_mask, kernel_5x5, iterations=1)

        # Dilate back - this restores large white regions (true background)
        white_restored = cv2.dilate(white_eroded, kernel_5x5, iterations=1)

        # Gaps are white pixels that disappeared after erosion
        # (they're surrounded by foreground, not true background)
        gaps = (white_mask > 0) & (white_restored == 0)

        # Dilate the gap mask slightly to include pixels near gaps
        gap_expanded = cv2.dilate(gaps.astype(np.uint8), kernel_5x5, iterations=1)

        # Only allow core pixels that are NOT near gaps to dilate
        core_mask = core_mask & (gap_expanded == 0)

    # Create kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    result = image.copy()

    if len(image.shape) == 3:
        # For color images, dilate each channel
        for c in range(image.shape[2]):
            # Extract only core pixels for this channel
            core_pixels = image[:, :, c].copy()
            core_pixels[core_mask == 0] = 255  # Set non-core to white

            # Dilate the core pixels
            dilated_channel = cv2.dilate(core_pixels, kernel, iterations=1)

            # Only overwrite pixels that are currently white
            result[:, :, c] = np.where(
                white_mask > 0,  # Where white background exists
                dilated_channel,  # Use dilated core value
                image[:, :, c]  # Keep original value
            )
    else:
        # For grayscale images
        core_pixels = image.copy()
        core_pixels[core_mask == 0] = 255
        dilated = cv2.dilate(core_pixels, kernel, iterations=1)
        result = np.where(white_mask > 0, dilated, image)

    return result

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

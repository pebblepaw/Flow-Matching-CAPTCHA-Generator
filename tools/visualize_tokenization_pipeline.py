"""
Visualization script for preprocessing and tokenization pipeline.

Selects random images from data/raw/train, applies preprocessing (hairline removal),
tokenizes them, and visualizes the results with original image and extracted tokens.
"""

IMAGE_COUNT = 15
PREPROCESSING_METHOD = 'color_voting'
TRAIN_DIR = 'data/raw/train'
OUTPUT_FILE = 'pipeline_visualization.png'
# testedseeds： 42， 142
RANDOM_SEED = 42
SHOW_FOREGROUND_MASK = True
TARGET_BLACK_ONLY = True

import random
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple

from src.preprocessing.hairline_removal import (
    color_voting_propagation,
    inpainting_removal,
    adaptive_hairline_removal
)
from src.tokenization.color_region_tokenizer import ColorRegionTokenizer


def get_random_images(train_dir: str, count: int) -> List[Path]:
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    all_images = list(train_path.glob("*.png"))
    if len(all_images) == 0:
        raise ValueError(f"No PNG images found in {train_dir}")

    count = min(count, len(all_images))
    selected = random.sample(all_images, count)
    return selected


def preprocess_image(image: np.ndarray, method: str = "color_voting") -> np.ndarray:
    if method == "color_voting":
        return color_voting_propagation(image, target_black_only=TARGET_BLACK_ONLY, iterations=2)
    elif method == "inpainting":
        return inpainting_removal(image, black_threshold=50, inpaint_radius=3)
    elif method == "adaptive":
        return adaptive_hairline_removal(image)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def visualize_pipeline(
    images: List[Tuple[Path, np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]],
    preprocessing_method: str,
    show_mask: bool = False
):
    n_images = len(images)
    n_cols = 4 if show_mask else 3
    fig = plt.figure(figsize=(6 * n_cols, 3 * n_images))

    for idx, (img_path, original, preprocessed, tokens, fg_mask) in enumerate(images):
        label = img_path.stem.split('-')[0]
        token_mismatch = len(tokens) != len(label)
        row = idx
        col = 0
        ax_original = plt.subplot(n_images, n_cols, row * n_cols + 1)
        ax_original.imshow(original)
        title_color = 'red' if token_mismatch else 'black'
        ax_original.set_title(f"Original\n{img_path.name}",
                             fontsize=9, fontweight='bold', color=title_color)
        ax_original.axis('off')

        if token_mismatch:
            for spine in ax_original.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)

        col += 1

        ax_preprocessed = plt.subplot(n_images, n_cols, row * n_cols + 2)
        ax_preprocessed.imshow(preprocessed)
        ax_preprocessed.set_title(f"Preprocessed\n(hairline removed)",
                                 fontsize=9, fontweight='bold', color=title_color)
        ax_preprocessed.axis('off')

        if token_mismatch:
            for spine in ax_preprocessed.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)

        col += 1

        if show_mask:
            ax_mask = plt.subplot(n_images, n_cols, row * n_cols + 3)
            ax_mask.imshow(fg_mask, cmap='gray')
            ax_mask.set_title(f"Foreground Mask\n({np.sum(fg_mask)} pixels)",
                             fontsize=9, fontweight='bold', color=title_color)
            ax_mask.axis('off')

            if token_mismatch:
                for spine in ax_mask.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    spine.set_visible(True)

            col += 1

        ax_tokens = plt.subplot(n_images, n_cols, row * n_cols + col + 1)

        if len(tokens) > 0:
            max_height = max(token.shape[0] for token in tokens)
            spacing = 10
            total_width = sum(token.shape[1] for token in tokens) + spacing * (len(tokens) - 1)
            canvas = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

            x_offset = 0
            for token in tokens:
                h, w = token.shape[:2]
                y_offset = (max_height - h) // 2
                canvas[y_offset:y_offset+h, x_offset:x_offset+w] = token

                cv2.rectangle(canvas, (x_offset, y_offset), (x_offset+w-1, y_offset+h-1),
                            (0, 0, 0), 2)

                x_offset += w + spacing

            ax_tokens.imshow(canvas)
            ax_tokens.set_title(f"Tokens: {len(tokens)}/{len(label)}\nLabel: '{label}'",
                               fontsize=9, fontweight='bold', color=title_color)
        else:
            ax_tokens.text(0.5, 0.5, "No tokens extracted",
                          ha='center', va='center', fontsize=12, color='red')
            ax_tokens.set_title(f"Tokens: 0/{len(label)}\nLabel: '{label}'",
                               fontsize=9, fontweight='bold', color=title_color)

        ax_tokens.axis('off')

        if token_mismatch:
            for spine in ax_tokens.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.suptitle(f"Preprocessing & Tokenization Pipeline\nMethod: {preprocessing_method}",
                 fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()

    return fig


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("Preprocessing & Tokenization Pipeline Visualization")
    print("=" * 70)
    print(f"Number of images: {IMAGE_COUNT}")
    print(f"Training directory: {TRAIN_DIR}")
    print(f"Preprocessing method: {PREPROCESSING_METHOD}")
    print(f"Output file: {OUTPUT_FILE}")
    if RANDOM_SEED is not None:
        print(f"Random seed: {RANDOM_SEED}")
    print()

    print("Selecting random images...")
    selected_images = get_random_images(TRAIN_DIR, IMAGE_COUNT)
    print(f"Selected {len(selected_images)} images")
    print()

    tokenizer = ColorRegionTokenizer(
        white_threshold=200,
        black_threshold=50,
        min_region_area=25,
        max_region_area=10000,
        target_height=80,
        padding=2,
        min_saturation=15,
        max_aspect_ratio=8.0,
        split_wide_regions=True
    )

    results = []
    for img_path in selected_images:
        print(f"Processing: {img_path.name}")

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(img_rgb, method=PREPROCESSING_METHOD)

        if SHOW_FOREGROUND_MASK:
            tokens, fg_mask = tokenizer.tokenize(
                preprocessed,
                return_mask=True,
                use_color_segmentation=True,
                remove_other_colors=True
            )
        else:
            tokens = tokenizer.tokenize(
                preprocessed,
                use_color_segmentation=True,
                remove_other_colors=True
            )
            fg_mask = None

        label = img_path.stem.split('-')[0]
        print(f"  Label: '{label}' (Expected: {len(label)} chars)")
        print(f"  Tokens extracted: {len(tokens)}")
        print(f"  Token shapes: {[f'{t.shape[1]}x{t.shape[0]}' for t in tokens]}")
        if SHOW_FOREGROUND_MASK and fg_mask is not None:
            print(f"  Foreground pixels: {np.sum(fg_mask)}")
        print()

        results.append((img_path, img_rgb, preprocessed, tokens, fg_mask))

    print("Creating visualization...")
    fig = visualize_pipeline(results, PREPROCESSING_METHOD, show_mask=SHOW_FOREGROUND_MASK)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {OUTPUT_FILE}")
    print()

    print("=" * 70)
    print("Pipeline visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Preprocessing runner that applies hairline removal to images.
Maintains the folder structure from source to destination.

Usage:
    python preprocess_runner.py [--source SOURCE] [--dest DEST] [--verbose]

Examples:
    # Use defaults (data/raw to data/processed)
    python preprocess_runner.py

    # Custom source and destination
    python preprocess_runner.py --source /path/to/source --dest /path/to/dest

    # With verbose output
    python preprocess_runner.py --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm
from hairline_removal import inpainting_remove_black_hairline, color_voting_remove_black_hairline


def process_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Process a single image with hairline removal.

    Applies color voting followed by inpainting as demonstrated in the
    hairline_removal.py smoke test.

    Args:
        image_path: Path to the input image

    Returns:
        Processed image array, or None if processing failed
    """
    try:
        # Load image in BGR format
        img = cv2.imread(str(image_path)) # BGR

        if img is None:
            print(f"Failed to load: {image_path.name}")
            return None

        # Apply preprocessing: voting then inpainting
        result_voting = color_voting_remove_black_hairline(img)
        result_voting_then_inpainting = inpainting_remove_black_hairline(result_voting)

        return result_voting_then_inpainting

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None


def run_preprocessing(
    source_dir: Path,
    dest_dir: Path,
    extensions: tuple = (".png", ".jpg", ".jpeg"),
    verbose: bool = False
) -> dict:
    """
    Process all images in source directory and save to destination,
    maintaining the folder structure.

    Args:
        source_dir: Root source directory
        dest_dir: Root destination directory
        extensions: Tuple of image file extensions to process
        verbose: Print detailed progress information

    Returns:
        Dictionary with statistics about processing
    """

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # Validate source directory
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(source_dir.rglob(f"*{ext}"))

    if not image_files:
        raise ValueError(f"No image files found in {source_dir}")

    if verbose:
        print(f"Found {len(image_files)} images to process")
        print(f"Source: {source_dir}")
        print(f"Destination: {dest_dir}")
        print()

    # Process images with progress bar
    stats = {
        "total": len(image_files),
        "successful": 0,
        "failed": 0,
        "skipped": 0
    }

    for image_path in tqdm(image_files, desc="Processing images"):
        # Calculate relative path and destination
        relative_path = image_path.relative_to(source_dir)
        dest_path = dest_dir / relative_path

        # Create destination subdirectory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already processed
        if dest_path.exists():
            if verbose:
                print(f"Skipped (already exists): {relative_path}")
            stats["skipped"] += 1
            continue

        # Process image
        processed = process_image(image_path)

        if processed is None:
            stats["failed"] += 1
            continue

        # Save processed image
        try:
            cv2.imwrite(str(dest_path), processed)
            stats["successful"] += 1
            if verbose:
                print(f"Processed: {relative_path}")
        except Exception as e:
            print(f"Failed to save {dest_path}: {e}")
            stats["failed"] += 1

    return stats


def print_summary(stats: dict) -> None:
    """Print processing summary statistics."""
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total images:     {stats['total']}")
    print(f"Successful:       {stats['successful']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Skipped:          {stats['skipped']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Apply hairline removal preprocessing to images while maintaining folder structure"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="data/raw",
        help="Source directory containing images (default: data/raw)"
    )

    parser.add_argument(
        "--dest",
        type=str,
        default="data/processed",
        help="Destination directory for processed images (default: data/processed)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    try:
        stats = run_preprocessing(
            source_dir=args.source,
            dest_dir=args.dest,
            verbose=args.verbose
        )

        print_summary(stats)

        if stats["failed"] > 0:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

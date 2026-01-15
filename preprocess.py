"""
Comprehensive preprocessing pipeline for CAPTCHA images
Includes: hairline removal, denoising, normalization, contrast enhancement
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging

# Import hairline removal functions
from hairline_removal import color_voting_propagation, adaptive_hairline_removal


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CAPTCHAPreprocessor:
    """Comprehensive CAPTCHA preprocessing pipeline"""
    
    def __init__(self, config=None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Dictionary with preprocessing parameters
        """
        self.config = config or self.get_default_config()
    
    @staticmethod
    def get_default_config():
        """Default preprocessing configuration"""
        return {
            'remove_hairlines': True,
            'hairline_iterations': 2,
            'target_black_only': False,  # Fixed key name
            'denoise': True,
            'denoise_strength': 5,
            'enhance_contrast': True,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8),
            'sharpen': False,
            'resize': None,  # (height, width) or None
            'normalize_brightness': True,
        }
    
    def remove_hairlines(self, image):
        """
        Remove thin connecting lines between characters
        
        Args:
            image: RGB numpy array
        Returns:
            Processed RGB numpy array
        """
        if not self.config['remove_hairlines']:
            return image
        
        return color_voting_propagation(
            image,
            target_black_only=self.config['target_black_only'],
            iterations=self.config['hairline_iterations']
        )
    
    def denoise(self, image):
        """
        Remove random noise while preserving edges
        
        Args:
            image: RGB numpy array
        Returns:
            Denoised RGB numpy array
        """
        if not self.config['denoise']:
            return image
        
        # Non-local means denoising - excellent for CAPTCHA
        h = self.config['denoise_strength']
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    
    def enhance_contrast(self, image):
        """
        Enhance local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: RGB numpy array
        Returns:
            Enhanced RGB numpy array
        """
        if not self.config['enhance_contrast']:
            return image
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_grid_size']
        )
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def sharpen(self, image):
        """
        Sharpen edges to enhance text clarity
        
        Args:
            image: RGB numpy array
        Returns:
            Sharpened RGB numpy array
        """
        if not self.config['sharpen']:
            return image
        
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def normalize_brightness(self, image):
        """
        Normalize brightness to consistent range
        
        Args:
            image: RGB numpy array
        Returns:
            Normalized RGB numpy array
        """
        if not self.config['normalize_brightness']:
            return image
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Normalize V channel to [50, 255]
        v = hsv[:, :, 2]
        v_min, v_max = v.min(), v.max()
        
        if v_max > v_min:
            v = 50 + (v - v_min) * (205 / (v_max - v_min))
            hsv[:, :, 2] = v
        
        # Convert back
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def resize_image(self, image):
        """
        Resize image if specified
        
        Args:
            image: RGB numpy array
        Returns:
            Resized RGB numpy array
        """
        if self.config['resize'] is None:
            return image
        
        h, w = self.config['resize']
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    def process(self, image):
        """
        Apply full preprocessing pipeline
        
        Args:
            image: RGB numpy array or PIL Image
        Returns:
            Preprocessed RGB numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Apply pipeline in optimal order
        image = self.normalize_brightness(image)
        image = self.remove_hairlines(image)
        image = self.denoise(image)
        image = self.enhance_contrast(image)
        image = self.sharpen(image)
        image = self.resize_image(image)
        
        return image


def process_single_image(img_path, preprocessor, output_dir):
    """
    Process a single image and save result
    
    Args:
        img_path: Path to input image
        preprocessor: CAPTCHAPreprocessor instance
        output_dir: Output directory
    Returns:
        tuple (success, img_path)
    """
    try:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to load: {img_path}")
            return False, img_path
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        processed = preprocessor.process(img_rgb)
        
        # Save
        output_path = output_dir / img_path.name
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), processed_bgr)
        
        return True, img_path
    
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return False, img_path


def preprocess_dataset(input_dir, output_dir, config=None, num_workers=4, limit=None):
    """
    Preprocess entire dataset with multiprocessing
    
    Args:
        input_dir: Input directory with raw images
        output_dir: Output directory for preprocessed images
        config: Preprocessing configuration
        num_workers: Number of parallel workers
        limit: Maximum number of images to process (for testing)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_paths = sorted(list(input_dir.glob("*.png")))
    if limit:
        image_paths = image_paths[:limit]
    
    logger.info(f"Found {len(image_paths)} images in {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using {num_workers} workers")
    
    # Create preprocessor
    preprocessor = CAPTCHAPreprocessor(config)
    logger.info(f"Preprocessing config: {preprocessor.config}")
    
    # Process with multiprocessing
    process_func = partial(process_single_image, preprocessor=preprocessor, output_dir=output_dir)
    
    success_count = 0
    failed_images = []
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, image_paths),
            total=len(image_paths),
            desc="Preprocessing"
        ))
    
    for success, img_path in results:
        if success:
            success_count += 1
        else:
            failed_images.append(img_path)
    
    logger.info(f"Successfully processed: {success_count}/{len(image_paths)}")
    
    if failed_images:
        logger.warning(f"Failed images ({len(failed_images)}):")
        for img_path in failed_images:
            logger.warning(f"  - {img_path}")
    
    return success_count, len(failed_images)


def visualize_preprocessing(input_path, output_path="preprocessing_comparison.png"):
    """
    Visualize preprocessing steps on a single image
    
    Args:
        input_path: Path to input image
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load image
    img = cv2.imread(str(input_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create preprocessor with different configs
    default_config = CAPTCHAPreprocessor.get_default_config()
    configs = [
        ("Original", None),
        ("Hairline Removal", {**default_config, 'denoise': False, 'enhance_contrast': False, 'normalize_brightness': False}),
        ("+ Denoising", {**default_config, 'enhance_contrast': False, 'normalize_brightness': False}),
        ("+ Contrast", {**default_config, 'normalize_brightness': False}),
        ("Full Pipeline", default_config),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (title, config) in enumerate(configs):
        if config is None:
            processed = img_rgb
        else:
            preprocessor = CAPTCHAPreprocessor(config)
            processed = preprocessor.process(img_rgb)
        
        axes[idx].imshow(processed)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide last subplot
    axes[-1].axis('off')
    
    plt.suptitle(f"Preprocessing Pipeline: {Path(input_path).name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    plt.close()


def main(args):
    """Main preprocessing entry point"""
    
    if args.visualize:
        # Visualize preprocessing on a single image
        logger.info(f"Creating visualization for {args.input_dir}")
        sample_images = list(Path(args.input_dir).glob("*.png"))[:1]
        if sample_images:
            visualize_preprocessing(sample_images[0], args.vis_output)
        else:
            logger.error("No images found for visualization")
        return
    
    # Full dataset preprocessing
    config = CAPTCHAPreprocessor.get_default_config()
    
    # Override config from args
    if args.no_hairline:
        config['remove_hairlines'] = False
    if args.no_denoise:
        config['denoise'] = False
    if args.no_contrast:
        config['enhance_contrast'] = False
    if args.sharpen:
        config['sharpen'] = True
    
    # Preprocess train dataset
    if args.train_dir:
        logger.info("=" * 70)
        logger.info("Preprocessing TRAINING dataset")
        logger.info("=" * 70)
        train_output = Path(args.output_dir) / 'train'
        preprocess_dataset(args.train_dir, train_output, config, args.num_workers, args.limit)
    
    # Preprocess test dataset
    if args.test_dir:
        logger.info("=" * 70)
        logger.info("Preprocessing TEST dataset")
        logger.info("=" * 70)
        test_output = Path(args.output_dir) / 'test'
        preprocess_dataset(args.test_dir, test_output, config, args.num_workers, args.limit)
    
    logger.info("=" * 70)
    logger.info("Preprocessing complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess CAPTCHA dataset')
    
    parser.add_argument('--train_dir', type=str, help='Training data directory')
    parser.add_argument('--test_dir', type=str, help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed',
                        help='Output directory for preprocessed data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images (for testing)')
    
    # Preprocessing options
    parser.add_argument('--no_hairline', action='store_true',
                        help='Disable hairline removal')
    parser.add_argument('--no_denoise', action='store_true',
                        help='Disable denoising')
    parser.add_argument('--no_contrast', action='store_true',
                        help='Disable contrast enhancement')
    parser.add_argument('--sharpen', action='store_true',
                        help='Enable sharpening')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization of preprocessing steps')
    parser.add_argument('--vis_output', type=str, default='preprocessing_comparison.png',
                        help='Output path for visualization')
    
    # Quick presets
    parser.add_argument('--input_dir', type=str,
                        help='Single input directory (for visualization or simple processing)')
    
    args = parser.parse_args()
    
    # Handle simple case
    if args.input_dir and not args.train_dir and not args.test_dir:
        args.train_dir = args.input_dir
    
    if not args.train_dir and not args.test_dir:
        parser.error("Must specify --train_dir, --test_dir, or --input_dir")
    
    main(args)

import shutil
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cv2.typing import MatLike
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
import torch

from torchvision import transforms
from src.config import SEED, CHAR_TO_IDX
from src.preprocessing import PreprocessingMethod
from src.tokenization import ColorRegionTokenizer
from tqdm import tqdm


def _create_tokens_visualization(tokens: list[MatLike], target_height: int = 80) -> MatLike:
    """Create a horizontal visualization of all tokens side-by-side.

    Args:
        tokens: List of token images (BGR)
        target_height: Height to resize tokens to (default: 80)

    Returns:
        Combined image with all tokens side-by-side, separated by red lines
    """
    if len(tokens) == 0:
        # Return white canvas with "NO TOKENS" text if no tokens
        empty = np.ones((target_height, 400, 3), dtype=np.uint8) * 255
        cv2.putText(empty, "NO TOKENS EXTRACTED", (10, target_height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return empty

    # Resize all tokens to target height while preserving aspect ratio
    resized_tokens = []
    for token in tokens:
        h, w = token.shape[:2]
        if h != target_height:
            scale = target_height / h
            new_w = max(1, int(w * scale))
            resized = cv2.resize(token, (new_w, target_height))
            resized_tokens.append(resized)
        else:
            resized_tokens.append(token)

    # Calculate total width with spacing (3px for red separator lines)
    spacing = 3
    total_width = sum(t.shape[1] for t in resized_tokens) + spacing * (len(resized_tokens) + 1)

    # Create white canvas
    canvas = np.ones((target_height, total_width, 3), dtype=np.uint8) * 255

    # Place tokens with red separator lines
    x_offset = spacing
    for i, token in enumerate(resized_tokens):
        w = token.shape[1]

        # Draw red separator line before this token (except for first token)
        if i > 0:
            # Red line is at x_offset - 1 (center of spacing)
            cv2.line(canvas, (x_offset - 2, 0), (x_offset - 2, target_height - 1),
                    (0, 0, 255), 1)  # BGR: (0, 0, 255) = Red

        # Place token
        canvas[:, x_offset:x_offset+w] = token
        x_offset += w + spacing

    return canvas


class CaptchaPreprocessPipeline:
    def __init__(
        self,
        image_dir: Path,
        cache_dir: Path,
        preprocessing_method: PreprocessingMethod = PreprocessingMethod.DEFAULT
    ):
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.preprocessing_method = preprocessing_method

        # Initialize tokenizer with optimized settings
        self.tokenizer = ColorRegionTokenizer(
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

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to remove hairlines."""
        return self.preprocessing_method.function(image)
    
    def _get_preprocessed_cache_path(self, img_path: Path) -> Path:
        return self.cache_dir / img_path.relative_to(self.image_dir)
    
    def _get_tokenized_cache_path(self, img_path: Path) -> Path:
        # E.g. data/raw/train/0wi5-0.png -> data/processed/train/0wi5/[0-0.png, 1-w.png, 2-i.png, 3-5.png]
        relative_path = img_path.relative_to(self.image_dir)
        label = img_path.stem
        tokenized_folder = self.cache_dir / relative_path.parent / label
        return tokenized_folder
    
    def _load_preprocessed_image(self, img_path: Path) -> MatLike | None:
        preprocessed_path = self._get_preprocessed_cache_path(img_path)
        if preprocessed_path.exists():
            return cv2.imread(str(preprocessed_path))

        img_bgr_raw = cv2.imread(str(img_path))
        if img_bgr_raw is None:
            return None

        img_bgr = self._preprocess_image(img_bgr_raw)
        preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preprocessed_path), img_bgr)
        return img_bgr

    def _load_tokens(self, img_path: Path) -> list[tuple[MatLike, str]]:
        label = img_path.stem.split('-')[0].upper()

        # Validate label (only alphanumeric)
        if not all(c in CHAR_TO_IDX for c in label):
            return []

        # Try loading from cache
        tokenized_folder = self._get_tokenized_cache_path(img_path)
        word_tokens: list[tuple[MatLike, str]] = []
        if tokenized_folder.exists():
            tokenized_paths = list(tokenized_folder.glob("*.png"))
            if len(tokenized_paths) == len(label):
                for token_path in tokenized_paths:
                    token = cv2.imread(str(token_path), cv2.IMREAD_GRAYSCALE)
                    char = token_path.stem.split('-')[1].upper()
                    word_tokens.append((token, char))
                return word_tokens
        
        # Not in cache, tokenize from raw image and save to cache
        img_bgr = self._load_preprocessed_image(img_path)
        if img_bgr is None:
            failed_dir = self.cache_dir / "000_failed_preprocessing_error"
            failed_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, failed_dir / img_path.name)
            return []
        
        # Tokenize
        try:
            tokens: list[MatLike] = self.tokenizer.tokenize(img_bgr) # type: ignore
        except Exception as e:
            print(f"Error tokenizing {img_path.name}: {e}")
            failed_dir = self.cache_dir / "000_failed_tokenization_error"
            failed_dir.mkdir(parents=True, exist_ok=True)

            # Create 3-row visualization: original, preprocessed, tokens (empty in this case)
            img_bgr_raw = cv2.imread(str(img_path))
            tokens_viz = _create_tokens_visualization([], target_height=img_bgr.shape[0])

            # Pad images to same width
            max_width = max(img_bgr_raw.shape[1], img_bgr.shape[1], tokens_viz.shape[1])
            img_raw_padded = np.pad(img_bgr_raw, ((0, 0), (0, max_width - img_bgr_raw.shape[1]), (0, 0)),
                                   constant_values=255)
            img_proc_padded = np.pad(img_bgr, ((0, 0), (0, max_width - img_bgr.shape[1]), (0, 0)),
                                    constant_values=255)
            tokens_padded = np.pad(tokens_viz, ((0, 0), (0, max_width - tokens_viz.shape[1]), (0, 0)),
                                  constant_values=255)

            combined = np.vstack((img_raw_padded, img_proc_padded, tokens_padded))
            cv2.imwrite(str(failed_dir / img_path.name), combined)
            return []

        # Only use if token count matches label length
        if len(tokens) != len(label):
            failed_dir = self.cache_dir / "000_failed_mismatch_token_count"
            failed_dir.mkdir(parents=True, exist_ok=True)

            # Create 3-row visualization: original, preprocessed, tokens
            img_bgr_raw = cv2.imread(str(img_path))
            tokens_viz = _create_tokens_visualization(tokens, target_height=img_bgr.shape[0])

            # Pad images to same width
            max_width = max(img_bgr_raw.shape[1], img_bgr.shape[1], tokens_viz.shape[1])
            img_raw_padded = np.pad(img_bgr_raw, ((0, 0), (0, max_width - img_bgr_raw.shape[1]), (0, 0)),
                                   constant_values=255)
            img_proc_padded = np.pad(img_bgr, ((0, 0), (0, max_width - img_bgr.shape[1]), (0, 0)),
                                    constant_values=255)
            tokens_padded = np.pad(tokens_viz, ((0, 0), (0, max_width - tokens_viz.shape[1]), (0, 0)),
                                  constant_values=255)

            combined = np.vstack((img_raw_padded, img_proc_padded, tokens_padded))
            cv2.imwrite(str(failed_dir / img_path.name), combined)
            return []

        # Add each character token with its label
        for idx, (token, char) in enumerate(zip(tokens, label)):
            # Convert token to grayscale
            token_gray = cv2.cvtColor(token, cv2.COLOR_BGR2GRAY)
            word_tokens.append((token_gray, char))

            # Save color token
            tokenized_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tokenized_folder / f"{idx}-{char}.png"), token)
        return word_tokens
    
    def run(self, img_path: Path) -> list[tuple[MatLike, str]]:
        return self._load_tokens(img_path)

class CaptchaCharacterDataset(Dataset):
    """
    Dataset that loads CAPTCHA images, tokenizes them, and extracts individual characters.
    """

    def __init__(
        self,
        image_dir: Path,
        cache_dir: Path,
        preprocessing_method: PreprocessingMethod = PreprocessingMethod.DEFAULT,
        transform: transforms.Compose = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
        ])
    ):
        """
        Args:
            image_dir: Directory containing CAPTCHA images
            preprocessing_method: Preprocessing method to use
            target_size: Target size for character images (width, height)
            split: "train" or "val"
            train_ratio: Ratio of training samples
            random_seed: Random seed for splitting
            augment: Whether to apply data augmentation
        """
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.preprocessing_method = preprocessing_method
        self.transform = transform
        self.seed = SEED
        self.processor = CaptchaPreprocessPipeline(image_dir, cache_dir, preprocessing_method)
        self.skipped = 0
        
        # Load and prepare dataset
        print(f"Loading dataset from {image_dir}...")
        self.samples = self._prepare_dataset()
        print(f"Loaded {len(self.samples)} character samples")


    def _prepare_dataset(self) -> list[tuple[MatLike, str]]:
        """
        Load all CAPTCHA images, tokenize them, and extract individual characters.
        Only keep samples where token count matches label length.
        """
        image_paths = sorted(list(self.image_dir.glob("*.png")))

        if len(image_paths) == 0:
            raise ValueError(f"No PNG images found in {self.image_dir}")

        samples: list[tuple[MatLike, str]] = []

        print(f"Processing {len(image_paths)} CAPTCHA images...")

        for img_path in tqdm(image_paths, desc="Tokenizing images"):
            tokens = self.processor.run(img_path)
            if len(tokens) == 0:
                self.skipped += 1
                continue
            samples.extend(tokens)

        print(f"Skipped {self.skipped} images (invalid/failed)")
        print(f"Total CAPTCHA samples: {len(samples)}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_img, char_label = self.samples[idx]

        token_tensor = self.transform(token_img)

        # Get label index
        label_idx = CHAR_TO_IDX[char_label]

        return token_tensor, label_idx

class CaptchaWordDataset(Dataset):
    """
    Dataset that loads CAPTCHA images, tokenizes them, and extracts individual words.
    """

    def __init__(
        self,
        image_dir: Path,
        cache_dir: Path,
        preprocessing_method: PreprocessingMethod = PreprocessingMethod.DEFAULT,
        transform: transforms.Compose = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
        ])
    ):
        """
        Args:
            image_dir: Directory containing CAPTCHA images
            preprocessing_method: Preprocessing method to use
            target_size: Target size for character images (width, height)
            split: "train" or "val"
            train_ratio: Ratio of training samples
            random_seed: Random seed for splitting
            augment: Whether to apply data augmentation
        """
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.preprocessing_method = preprocessing_method
        self.transform = transform
        self.seed = SEED
        self.processor = CaptchaPreprocessPipeline(image_dir, cache_dir, preprocessing_method)
        self.skipped = 0

        # Load and prepare dataset
        print(f"Loading dataset from {image_dir}...")
        self.samples = self._prepare_dataset()
        print(f"Loaded {len(self.samples)} character samples")


    def _prepare_dataset(self) -> list[list[tuple[MatLike, str]]]:
        """
        Load all CAPTCHA images, tokenize them, and extract individual characters.
        Only keep samples where token count matches label length.
        """
        image_paths = sorted(list(self.image_dir.glob("*.png")))

        if len(image_paths) == 0:
            raise ValueError(f"No PNG images found in {self.image_dir}")

        samples: list[list[tuple[MatLike, str]]] = []

        print(f"Processing {len(image_paths)} CAPTCHA images...")

        for img_path in tqdm(image_paths, desc="Tokenizing images"):
            # copy failed preprocessing images to a separate directory
            tokens = self.processor.run(img_path)
            if len(tokens) == 0:
                self.skipped += 1
                continue
            samples.append(tokens)

        print(f"Skipped {self.skipped} images (invalid/failed)")
        print(f"Total CAPTCHA word samples: {len(samples)}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        tokens_tensors = [self.transform(token_img) for token_img, _ in tokens]
        labels = [CHAR_TO_IDX[token_label] for _, token_label in tokens]

        return tokens_tensors, labels

def create_dataloaders(
    img_dir: Path,
    img_cache_dir: Path,
    train_val_ratio: float = 0.8,
    batch_size: int = 64,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        img_dir: Directory containing images
        img_cache_dir: Directory to cache processed images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = CaptchaCharacterDataset(img_dir, img_cache_dir)

    # Split into train and val
    train_len = int(len(dataset) * train_val_ratio)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(SEED)
    )

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(
        img_dir=Path("data/filtered/train"),
        img_cache_dir=Path("data/processed/train_cache"),
        train_val_ratio=0.8,
        batch_size=64,
        num_workers=0
    )
    print(len(train_loader), len(val_loader))

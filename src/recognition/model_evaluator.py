"""
Word-level accuracy visualizer for CAPTCHA recognition.

Visualizes prediction results at the word level (all characters must be correctly
predicted for a word to be considered correct), using CaptchaWordDataset.
"""

import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch

from src.config import IDX_TO_CHAR, SEED
from src.recognition.data_loader import CaptchaWordDataset
from src.recognition.model.character_cnn import CharacterCNN


def load_trained_model(model_path: Path, device: torch.device) -> CharacterCNN:
    """Load the trained character recognition model from checkpoint."""
    model = CharacterCNN(input_channels=1, num_classes=36)
    checkpoint = torch.load(str(model_path), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)

    print(f"Loaded model from: {model_path}")

    return model


def predict_word(
    model: CharacterCNN,
    tokens: List[torch.Tensor],
    device: torch.device
) -> Tuple[str, List[float]]:
    """
    Predict a complete word from a list of character token tensors.

    Args:
        model: Trained CharacterCNN model
        tokens: List of token tensors (already preprocessed and transformed)
        device: torch device

    Returns:
        Tuple of (predicted_word_string, list_of_confidences)
    """
    predictions = []
    confidences = []

    for token in tokens:
        token = torch.stack([token]).to(device)

        # Predict character
        pred, conf = model.predict_char(token, return_confidence=True)
        predictions.append(pred)
        confidences.append(conf)

    return ''.join(predictions), confidences

def get_best_model_path(model_name: str) -> Path:
    """Get the path to the best model checkpoint."""
    model_files = list((Path("checkpoints") / model_name).glob("best_model_*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No trained model found in {Path('checkpoints') / model_name}")

    # Sort by validation accuracy (extract from filename)
    model_files.sort(key=lambda x: float(x.stem.split('_val_acc')[-1]), reverse=True)
    return model_files[0]

def main(
    model_name: str,
    image_dir: str = "data/raw/test",
    cache_dir: str = "data/processed/test_cache",
):
    """
    Main function to run word-level accuracy visualization.

    Args:
        model_name: Name of the trained model
        image_dir: Directory containing CAPTCHA images
        cache_dir: Directory for caching preprocessed data
        preprocessing_method: Preprocessing method to use
    """
    model_path = get_best_model_path(model_name)

    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Word-Level CAPTCHA Recognition Accuracy Visualizer")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Image directory: {image_dir}")
    print(f"Cache directory: {cache_dir}")
    print("=" * 70)
    print()

    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path, device)
    print()

    # Load dataset
    print("Loading dataset...")
    image_path = Path(image_dir)
    cache_path = Path(cache_dir)
    dataset = CaptchaWordDataset(
        image_dir=image_path,
        cache_dir=cache_path
    )
    print()

    failed = dataset.skipped
    valid = len(dataset)
    total_samples = failed + valid

    print(f"Processing {total_samples} samples (failed_preprocessing: {failed}, valid: {valid})...\n")

    # Process samples and make predictions
    results = []
    word_correct_count = 0
    total_words = 0
    total_chars_correct = 0
    total_chars = 0

    for sample_idx in range(len(dataset)):
        # Get sample from dataset
        tokens_tensors, label_indices = dataset[sample_idx]

        # Convert label indices back to characters
        gt_word = ''.join(IDX_TO_CHAR[idx] for idx in label_indices)
        total_words += 1
        total_chars += len(gt_word)

        # Get token images (need to access internal structure)
        sample_tokens_raw = dataset.samples[sample_idx]
        tokens_images = [token_img for token_img, _ in sample_tokens_raw]

        # Predict word
        pred_word, confidences = predict_word(model, tokens_tensors, device)

        # Check word-level correctness (all characters must match)
        is_correct = (pred_word == gt_word)
        if is_correct:
            word_correct_count += 1

        # Character-level accuracy
        for gt_char, pred_char in zip(gt_word, pred_word):
            if gt_char == pred_char:
                total_chars_correct += 1

        # Store result
        results.append((sample_idx, gt_word, pred_word, tokens_images, confidences, is_correct))

        # Print result
        # status = "✓ CORRECT" if is_correct else "✗ WRONG"
        # print(f"Sample {sample_idx}: '{gt_word}' -> '{pred_word}' {status}")
        # if confidences:
        #     avg_conf = sum(confidences) / len(confidences)
        #     print(f"  Avg confidence: {avg_conf*100:.1f}%")

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Word-Level Accuracy:      {word_correct_count}/{total_words} = {100*word_correct_count/total_words:.2f}%")
    print(f"Character-Level Accuracy: {total_chars_correct}/{total_chars} = {100*total_chars_correct/total_chars:.2f}%")
    print(f"Segmentation Accuracy:    {total_samples - failed}/{total_samples} = {100*(total_samples - failed)/total_samples:.2f}%")
    print(f"Overall (incl. failed):   {word_correct_count}/{total_samples} = {100*word_correct_count/total_samples:.2f}%")
    print("=" * 70)
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CAPTCHA recognition model")
    parser.add_argument("--model-name", type=str, help="Name of the trained model")
    args = parser.parse_args()

    if args.model_name is None:
        parser.error("Model name is required")

    main(model_name=args.model_name)

"""
Visualize character recognition results on CAPTCHA images.

Loads the trained model and displays predictions alongside ground truth labels.
"""

import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple

from model_structure import CharacterCNN
from preprocessing.hairline_removal import color_voting_propagation
from tokenization.color_region_tokenizer import ColorRegionTokenizer


def load_trained_model(model_path: str, device: torch.device) -> CharacterCNN:
    """Load the trained character recognition model."""
    model = CharacterCNN(input_channels=1, num_classes=36)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    print(f"Loaded model from: {model_path}")
    print(f"Model validation accuracy: {checkpoint['val_acc']*100:.2f}%")
    print(f"Trained for {checkpoint['epoch']} epochs")

    return model


def predict_tokens(
    model: CharacterCNN,
    tokens: List[np.ndarray],
    device: torch.device,
    target_size: Tuple[int, int] = (32, 32)
) -> List[Tuple[str, float]]:
    """
    Predict characters for a list of token images.

    Returns:
        List of (predicted_char, confidence) tuples
    """
    predictions = []

    for token in tokens:
        # Convert to grayscale if needed
        if len(token.shape) == 3:
            token_gray = cv2.cvtColor(token, cv2.COLOR_RGB2GRAY)
        else:
            token_gray = token

        # Resize to target size
        token_resized = cv2.resize(token_gray, target_size)

        # Convert to tensor and normalize
        token_tensor = torch.from_numpy(token_resized).float() / 255.0
        token_tensor = token_tensor.unsqueeze(0)  # Add channel dimension
        token_tensor = (token_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        token_tensor = token_tensor.unsqueeze(0)  # Add batch dimension
        token_tensor = token_tensor.to(device)

        # Predict
        with torch.no_grad():
            output = model(token_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_char = model.idx_to_char[predicted_idx.item()]
            conf_score = confidence.item()

        predictions.append((predicted_char, conf_score))

    return predictions


def visualize_predictions(
    results: List[Tuple[Path, np.ndarray, np.ndarray, List[np.ndarray], str, str, List[float], bool]],
    output_file: str = "prediction_results.png"
):
    """
    Visualize predictions with original images, tokens, and predictions.

    Args:
        results: List of (path, original, preprocessed, tokens, ground_truth, prediction, confidences, is_correct)
    """
    n_images = len(results)
    fig = plt.figure(figsize=(24, 4 * n_images))

    for idx, (img_path, original, preprocessed, tokens, gt_label, pred_label, confidences, is_correct) in enumerate(results):
        row = idx

        # Column 1: Original image
        ax_original = plt.subplot(n_images, 4, row * 4 + 1)
        ax_original.imshow(original)
        title_color = 'green' if is_correct else 'red'
        ax_original.set_title(f"Original\n{img_path.name}", fontsize=10, fontweight='bold', color=title_color)
        ax_original.axis('off')

        if not is_correct:
            for spine in ax_original.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)

        # Column 2: Preprocessed image
        ax_preprocessed = plt.subplot(n_images, 4, row * 4 + 2)
        ax_preprocessed.imshow(preprocessed)
        ax_preprocessed.set_title(f"Preprocessed\n(hairline removed)", fontsize=10, fontweight='bold', color=title_color)
        ax_preprocessed.axis('off')

        if not is_correct:
            for spine in ax_preprocessed.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)

        # Column 3: Extracted tokens
        ax_tokens = plt.subplot(n_images, 4, row * 4 + 3)

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
            ax_tokens.set_title(f"Tokens: {len(tokens)}", fontsize=10, fontweight='bold', color=title_color)
        else:
            ax_tokens.text(0.5, 0.5, "No tokens", ha='center', va='center', fontsize=12, color='red')
            ax_tokens.set_title(f"Tokens: 0", fontsize=10, fontweight='bold', color='red')

        ax_tokens.axis('off')

        if not is_correct:
            for spine in ax_tokens.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)

        # Column 4: Prediction results
        ax_pred = plt.subplot(n_images, 4, row * 4 + 4)
        ax_pred.axis('off')

        # Create text display
        result_text = f"Ground Truth: {gt_label}\n"
        result_text += f"Prediction:   {pred_label}\n"
        result_text += f"Match: {'✓ CORRECT' if is_correct else '✗ WRONG'}\n\n"

        if len(confidences) > 0:
            result_text += "Character Confidences:\n"
            for i, (gt_char, pred_char, conf) in enumerate(zip(gt_label, pred_label, confidences)):
                char_match = '✓' if gt_char == pred_char else '✗'
                result_text += f"  {i+1}. {pred_char} ({conf*100:.1f}%) {char_match}\n"

            avg_conf = sum(confidences) / len(confidences)
            result_text += f"\nAvg Confidence: {avg_conf*100:.1f}%"

        text_color = 'green' if is_correct else 'red'
        ax_pred.text(0.1, 0.95, result_text,
                    fontsize=10,
                    verticalalignment='top',
                    fontfamily='monospace',
                    color=text_color,
                    fontweight='bold')

        if not is_correct:
            for spine in ax_pred.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.suptitle("Character Recognition Results", fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")

    return fig


def main():
    # Configuration
    MODEL_PATH = "char_recognizer.pth"
    TEST_DIR = "data/raw/train"  # Can also use a separate test directory
    NUM_SAMPLES = 15
    RANDOM_SEED = 42
    OUTPUT_FILE = "prediction_results.png"

    # Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("Character Recognition Result Visualization")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print("=" * 70)
    print()

    # Load model
    model = load_trained_model(MODEL_PATH, device)
    print()

    # Initialize tokenizer
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

    # Get random test images
    test_path = Path(TEST_DIR)
    all_images = list(test_path.glob("*.png"))

    if len(all_images) == 0:
        raise ValueError(f"No PNG images found in {TEST_DIR}")

    num_samples = min(NUM_SAMPLES, len(all_images))
    selected_images = random.sample(all_images, num_samples)

    print(f"Processing {num_samples} random images...\n")

    # Process images and make predictions
    results = []
    correct_count = 0
    total_chars_correct = 0
    total_chars = 0

    for img_path in selected_images:
        # Extract ground truth label
        gt_label = img_path.stem.split('-')[0].upper()

        # Load and preprocess image
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preprocessed = color_voting_propagation(img_rgb, target_black_only=True, iterations=2)

        # Tokenize
        try:
            tokens = tokenizer.tokenize(preprocessed)
        except Exception as e:
            print(f"Error tokenizing {img_path.name}: {e}")
            continue

        # Skip if token count doesn't match label
        if len(tokens) != len(gt_label):
            print(f"Skipping {img_path.name}: token count mismatch ({len(tokens)} != {len(gt_label)})")
            continue

        # Make predictions
        predictions = predict_tokens(model, tokens, device)
        pred_chars = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        pred_label = ''.join(pred_chars)

        # Check correctness
        is_correct = (pred_label == gt_label)
        if is_correct:
            correct_count += 1

        # Character-level accuracy
        for gt_char, pred_char in zip(gt_label, pred_label):
            total_chars += 1
            if gt_char == pred_char:
                total_chars_correct += 1

        results.append((img_path, img_rgb, preprocessed, tokens, gt_label, pred_label, confidences, is_correct))

        # Print result
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"{img_path.name}: '{gt_label}' -> '{pred_label}' {status}")
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"  Avg confidence: {avg_conf*100:.1f}%")

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Full CAPTCHA Accuracy: {correct_count}/{len(results)} = {100*correct_count/len(results):.2f}%")
    print(f"Character-level Accuracy: {total_chars_correct}/{total_chars} = {100*total_chars_correct/total_chars:.2f}%")
    print("=" * 70)
    print()

    # Visualize results
    print("Creating visualization...")
    visualize_predictions(results, OUTPUT_FILE)

    print()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
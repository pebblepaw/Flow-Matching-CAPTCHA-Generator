"""
Evaluate generated CAPTCHAs using the trained CharacterCNN classifier.
Tests recognition accuracy on V2 generated images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


class CharacterCNN(nn.Module):
    """CNN for character recognition (same as training)"""
    
    def __init__(self, num_classes=36):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_classifier(checkpoint_path, device):
    """Load trained CharacterCNN classifier."""
    print(f"Loading classifier from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = CharacterCNN(num_classes=36).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Classifier loaded (accuracy: {checkpoint.get('test_accuracy', 'N/A')})")
    return model


def class_to_char(class_idx):
    """Convert class index to character (0-9, a-z)."""
    if class_idx < 10:
        return str(class_idx)
    else:
        return chr(ord('a') + class_idx - 10)


def char_to_class(char):
    """Convert character to class index."""
    if char.isdigit():
        return int(char)
    else:
        return ord(char.lower()) - ord('a') + 10


def extract_characters_from_captcha(captcha_image, num_chars=4, char_size=64, spacing=10):
    """
    Extract individual character images from a CAPTCHA.
    
    Args:
        captcha_image: PIL Image of CAPTCHA
        num_chars: Number of characters in CAPTCHA
        char_size: Size of each character (64x64)
        spacing: Spacing between characters
    
    Returns:
        List of PIL Images (characters)
    """
    chars = []
    x_offset = 0
    
    for i in range(num_chars):
        # Extract character region
        char_img = captcha_image.crop((x_offset, 0, x_offset + char_size, char_size))
        chars.append(char_img)
        x_offset += char_size + spacing
    
    return chars


def evaluate_captchas(classifier, captchas_dir, labels_file, device):
    """
    Evaluate classifier on generated CAPTCHAs.
    
    Args:
        classifier: Trained CharacterCNN model
        captchas_dir: Directory containing generated CAPTCHAs
        labels_file: File with CAPTCHA labels
        device: torch device
    
    Returns:
        Dict with evaluation metrics
    """
    # Load labels
    print(f"\nLoading labels from {labels_file}...")
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split('\t')
            labels_dict[filename] = label.lower()
    
    print(f"Found {len(labels_dict)} CAPTCHAs to evaluate")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Evaluation
    classifier.eval()
    
    total_chars = 0
    correct_chars = 0
    total_captchas = 0
    correct_captchas = 0
    
    per_class_correct = {i: 0 for i in range(36)}
    per_class_total = {i: 0 for i in range(36)}
    
    confusion_examples = []  # Store some misclassifications
    
    with torch.no_grad():
        for filename, true_label in tqdm(labels_dict.items(), desc="Evaluating"):
            captcha_path = Path(captchas_dir) / filename
            
            if not captcha_path.exists():
                continue
            
            # Load CAPTCHA
            captcha_img = Image.open(captcha_path).convert('RGB')
            
            # Extract characters
            char_images = extract_characters_from_captcha(captcha_img)
            
            # Predict each character
            predicted_label = ""
            captcha_correct = True
            
            for i, (char_img, true_char) in enumerate(zip(char_images, true_label)):
                # Prepare input
                char_tensor = transform(char_img).unsqueeze(0).to(device)
                
                # Predict
                output = classifier(char_tensor)
                pred_class = output.argmax(dim=1).item()
                pred_char = class_to_char(pred_class)
                predicted_label += pred_char
                
                # Track accuracy
                true_class = char_to_class(true_char)
                is_correct = (pred_class == true_class)
                
                total_chars += 1
                per_class_total[true_class] += 1
                
                if is_correct:
                    correct_chars += 1
                    per_class_correct[true_class] += 1
                else:
                    captcha_correct = False
                    # Store confusion example
                    if len(confusion_examples) < 20:
                        confusion_examples.append({
                            'filename': filename,
                            'position': i,
                            'true': true_char,
                            'predicted': pred_char
                        })
            
            # CAPTCHA-level accuracy
            total_captchas += 1
            if captcha_correct:
                correct_captchas += 1
    
    # Compute metrics
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    captcha_accuracy = correct_captchas / total_captchas if total_captchas > 0 else 0
    
    per_class_accuracy = {}
    for class_idx in range(36):
        if per_class_total[class_idx] > 0:
            per_class_accuracy[class_to_char(class_idx)] = \
                per_class_correct[class_idx] / per_class_total[class_idx]
    
    results = {
        'character_accuracy': char_accuracy,
        'captcha_accuracy': captcha_accuracy,
        'total_characters': total_chars,
        'correct_characters': correct_chars,
        'total_captchas': total_captchas,
        'correct_captchas': correct_captchas,
        'per_class_accuracy': per_class_accuracy,
        'confusion_examples': confusion_examples
    }
    
    return results


def print_results(results):
    """Print evaluation results in a nice format."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Character Accuracy: {results['character_accuracy']*100:.2f}% "
          f"({results['correct_characters']}/{results['total_characters']})")
    print(f"  CAPTCHA Accuracy:   {results['captcha_accuracy']*100:.2f}% "
          f"({results['correct_captchas']}/{results['total_captchas']})")
    
    print(f"\nPer-Class Accuracy (Top 10 and Bottom 10):")
    sorted_classes = sorted(results['per_class_accuracy'].items(), 
                           key=lambda x: x[1], reverse=True)
    
    print("\n  Best performing:")
    for char, acc in sorted_classes[:10]:
        print(f"    '{char}': {acc*100:.1f}%")
    
    print("\n  Worst performing:")
    for char, acc in sorted_classes[-10:]:
        print(f"    '{char}': {acc*100:.1f}%")
    
    if results['confusion_examples']:
        print(f"\nSample Misclassifications:")
        for ex in results['confusion_examples'][:10]:
            print(f"  {ex['filename']} pos {ex['position']}: "
                  f"'{ex['true']}' → '{ex['predicted']}'")
    
    print("\n" + "="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate generated CAPTCHAs')
    parser.add_argument('--classifier', type=str,
                       default='results/character_cnn/best_model.pt',
                       help='Path to trained CharacterCNN checkpoint')
    parser.add_argument('--captchas_dir', type=str,
                       default='generated_captchas_v2',
                       help='Directory containing generated CAPTCHAs')
    parser.add_argument('--labels_file', type=str,
                       default='generated_captchas_v2/labels.txt',
                       help='File with CAPTCHA labels')
    parser.add_argument('--output', type=str,
                       default='evaluation_results_v2.json',
                       help='Output file for results')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load classifier
    classifier = load_classifier(args.classifier, device)
    
    # Evaluate
    results = evaluate_captchas(
        classifier,
        args.captchas_dir,
        args.labels_file,
        device
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()

#!/bin/bash
# Download and view the latest generated samples

SAMPLE_DIR="./results/character_model_v2_samples"

echo "=================================================="
echo "DOWNLOADING LATEST SAMPLE IMAGES"
echo "=================================================="
echo ""

# Create local directory
mkdir -p "$SAMPLE_DIR"

# Download all sample images
echo "Downloading sample images from cluster..."
scp -q soc-cluster:~/cs4243_captcha/results/character_model_v2/samples/epoch_*.png "$SAMPLE_DIR/" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ Download complete!"
    echo ""
    echo "Sample images saved to: $SAMPLE_DIR"
    echo ""
    echo "Available samples:"
    ls -lh "$SAMPLE_DIR"/epoch_*.png 2>/dev/null
    echo ""
    echo "To view samples, open the PNG files in $SAMPLE_DIR"
    echo "Latest sample: $(ls -t $SAMPLE_DIR/epoch_*.png 2>/dev/null | head -1)"
else
    echo "✗ No samples available yet"
fi

echo ""
echo "=================================================="

#!/bin/bash
#SBATCH --job-name=captcha_flow
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Navigate to project directory
cd $HOME/captcha_flow || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust based on your cluster)
module load python/3.10
module load cuda/12.1

# Print GPU info
nvidia-smi

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Print Python and PyTorch versions
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Training parameters
TRAIN_DIR="./data/train"
OUTPUT_DIR="./output"
BATCH_SIZE=32
EPOCHS=200
LR=2e-4
BASE_CHANNELS=64
NUM_WORKERS=8

echo "=========================================="
echo "Training Configuration:"
echo "  Train dir: $TRAIN_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Base channels: $BASE_CHANNELS"
echo "=========================================="

# Run training
python train.py \
    --train_dir "$TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --base_channels $BASE_CHANNELS \
    --num_workers $NUM_WORKERS \
    --sample_every 10 \
    --save_every 10

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed with exit code $?"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi

# Deactivate virtual environment
deactivate

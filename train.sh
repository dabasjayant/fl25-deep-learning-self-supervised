#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-0:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Fail on error (but allow cleanup to run)
set -u 

# Go to repo root
cd /gpfs/home/rrr9340/fl25-deep-learning-self-supervised

if [ ! -f "train.py" ]; then
    echo "❌ ERROR: train.py not found in $(pwd)"
    exit 1
fi

# Load modules
module load python/gpu/3.10.6-cuda12.9

# Activate venv
source venv/bin/activate

# Create precomputed directory (ensure it exists)
mkdir -p precomputed_augmentations

echo "========================================"
echo "Starting training pipeline"
echo "========================================"

# Trap to save precomputed dataset on exit (timeout, error, etc.)
cleanup() {
    EXIT_CODE=$?
    echo ""
    echo "========================================"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "Job timed out - but precomputed dataset is saved!"
    else
        echo "Training interrupted (exit code: $EXIT_CODE)"
    fi
    echo "Precomputed augmentations saved to: $(pwd)/precomputed_augmentations/"
    echo "You can rerun with: sbatch train.sh"
    echo "Next run will use cached augmentations (much faster)"
    echo "End time: $(date)"
    echo "========================================"
}

trap cleanup EXIT

# Run training with optimized settings
python -u train.py \
    --epochs 50 \
    --batch_size 1024 \
    --num_workers 4 \
    --save_dir .

echo ""
echo "Training script finished!"
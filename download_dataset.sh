#!/bin/bash
# =============================================================================
# Download  Dataset from Hugging Face
# =============================================================================
#  downloads the dataset to Greene HPC scratch space.
#
# Usage:
#   1. Start an interactive session or run via sbatch
#   2. bash download_dataset.sh
#
# =============================================================================

set -e  # Exit on error

# Configuration
DATASET_NAME="tsbpp/fall2025_deeplearning"
SCRATCH="/scratch/${USER}"
DATA_DIR="${SCRATCH}/data/fall2025_deeplearning"

echo "=========================================="
echo "Downloading Fall 2025 Deep Learning Dataset"
echo "=========================================="
echo ""
echo "Dataset: ${DATASET_NAME}"
echo "Target directory: ${DATA_DIR}"
echo ""

# Create data directory
mkdir -p "${DATA_DIR}"

# Note: This script should be run inside Singularity container
# with conda environment activated

# Download dataset using Python
echo "Downloading dataset from Hugging Face..."
python << 'EOF'
from datasets import load_dataset
import os

# Set cache directory to scratch
cache_dir = os.path.join(os.environ.get('SCRATCH', '/scratch/' + os.environ['USER']), 'huggingface_cache')
data_dir = os.path.join(os.environ.get('SCRATCH', '/scratch/' + os.environ['USER']), 'data/fall2025_deeplearning')

print(f"Cache directory: {cache_dir}")
print(f"Data directory: {data_dir}")

# Create directories
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Download dataset
print("\nDownloading dataset...")
dataset = load_dataset(
    "tsbpp/fall2025_deeplearning",
    cache_dir=cache_dir,
    trust_remote_code=True
)

print("\nDataset downloaded successfully!")
print(f"\nDataset info:")
print(dataset)

# Save dataset to disk
print(f"\nSaving dataset to {data_dir}...")
dataset.save_to_disk(data_dir)

print("\nDataset saved successfully!")
print(f"Location: {data_dir}")

# Print dataset statistics
for split in dataset.keys():
    print(f"\n{split} split: {len(dataset[split])} samples")
    if len(dataset[split]) > 0:
        print(f"Features: {dataset[split].features}")
        print(f"Example: {dataset[split][0]}")

EOF

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Dataset location: ${DATA_DIR}"
echo ""
echo "To use this dataset in training, update your config:"
echo "  data:"
echo "    dataset: 'fall2025_deeplearning'"
echo "    data_root: '${DATA_DIR}'"
echo ""

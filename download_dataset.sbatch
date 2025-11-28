#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --account=csci-ga-2572-2025fa
#SBATCH --partition=n1s8-t4-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/download_dataset_%j.out
#SBATCH --error=logs/download_dataset_%j.err

# Download dataset job for Slurm
# Usage: sbatch download_dataset.sbatch

echo "Dataset Download Job Started"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Create logs directory
mkdir -p logs

# Singularity paths (must match env.sh)
export SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
export OVERLAY_FILE=/scratch/${USER}/ssl_env.ext3

# Check if overlay exists
if [ ! -f "${OVERLAY_FILE}" ]; then
    echo "Error: Overlay file not found at ${OVERLAY_FILE}"
    echo "Please run env.sh first to set up the environment"
    exit 1
fi

# Run download script with Singularity
singularity exec --nv \
    --overlay ${OVERLAY_FILE}:ro \
    ${SINGULARITY_IMAGE} \
    /bin/bash -c "
source /ext3/env.sh
conda activate ssl_env
bash download_dataset.sh
"

echo "Dataset Download Job Completed"
echo "Date: $(date)"

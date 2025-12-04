#!/bin/bash
#SBATCH --job-name=dino_base
#SBATCH --account=
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dino_base_multi_4096_%j.out
#SBATCH --error=logs/dino_base_multi_4096_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load any modules you need (adjust based on your cluster)
# module load python/gpu/3.10.6-cuda12.9

# Go to repo root
cd /scratch/{NET_ID}/fl25-deep-learning-self-supervised/src/models

# Activate your conda/venv if needed
source ./../../.venv/bin/activate

# --- Debug Info ---
echo "Job started on $(hostname) at $(date)"
nvidia-smi

# Run caching with optimized settings
accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --mixed_precision fp16 \
    dino_base_large.py \
    --data_path ./../../dataset/images \
    --external_data ./../../dataset/coco/sampled_96 ./../../dataset/inat_2021/sampled_96 ./../../dataset/open_image/sampled_96 ./../../dataset/places365/sampled_96 \
    --epochs 500 \
    --batch_size 128 \
    --grad_accum_steps 16 \
    --workers 14 \
    --lr 0.0005 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 20 \
    --output_dir ./checkpoints_ssl_large

echo "End time: $(date)"
echo "Done!"
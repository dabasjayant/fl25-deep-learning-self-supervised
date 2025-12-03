#!/bin/bash
#SBATCH --job-name=dino_base
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G

#SBATCH --time=2-00:00:00
#SBATCH --output=dino_base_%j.out
#SBATCH --error=dino_base_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load any modules you need (adjust based on your cluster)
module load python/gpu/3.10.6-cuda12.9

# Go to repo root
cd /scratch/{...}/fl25-deep-learning-self-supervised/src/models

# Activate your conda/venv if needed
source ./../../.venv/bin/activate

# Run caching with optimized settings
accelerate launch --multi_gpu dino_base.py \
    --data_path ./../../dataset/images \
    --external_data ./../../dataset/coco/sampled_96 ./../../dataset/inat_2021/sampled_96 ./../../dataset/open_image/sampled_96 ./../../dataset/places365/sampled_96 \
    --epochs 800 \
    --batch_size 512 \
    --grad_accum_steps 4 \
    --workers 14 \
    --lr 0.0005 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 20 \
    --output_dir ./checkpoints_ssl

echo "End time: $(date)"
echo "Done!"
# NYU Greene HPC Setup for Self-Supervised Learning

## Quick Start with Singularity (Recommended by NYU)

NYU HPC recommends using Singularity containers with overlays for reproducible environments. This approach is robust to spot instance preemptions.

### Step 1: Initial Setup (One-Time)

```bash
# SSH to Greene
ssh <netid>@greene.hpc.nyu.edu

# Start interactive session
srun --account=csci-ga-2572-2025fa --partition=n1s8-t4-1 \
     --nodes=1 --cpus-per-task=4 --mem=32GB --time=02:00:00 \
     --gres=gpu:1 --pty /bin/bash

# Clone your repository to scratch
cd /scratch/${USER}
git clone <your-repo-url> fl25-deep-learning-self-supervised
cd fl25-deep-learning-self-supervised

# Run setup script (creates Singularity overlay with conda)
bash env.sh
```

This creates:
- `/scratch/${USER}/ssl_env.ext3` - Singularity overlay (50GB) with Miniconda
- Conda environment named `ssl_env` with PyTorch, datasets, and all dependencies

### Step 2: Download Dataset

```bash
# Submit as batch job (recommended)
sbatch download_dataset.sbatch

# Or download interactively
singularity exec --nv \
    --overlay /scratch/${USER}/ssl_env.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate ssl_env
bash download_dataset.sh
"
```

### Step 3: Start Training

```bash
# Submit training job
sbatch train_ssl

# Monitor logs
tail -f logs/train_<job_id>.out
```

## Spot Instance Resilience

Your jobs are configured with `#SBATCH --requeue` to automatically restart if preempted by spot instance termination.

**Key Features:**
- ✅ Automatic checkpointing every epoch (saves to `/scratch/${USER}/checkpoints`)
- ✅ Always saves `checkpoint_latest.pth` for resuming after preemption
- ✅ Saves `checkpoint_best.pth` for best model
- ✅ Periodic checkpoints every N epochs (configurable)

**To resume training after preemption:**

Edit your config file and set:
```yaml
checkpoint:
  resume: "/scratch/${USER}/checkpoints/<experiment_name>/checkpoint_latest.pth"
```

## Key Files

### Training Scripts
- `train_ssl` - Main Slurm batch script for training
- `download_dataset.sbatch` - Batch script for dataset download
- `env_setup.sh` - One-time environment setup with Singularity overlay
- `train.py` - Training script with checkpointing

### Configuration
- `configs/simclr_resnet18_fall2025.yaml` - Main config for course dataset
- `configs/simclr_resnet18_cifar10.yaml` - Example config for CIFAR-10
- Modify configs to experiment with different methods/architectures

## Singularity Resources

**Overlay templates:** `/share/apps/overlay-fs-ext3/`
- `overlay-15GB-500K.ext3.gz` - 15GB, 500K inodes
- `overlay-50GB-10M.ext3.gz` - 50GB, 10M inodes (used in this setup)

**Singularity images:** `/scratch/work/public/singularity/`
- `cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif` - Older CUDA version
- `cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` - Used in this setup

**Documentation:**
- [NYU HPC Singularity with Miniconda](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda)
- [Google Spot VMs](https://cloud.google.com/compute/docs/instances/spot)

## Data Transfer

To transfer data from Greene storage to cloud burst session:

```bash
# From login node, copy Singularity image
scp -rp greene-dtn:/scratch/work/public/singularity/ubuntu-20.04.3.sif .

# Transfer datasets
scp -rp greene-dtn:/scratch/${USER}/data /scratch/${USER}/
```

## Testing Your Setup

```bash
# Interactive test with Singularity
singularity shell --nv \
    --overlay /scratch/${USER}/ssl_env.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

# Inside container
source /ext3/env.sh
conda activate ssl_env

# Test PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test dataset loading
python -c "from datasets import load_from_disk; print('datasets library working')"
```

## Troubleshooting

**If job fails due to preemption:**
- Check logs: `logs/train_<job_id>.out`
- Verify checkpoint exists: `ls /scratch/${USER}/checkpoints/<experiment_name>/checkpoint_latest.pth`
- Resume by setting `checkpoint.resume` in config and resubmitting

**If overlay is corrupted:**
- Remove: `rm /scratch/${USER}/ssl_env.ext3`
- Re-run: `bash env.sh`

**If dataset download fails:**
- Check network: datasets are downloaded from Hugging Face
- Retry: `sbatch download_dataset.sbatch`

## Experiment Configuration

Edit YAML configs to experiment:

```yaml
# Change SSL method
ssl:
  method: "byol"  # or "moco", "simclr"

# Change backbone
model:
  name: "resnet50"  # or "resnet18", "vit"

# Change dataset
data:
  dataset: "fall2025_deeplearning"
  data_root: "/scratch/${USER}/data/fall2025_deeplearning"
```

Then submit: `sbatch train_ssl`

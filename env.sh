#!/bin/bash
#
# Usage:
#   1. SSH to Greene: ssh <netid>@greene.hpc.nyu.edu
#   2. Start an interactive session (see below)
#   3. Run this script: bash hpc/scripts/setup_env.sh

set -e  # Exit on error

# Configuration
OVERLAY_SIZE="50G-10M"  # 50GB with 10M inodes
OVERLAY_NAME="ssl_env.ext3"
CONDA_ENV_NAME="ssl_env"
PYTHON_VERSION="3.10"

# Paths
SCRATCH="/scratch/${USER}"
OVERLAY_PATH="${SCRATCH}/${OVERLAY_NAME}"
OVERLAY_SOURCE="/scratch/work/public/overlay-fs-ext3/overlay-${OVERLAY_SIZE}.ext3.gz"
SINGULARITY_IMG="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

echo "SSL Environment Setup"
echo "  1. Copy and extract the overlay filesystem"
echo "  2. Install Miniconda in the overlay"
echo "  3. Create a conda environment with SSL dependencies"
echo "Configuration:"
echo "  - Overlay: ${OVERLAY_PATH}"
echo "  - Conda env: ${CONDA_ENV_NAME}"
echo "  - Python: ${PYTHON_VERSION}"

# Check if running in an interactive session
if [ -z "${SLURM_JOB_ID}" ]; then
    echo "WARNING: You should run this in an interactive session!"
    echo ""
    echo "Please run the following command first:"
    echo ""
    echo "  srun --nodes=1 --tasks-per-node=1 --cpus-per-task=4 \\"
    echo "       --mem=32GB --time=02:00:00 --gres=gpu:1 --pty /bin/bash"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create and extract overlay
echo ""
echo "Step 1: Setting up overlay filesystem..."
if [ -f "${OVERLAY_PATH}" ]; then
    echo "  Overlay already exists at ${OVERLAY_PATH}"
    read -p "  Recreate overlay? This will delete existing environment! (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "${OVERLAY_PATH}"
    else
        echo "  Keeping existing overlay"
    fi
fi

if [ ! -f "${OVERLAY_PATH}" ]; then
    echo "  Copying overlay template..."
    cp "${OVERLAY_SOURCE}" "${SCRATCH}/"
    echo "  Extracting overlay (this may take a few minutes)..."
    gunzip -v "${SCRATCH}/overlay-${OVERLAY_SIZE}.ext3.gz"
    mv "${SCRATCH}/overlay-${OVERLAY_SIZE}.ext3" "${OVERLAY_PATH}"
    echo "  Overlay created at ${OVERLAY_PATH}"
fi

# Install Miniconda and create environment
echo ""
echo "Step 2: Installing Miniconda and creating conda environment..."
echo ""

singularity exec --nv \
    --overlay "${OVERLAY_PATH}:rw" \
    "${SINGULARITY_IMG}" \
    /bin/bash << 'SINGULARITY_SCRIPT'

set -e

# Check if conda is already installed
if [ -d "/ext3/miniconda3" ]; then
    echo "  Miniconda already installed"
else
    echo "  Downloading Miniconda..."
    cd /ext3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    echo "  Installing Miniconda..."
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    echo "  Miniconda installed"
fi

# Create environment activation script
echo "  Creating environment activation script..."
cat > /ext3/env.sh << 'ENV_SCRIPT'
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export CONDA_ENVS_PATH=/ext3/miniconda3/envs
ENV_SCRIPT

# Source the environment
source /ext3/env.sh

# Create conda environment if it doesn't exist
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "  Conda environment '${CONDA_ENV_NAME}' already exists"
else
    echo "  Creating conda environment '${CONDA_ENV_NAME}'..."
    conda create -y -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}
fi

# Activate environment and install packages
echo "  Installing packages (this may take several minutes)..."
source /ext3/env.sh
conda activate ${CONDA_ENV_NAME}

# Install PyTorch with CUDA support (skip torchaudio as it's not needed for vision tasks)
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install common deep learning packages
pip install --quiet \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm \
    wandb \
    tensorboard \
    pyyaml \
    omegaconf \
    hydra-core \
    timm \
    einops \
    jupyter \
    huggingface_hub \
    datasets \
    ipykernel \
    ipywidgets \
    pillow

echo ""
echo "  Packages installed successfully!"
echo ""
echo "Installed packages:"
conda list | grep -E "torch|numpy|pandas|wandb|tensorboard"

SINGULARITY_SCRIPT

echo ""
echo "Setup Complete!"
echo "Your environment is ready. To use it in future sessions:"
echo ""
echo "singularity exec --nv \\"
echo "    --overlay /scratch/\${USER}/ssl_env.ext3:ro \\"
echo "    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\"
echo "    /bin/bash -c 'source /ext3/env.sh && conda activate ssl_env && <your command>'"
echo ""
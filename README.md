# Self-Supervised Learning Project

Deep Learning course final project on self-supervised learning.

## Quick Links

- **Framework Documentation**: See `experiments/README.md` for detailed usage
- **HPC Setup**: Run `bash env.sh` to set up environment on Greene
- **Training**: `python train.py --config configs/<config_file>.yaml`

## Structure

- `src/` - Core framework (models, SSL methods, data loading)
- `configs/` - Experiment configurations
- `experiments/` - Results and documentation
- `train.py` - Main training script
- `train_ssl` - HPC job submission script

## Getting Started

1. Setup environment: `bash env.sh` (on Greene) or `pip install torch torchvision pyyaml`
2. Choose a config: `configs/simclr_resnet18_cifar10.yaml`
3. Train: `python train.py --config configs/simclr_resnet18_cifar10.yaml`

See `experiments/README.md` for comprehensive documentation.
# Self-Supervised Learning Framework

A flexible framework for experimenting with different self-supervised learning (SSL) methods and architectures.

## Project Structure

```
fl25-deep-learning-self-supervised/
├── src/
│   ├── models/              # Backbone architectures
│   │   ├── resnet.py        # ResNet-18, ResNet-50
│   │   ├── vit.py           # Vision Transformer
│   │   └── builder.py       # Model factory
│   ├── ssl_methods/         # SSL training methods
│   │   ├── simclr.py        # SimCLR
│   │   ├── moco.py          # MoCo (Momentum Contrast)
│   │   ├── byol.py          # BYOL (Bootstrap Your Own Latent)
│   │   └── builder.py       # SSL method factory
│   ├── data/                # Data loading and augmentation
│   │   ├── datasets.py      # Dataset loaders
│   │   └── transforms.py    # Augmentation transforms
│   └── utils/               # Training utilities
│       ├── train_utils.py   # Training helpers
│       └── logger.py        # Logging utilities
├── configs/                 # Experiment configurations
│   ├── simclr_resnet18_cifar10.yaml
│   ├── moco_resnet50_stl10.yaml
│   └── byol_vit_cifar100.yaml
├── experiments/             # Results and analysis
├── train.py                 # Main training script
├── train_ssl                # HPC job submission script
└── env.sh                   # Environment setup for HPC

```

## Supported Components

### Backbones
- **ResNet**: ResNet-18, ResNet-50
- **Vision Transformer (ViT)**: Configurable depth and width

### SSL Methods
- **SimCLR**: Contrastive learning with NT-Xent loss
- **MoCo**: Momentum contrast with queue of negative samples
- **BYOL**: Self-supervised learning without negative pairs

### Datasets
- CIFAR-10, CIFAR-100
- STL-10
- ImageNet (requires manual download)

## Quick Start

### 1. Setup Environment (Local)

```bash
# Install dependencies
pip install torch torchvision pyyaml

# Optional: Install wandb for logging
pip install wandb
```

### 2. Setup Environment (Greene HPC)

```bash
# Start interactive session
srun --account=csci-ga-2572-2025fa --partition=n1s8-t4-1 \
     --nodes=1 --cpus-per-task=4 --mem=32GB \
     --time=02:00:00 --gres=gpu:1 --pty /bin/bash

# Run environment setup
bash env.sh

# This will:
# - Load CUDA and cuDNN modules
# - Create conda environment with PyTorch
# - Install all required packages
```

### 3. Train a Model

**Local training:**
```bash
# Train SimCLR with ResNet-18 on CIFAR-10
python train.py --config configs/simclr_resnet18_cifar10.yaml

# Train MoCo with ResNet-50 on STL-10
python train.py --config configs/moco_resnet50_stl10.yaml

# Train BYOL with ViT on CIFAR-100
python train.py --config configs/byol_vit_cifar100.yaml
```

**HPC training:**
```bash
# Submit job to Greene
sbatch train_ssl

# Monitor job
squeue -u $USER

# Check logs
tail -f logs/ssl_training_*.out
```

## Creating New Experiments

### 1. Create a Configuration File

```yaml
# configs/my_experiment.yaml
experiment_name: "my_ssl_experiment"

model:
  name: "resnet18"              # or "resnet50", "vit"
  
ssl:
  method: "simclr"              # or "moco", "byol"
  temperature: 0.5
  projection_dim: 128
  
data:
  dataset: "cifar10"
  batch_size: 256
  
training:
  epochs: 200
  learning_rate: 0.3
```

### 2. Run Your Experiment

```bash
python train.py --config configs/my_experiment.yaml
```

## Customization

### Adding a New Backbone

1. Create `src/models/my_model.py`:
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Your architecture here
        
    def forward(self, x, return_features=False):
        # Forward pass
        if return_features:
            return output, features
        return output
```

2. Register in `src/models/builder.py`:
```python
def build_model(config):
    if model_name == 'my_model':
        return MyModel(**config)
```

### Adding a New SSL Method

1. Create `src/ssl_methods/my_method.py`:
```python
import torch.nn as nn

class MySSLMethod(nn.Module):
    def __init__(self, base_encoder, **kwargs):
        super().__init__()
        self.encoder = base_encoder
        # Add projection heads, etc.
        
    def forward(self, x1, x2):
        # SSL forward pass
        return loss
```

2. Register in `src/ssl_methods/builder.py`

## Hyperparameter Tuning

Key hyperparameters to experiment with:

- **Batch size**: Larger is usually better for contrastive methods (256-2048)
- **Learning rate**: Scale with batch size (lr = base_lr * batch_size / 256)
- **Temperature**: Controls hardness of negatives (0.05-0.5)
- **Augmentation strength**: Balance between too easy and too hard
- **Training length**: SSL methods need more epochs (200-1000)

## Monitoring Training

Checkpoints are saved to `checkpoints/<experiment_name>/`

Logs are saved to `logs/`

For W&B integration, set in config:
```yaml
logging:
  use_wandb: true
  wandb_project: "my-ssl-project"
```

## HPC Best Practices

1. **Test locally first**: Debug on small dataset before HPC submission
2. **Use appropriate partitions**: 
   - `n1s8-t4-1` for single GPU experiments
   - `c12m85-a100-1` for A100 GPU
3. **Monitor GPU quota**: You have 300 GPU hours per semester
4. **Save checkpoints frequently**: Enables recovery from node failures
5. **Use scratch space**: Store datasets in `/scratch/$USER/`

## Tips for Good SSL Training

1. **Strong augmentations are crucial**: Use color jitter, blur, and crop
2. **Large batches help**: Especially for SimCLR and MoCo
3. **Longer training**: SSL needs 200-1000 epochs typically
4. **Learning rate scheduling**: Warmup + cosine annealing works well
5. **Projection head size**: 128-256 dimensions is standard
6. **Evaluation**: Linear probe or fine-tuning on downstream tasks

## Troubleshooting

**Import errors**: Make sure you're running from project root
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**CUDA out of memory**: Reduce batch size or use gradient accumulation

**Slow data loading**: Increase `num_workers` in config

**NaN loss**: Reduce learning rate or check augmentations aren't too strong

## References

- [SimCLR Paper](https://arxiv.org/abs/2002.05709)
- [MoCo Paper](https://arxiv.org/abs/1911.05722)
- [BYOL Paper](https://arxiv.org/abs/2006.07733)

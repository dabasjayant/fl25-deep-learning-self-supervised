"""
Self-Supervised Learning Training Script

Usage:
    python train.py --config configs/simclr_resnet18_cifar10.yaml
    
    # With command line overrides:
    python train.py --config configs/simclr_resnet18_cifar10.yaml --batch_size 128 --epochs 100
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import build_model
from ssl_methods import build_ssl_method
from data import get_dataloader
from utils.train_utils import AverageMeter, save_checkpoint, load_checkpoint, adjust_learning_rate
from utils.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SSL Training")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (overrides config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per GPU (overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use for training"
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, optimizer, device, epoch, logger, config):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    
    for batch_idx, (images, _) in enumerate(dataloader):
        # images is a list of two augmented views
        img1, img2 = images[0].to(device), images[1].to(device)
        
        # Forward pass
        ssl_method = config['ssl']['method']
        
        if ssl_method == 'simclr':
            z1, z2 = model(img1, img2)
            loss = model.compute_loss(z1, z2)
        # elif ssl_method == 'moco':
        #     logits, labels = model(img1, img2)
        #     loss = model.compute_loss(logits, labels)
        # elif ssl_method == 'byol':
        #     loss = model(img1, img2)
        else:
            raise ValueError(f"Unknown SSL method: {ssl_method}. Only 'simclr' is currently supported.")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # # Update target network for BYOL (commented out - not using BYOL)
        # if ssl_method == 'byol':
        #     model.update_moving_average()
        
        losses.update(loss.item(), img1.size(0))
        
        # Logging
        if batch_idx % config['logging']['log_every_n_steps'] == 0:
            logger.log(
                f"Epoch [{epoch}][{batch_idx}/{len(dataloader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})"
            )
    
    return losses.avg


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_root'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.resume:
        config['checkpoint']['resume'] = args.resume
    
    # Setup logging
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        use_wandb=config['logging'].get('use_wandb', False)
    )
    
    logger.log("=" * 60)
    logger.log("Self-Supervised Learning Training")
    logger.log("=" * 60)
    logger.log(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    logger.log(f"Model: {config['model']['name']}")
    logger.log(f"SSL Method: {config['ssl']['method']}")
    logger.log(f"Dataset: {config['data']['dataset']}")
    logger.log(f"Device: {args.device}")
    logger.log("=" * 60)
    
    # Build model
    logger.log("Building model...")
    base_encoder = build_model(config['model'])
    model = build_ssl_method(config['ssl'], base_encoder)
    model = model.to(args.device)
    
    # Setup optimizer
    optimizer_name = config['optimizer']['name'].lower()
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if config['checkpoint'].get('resume'):
        start_epoch = load_checkpoint(
            model, optimizer, config['checkpoint']['resume']
        )
    
    # Setup data loader
    logger.log("Loading data...")
    train_loader = get_dataloader(config, is_train=True)
    logger.log(f"Training samples: {len(train_loader.dataset)}")
    
    # Training loop
    logger.log("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, config)
        logger.log(f"\nEpoch {epoch + 1}/{config['training']['epochs']} - LR: {lr:.6f}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, args.device, epoch, logger, config
        )
        
        logger.log(f"Epoch {epoch + 1} - Average Loss: {train_loss:.4f}")
        
        # Update best loss
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss
            logger.log(f"New best loss: {best_loss:.4f}")
        
        # Save checkpoint (always save latest + best)
        checkpoint_dir = Path(config['checkpoint']['save_dir']) / config.get('experiment_name', 'default')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'config': config,
        }
        
        # Always save latest checkpoint (for resuming after preemption)
        latest_path = checkpoint_dir / 'checkpoint_latest.pth'
        save_checkpoint(checkpoint_state, latest_path, is_best=False)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'checkpoint_best.pth'
            save_checkpoint(checkpoint_state, best_path, is_best=False)
        
        # Save periodic checkpoints
        if (epoch + 1) % config['logging']['save_every_n_epochs'] == 0:
            save_checkpoint(
                checkpoint_state,
                filename=checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth',
                is_best=False
            )
            logger.log(f"Checkpoint saved at epoch {epoch + 1}")
    
    logger.log("\n" + "=" * 60)
    logger.log("Training completed!")
    logger.log("=" * 60)


if __name__ == "__main__":
    main()
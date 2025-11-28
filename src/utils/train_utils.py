"""
Training utilities.
"""

import torch
import torch.nn as nn
import os
from pathlib import Path


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth', is_best=False):
    """Save checkpoint to disk."""
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    if is_best:
        best_path = filepath.parent / 'checkpoint_best.pth'
        torch.save(state, best_path)
        print(f"Best checkpoint saved: {best_path}")


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load checkpoint from disk with robust error handling.
    Supports loading with/without optimizer and scheduler state.
    """
    if os.path.exists(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')
        
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Load model state
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']}, best_loss: {best_loss:.4f})")
        return start_epoch, best_loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf')


def adjust_learning_rate(optimizer, epoch, config):
    """Adjust learning rate with cosine annealing."""
    lr = config['training']['learning_rate']
    total_epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 10)
    
    if epoch < warmup_epochs:
        # Linear warmup
        lr = lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

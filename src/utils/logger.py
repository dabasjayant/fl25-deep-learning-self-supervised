"""
Logging utilities.
"""

import os
from datetime import datetime


class Logger:
    """Simple logger for training."""
    
    def __init__(self, log_dir='logs', use_wandb=False):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'train_{timestamp}.log')
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, logging locally only")
                self.use_wandb = False
    
    def log(self, message):
        """Log message to console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_metrics(self, metrics, step):
        """Log metrics."""
        message = f"Step {step}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log(message)
        
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

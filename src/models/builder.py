"""
Model builder for different backbone architectures.
"""

import torch.nn as nn
from .resnet import ResNet18, ResNet50
from .vit import ViT


def build_model(config):
    """
    Build backbone model based on configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        model: PyTorch model instance
    """
    model_name = config.get('name', 'resnet18').lower()
    
    if model_name == 'resnet18':
        model = ResNet18(
            num_classes=config.get('num_classes', 1000),
            zero_init_residual=config.get('zero_init_residual', False)
        )
    elif model_name == 'resnet50':
        model = ResNet50(
            num_classes=config.get('num_classes', 1000),
            zero_init_residual=config.get('zero_init_residual', False)
        )
    elif model_name == 'vit':
        model = ViT(
            image_size=config.get('image_size', 224),
            patch_size=config.get('patch_size', 16),
            num_classes=config.get('num_classes', 1000),
            dim=config.get('dim', 768),
            depth=config.get('depth', 12),
            heads=config.get('heads', 12),
            mlp_dim=config.get('mlp_dim', 3072),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

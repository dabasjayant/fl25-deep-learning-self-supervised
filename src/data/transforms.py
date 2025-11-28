"""
Data augmentation transforms for SSL.
"""

import torchvision.transforms as transforms
from PIL import ImageFilter
import random


class GaussianBlur:
    """Gaussian blur augmentation."""
    
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_ssl_transforms(config, is_train=True):
    """
    Get SSL data augmentation transforms.
    
    Args:
        config: Configuration dict with augmentation parameters
        is_train: Whether these are training transforms
        
    Returns:
        transform: Composition of transforms
    """
    image_size = config.get('image_size', 224)
    
    if is_train:
        # Strong augmentations for SSL
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=config.get('color_jitter_prob', 0.8)),
            transforms.RandomGrayscale(p=config.get('grayscale_prob', 0.2)),
            transforms.RandomApply(
                [GaussianBlur([0.1, 2.0])],
                p=config.get('gaussian_blur_prob', 0.5)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        # Minimal augmentation for evaluation
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # Slightly larger
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return transform


class TwoCropsTransform:
    """Take two random crops of one image for contrastive learning."""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

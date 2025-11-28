"""
Dataset loaders for Fall 2025 Deep Learning course.
Only supports the tsbpp/fall2025_deeplearning dataset from Hugging Face.
"""

import torch
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from .transforms import get_ssl_transforms, TwoCropsTransform


class HuggingFaceDataset(Dataset):
    """Wrapper for Hugging Face datasets."""
    
    def __init__(self, data_root, split='train', transform=None):
        """
        Args:
            data_root: Path to saved dataset directory
            split: Dataset split ('train', 'test', 'validation')
            transform: Transforms to apply
        """
        from datasets import load_from_disk
        
        self.transform = transform
        
        # Load dataset from disk
        dataset_dict = load_from_disk(data_root)
        
        # Get the appropriate split
        if split in dataset_dict:
            self.dataset = dataset_dict[split]
        else:
            # If split not found, try common alternatives
            if split == 'train' and 'train' not in dataset_dict:
                available = list(dataset_dict.keys())
                print(f"Warning: 'train' split not found. Available splits: {available}")
                self.dataset = dataset_dict[available[0]]
            elif split == 'test' and 'test' not in dataset_dict:
                if 'validation' in dataset_dict:
                    self.dataset = dataset_dict['validation']
                else:
                    available = list(dataset_dict.keys())
                    self.dataset = dataset_dict[available[-1]]
            else:
                raise ValueError(f"Split '{split}' not found in dataset")
        
        print(f"Loaded {len(self.dataset)} samples from {split} split")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different possible formats
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            # Assume first key is the image
            image = list(item.values())[0]
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Get label if exists (for evaluation)
        label = item.get('label', item.get('labels', 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataset(config, is_train=True):
    """
    Get dataset for SSL training.
    
    Args:
        config: Configuration dict
        is_train: Whether to get training or validation set
        
    Returns:
        dataset: PyTorch dataset
    """
    dataset_name = config.get('dataset', 'fall2025_deeplearning').lower()
    data_root = config.get('data_root', '/scratch/${USER}/data/fall2025_deeplearning')
    
    # Get transforms
    transform = get_ssl_transforms(config.get('augmentation', {}), is_train)
    
    # For SSL training, apply two crops
    if is_train:
        transform = TwoCropsTransform(transform)
    
    if dataset_name == 'fall2025_deeplearning':
        split = 'train' if is_train else 'test'
        dataset = HuggingFaceDataset(
            data_root=data_root,
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Only 'fall2025_deeplearning' is supported.")
    
    return dataset


def get_dataloader(config, is_train=True):
    """
    Get dataloader for SSL training.
    
    Args:
        config: Configuration dict
        is_train: Whether to get training or validation loader
        
    Returns:
        loader: PyTorch DataLoader
    """
    dataset = get_dataset(config, is_train)
    
    batch_size = config.get('training', {}).get('batch_size', 256)
    num_workers = config.get('data', {}).get('num_workers', 8)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    
    return loader

#!/usr/bin/env python3
"""
Create Kaggle Submission using Trained SimCLR Encoder + kNN
============================================================

This script:
1. Loads your trained SimCLR encoder (frozen)
2. Extracts features from train set (feature bank)
3. Extracts features from test set
4. Uses k-NN to predict labels for test set
5. Creates submission.csv in Kaggle format

Usage:
    python create_submission.py \
        --checkpoint best_model/best_model.pth \
        --data_dir ./kaggle_data_sun397 \
        --output submission.csv \
        --k 20
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


# ==============================================================================
# Model Components (same as train.py/validate.py)
# ==============================================================================

class Encoder(nn.Module):
    """Encoder backbone (ResNet without final classification layer)"""
    
    def __init__(self, architecture='resnet18'):
        super().__init__()
        
        if architecture == 'resnet18':
            resnet = models.resnet18(weights=None)
            self.feature_dim = 512
        elif architecture == 'resnet50':
            resnet = models.resnet50(weights=None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        return features


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning"""
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    """Complete SimCLR model with encoder and projection head"""
    
    def __init__(self, architecture='resnet18'):
        super().__init__()
        self.architecture = architecture
        self.encoder = Encoder(architecture)
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=self.encoder.feature_dim,
            output_dim=128
        )

    def forward(self, x):
        features = self.encoder(x)
        embeddings = self.projection_head(features)
        return embeddings

    def get_features(self, x):
        return self.encoder(x)


# ==============================================================================
# Dataset Classes
# ==============================================================================

class LabeledImageDataset(Dataset):
    """Dataset for images with labels (train/val sets)"""
    
    def __init__(self, image_dir, csv_path, transform=None):
        """
        Args:
            image_dir: Directory containing images
            csv_path: Path to CSV with 'filename' and 'class_id' columns
            transform: Image transforms
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        self.df = pd.read_csv(csv_path)
        self.filenames = self.df['filename'].tolist()
        self.labels = self.df['class_id'].tolist()
        
        print(f"Loaded {len(self.filenames)} labeled images")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.image_dir / self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx], self.filenames[idx]


class UnlabeledImageDataset(Dataset):
    """Dataset for images without labels (test set)"""
    
    def __init__(self, image_dir, csv_path, transform=None):
        """
        Args:
            image_dir: Directory containing images
            csv_path: Path to CSV with 'filename' column
            transform: Image transforms
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        self.df = pd.read_csv(csv_path)
        self.filenames = self.df['filename'].tolist()
        
        print(f"Loaded {len(self.filenames)} test images")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.image_dir / self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.filenames[idx]


# ==============================================================================
# Feature Extraction
# ==============================================================================

def extract_features_labeled(encoder, dataloader, device):
    """Extract features from labeled dataset"""
    encoder.eval()
    all_features = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Extracting train features"):
            images = images.to(device)
            features = encoder(images)
            
            all_features.append(features.cpu())
            all_labels.extend(labels if isinstance(labels, list) else labels.tolist())
            all_filenames.extend(filenames)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.tensor(all_labels)
    
    return all_features, all_labels, all_filenames


def extract_features_unlabeled(encoder, dataloader, device):
    """Extract features from unlabeled dataset"""
    encoder.eval()
    all_features = []
    all_filenames = []

    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Extracting test features"):
            images = images.to(device)
            features = encoder(images)
            
            all_features.append(features.cpu())
            all_filenames.extend(filenames)

    all_features = torch.cat(all_features, dim=0)
    
    return all_features, all_filenames


# ==============================================================================
# k-NN Prediction
# ==============================================================================

def knn_predict_batched(train_features, train_labels, test_features, k=20, batch_size=256):
    """
    k-NN prediction using majority voting (batched for memory efficiency)
    
    Args:
        train_features: [num_train, feature_dim]
        train_labels: [num_train]
        test_features: [num_test, feature_dim]
        k: number of neighbors
        batch_size: batch size for processing
        
    Returns:
        predictions: [num_test]
    """
    all_predictions = []
    
    # Normalize train features once
    train_features_norm = nn.functional.normalize(train_features, dim=1)
    
    for i in tqdm(range(0, len(test_features), batch_size), desc=f"kNN prediction (k={k})"):
        batch_features = test_features[i:i + batch_size]
        batch_features_norm = nn.functional.normalize(batch_features, dim=1)
        
        # Compute similarity: [batch_size, num_train]
        similarity = torch.matmul(batch_features_norm, train_features_norm.T)
        
        # Get top-k nearest neighbors
        topk_similarities, topk_indices = similarity.topk(k, dim=1)
        topk_labels = train_labels[topk_indices]
        
        # Majority voting for batch
        for j in range(topk_labels.shape[0]):
            unique_labels, counts = torch.unique(topk_labels[j], return_counts=True)
            predicted_label = unique_labels[counts.argmax()]
            all_predictions.append(predicted_label.item())
    
    return torch.tensor(all_predictions)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create Kaggle Submission with kNN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained SimCLR checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test folders and CSVs')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--k', type=int, default=20,
                        help='Number of neighbors for kNN')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--use_val', action='store_true',
                        help='Include validation set in feature bank')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)

    # -------------------------------------------------------------------------
    # Load checkpoint and create encoder
    # -------------------------------------------------------------------------
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    architecture = checkpoint.get('architecture', 'resnet18')
    print(f"Architecture: {architecture}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'loss' in checkpoint:
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")

    # Create model and load weights
    model = SimCLR(architecture=architecture)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract encoder and freeze
    encoder = model.encoder
    encoder = encoder.to(device)
    encoder.eval()
    
    for param in encoder.parameters():
        param.requires_grad = False
    print("Encoder loaded and frozen")

    # -------------------------------------------------------------------------
    # Define transform (no augmentation for evaluation)
    # -------------------------------------------------------------------------
    eval_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------
    print("\nLoading datasets...")
    
    # Check what files exist
    train_csv = data_dir / 'train_labels.csv'
    val_csv = data_dir / 'val_labels.csv'
    test_csv = data_dir / 'test_images.csv'
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    # Load train dataset
    train_dataset = LabeledImageDataset(
        data_dir / 'train',
        train_csv,
        transform=eval_transform
    )
    
    # Optionally include validation set in feature bank
    if args.use_val and val_csv.exists():
        val_dataset = LabeledImageDataset(
            data_dir / 'val',
            val_csv,
            transform=eval_transform
        )
        print(f"Including validation set ({len(val_dataset)} images) in feature bank")
    else:
        val_dataset = None
    
    # Load test dataset (unlabeled)
    test_dataset = UnlabeledImageDataset(
        data_dir / 'test',
        test_csv,
        transform=eval_transform
    )

    print(f"\nDataset sizes:")
    print(f"  Train (feature bank): {len(train_dataset)}")
    if val_dataset:
        print(f"  Val (added to feature bank): {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # -------------------------------------------------------------------------
    # Extract features
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Extracting Features")
    print("="*60)
    
    # Extract train features
    train_features, train_labels, _ = extract_features_labeled(encoder, train_loader, device)
    print(f"Train features shape: {train_features.shape}")
    
    # Optionally add validation features
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        val_features, val_labels, _ = extract_features_labeled(encoder, val_loader, device)
        print(f"Val features shape: {val_features.shape}")
        
        # Combine train + val for feature bank
        train_features = torch.cat([train_features, val_features], dim=0)
        train_labels = torch.cat([train_labels, val_labels], dim=0)
        print(f"Combined feature bank shape: {train_features.shape}")
    
    # Extract test features
    test_features, test_filenames = extract_features_unlabeled(encoder, test_loader, device)
    print(f"Test features shape: {test_features.shape}")

    # -------------------------------------------------------------------------
    # k-NN Prediction
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"Running k-NN (k={args.k})")
    print("="*60)
    
    predictions = knn_predict_batched(
        train_features, 
        train_labels, 
        test_features, 
        k=args.k
    )

    # -------------------------------------------------------------------------
    # Create submission file
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Creating Submission")
    print("="*60)
    
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions.numpy()
    })
    
    # Save submission
    submission_df.to_csv(args.output, index=False)
    
    print(f"\nSubmission saved to: {args.output}")
    print(f"Total predictions: {len(submission_df)}")
    
    # Validate format
    print("\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")
    
    # Show statistics
    print(f"\nPrediction statistics:")
    print(f"  Min class_id: {submission_df['class_id'].min()}")
    print(f"  Max class_id: {submission_df['class_id'].max()}")
    print(f"  Unique classes predicted: {submission_df['class_id'].nunique()}")
    
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    
    print(f"\nClass distribution (top 10):")
    print(submission_df['class_id'].value_counts().head(10))


if __name__ == '__main__':
    main()
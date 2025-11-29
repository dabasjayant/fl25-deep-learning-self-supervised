#!/usr/bin/env python3
"""
SimCLR Validation/Evaluation Script
Loads trained encoder, builds kNN feature bank, and evaluates on test set

Usage: 
    # With local directories:
    python validate.py --checkpoint best_model.pth --train_dir eval_public/train --test_dir eval_public/test
    
    # With HuggingFace dataset:
    python validate.py --checkpoint best_model.pth --use_hf
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Optional imports
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not found. Some metrics will be unavailable.")

try:
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False


# ==============================================================================
# Dataset Classes
# ==============================================================================

class LabeledDataset(Dataset):
    """
    Dataset for local directory structure with images organized by label subdirectories
    Structure: root_dir/label_name/image.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        # Build label mapping and collect image paths
        label_dirs = sorted([d for d in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, d))])
        
        for idx, label_dir in enumerate(label_dirs):
            self.label_to_idx[label_dir] = idx
            label_path = os.path.join(root_dir, label_dir)
            
            for fname in os.listdir(label_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(label_path, fname))
                    self.labels.append(idx)
        
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        print(f"Loaded {len(self.image_paths)} images from {self.num_classes} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class HFLabeledDataset(Dataset):
    """
    Wrapper for HuggingFace dataset for evaluation (with labels)
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')

        label = item.get('label', -1)

        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================================================================
# Model Components
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
# Feature Extraction
# ==============================================================================

def extract_features(encoder, dataloader, device):
    """
    Extract features from all images in dataloader using the encoder
    
    Returns:
        features: Tensor of shape [num_samples, feature_dim]
        labels: Tensor of shape [num_samples]
    """
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = encoder(images)

            all_features.append(features.cpu())
            all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels


# ==============================================================================
# kNN Classifier (Majority Voting - matches your notes)
# ==============================================================================

def knn_predict(train_features, train_labels, test_features, k=20):
    """
    k-NN prediction using majority voting
    
    Args:
        train_features: [num_train, feature_dim]
        train_labels: [num_train]
        test_features: [num_test, feature_dim]
        k: number of neighbors
        
    Returns:
        predictions: [num_test]
    """
    # Normalize features (important for cosine similarity)
    train_features = nn.functional.normalize(train_features, dim=1)
    test_features = nn.functional.normalize(test_features, dim=1)
    
    # Compute similarity: [num_test, num_train]
    similarity = torch.matmul(test_features, train_features.T)
    
    # Get top-k nearest neighbors
    topk_similarities, topk_indices = similarity.topk(k, dim=1)  # [num_test, k]
    
    # Get labels of nearest neighbors
    topk_labels = train_labels[topk_indices]  # [num_test, k]
    
    # Majority voting
    predictions = []
    for i in range(topk_labels.shape[0]):
        # Count votes for each label
        unique_labels, counts = torch.unique(topk_labels[i], return_counts=True)
        # Pick label with most votes
        predicted_label = unique_labels[counts.argmax()]
        predictions.append(predicted_label.item())
    
    return torch.tensor(predictions)


def knn_predict_batched(train_features, train_labels, test_features, k=20, batch_size=256):
    """
    k-NN prediction in batches to avoid memory issues
    """
    all_predictions = []
    
    # Normalize train features once
    train_features_norm = nn.functional.normalize(train_features, dim=1)
    
    for i in tqdm(range(0, len(test_features), batch_size), desc=f"kNN prediction (k={k})"):
        batch_features = test_features[i:i + batch_size]
        batch_features_norm = nn.functional.normalize(batch_features, dim=1)
        
        # Compute similarity for this batch
        similarity = torch.matmul(batch_features_norm, train_features_norm.T)
        
        # Get top-k
        topk_similarities, topk_indices = similarity.topk(k, dim=1)
        topk_labels = train_labels[topk_indices]
        
        # Majority voting for batch
        for j in range(topk_labels.shape[0]):
            unique_labels, counts = torch.unique(topk_labels[j], return_counts=True)
            predicted_label = unique_labels[counts.argmax()]
            all_predictions.append(predicted_label.item())
    
    return torch.tensor(all_predictions)


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

def top1_accuracy(predictions, labels):
    """Calculate top-1 accuracy"""
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def top5_accuracy(train_features, train_labels, test_features, test_labels):
    """
    Calculate top-5 accuracy using kNN
    """
    train_features = nn.functional.normalize(train_features, dim=1)
    test_features = nn.functional.normalize(test_features, dim=1)
    
    similarity = torch.matmul(test_features, train_features.T)
    
    # Get top-5 neighbors for each test sample
    # We need enough neighbors to potentially get 5 unique labels
    topk_similarities, topk_indices = similarity.topk(min(50, train_features.shape[0]), dim=1)
    topk_labels = train_labels[topk_indices]
    
    correct = 0
    for i in range(len(test_labels)):
        # Get top 5 most common labels among neighbors
        unique_labels, counts = torch.unique(topk_labels[i], return_counts=True)
        sorted_indices = counts.argsort(descending=True)
        top5_predicted = unique_labels[sorted_indices[:5]]
        
        if test_labels[i] in top5_predicted:
            correct += 1
    
    return correct / len(test_labels)


def per_class_accuracy(predictions, labels, num_classes):
    """Calculate per-class accuracy"""
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    
    for pred, label in zip(predictions, labels):
        label_idx = label.item() if isinstance(label, torch.Tensor) else label
        pred_idx = pred.item() if isinstance(pred, torch.Tensor) else pred
        
        per_class_total[label_idx] += 1
        if pred_idx == label_idx:
            per_class_correct[label_idx] += 1
    
    per_class_acc = per_class_correct / (per_class_total + 1e-8)
    return per_class_acc


# ==============================================================================
# Main Evaluation Function
# ==============================================================================

def evaluate(encoder, train_loader, test_loader, device, k_values=[1, 10, 20], num_classes=None):
    """
    Complete evaluation pipeline
    """
    print("\n" + "="*60)
    print("Building Feature Bank from Training Data")
    print("="*60)
    
    train_features, train_labels = extract_features(encoder, train_loader, device)
    print(f"Feature bank shape: {train_features.shape}")
    print(f"Labels shape: {train_labels.shape}")
    
    if num_classes is None:
        num_classes = len(torch.unique(train_labels))
    print(f"Number of classes: {num_classes}")
    
    print("\n" + "="*60)
    print("Extracting Test Features")
    print("="*60)
    
    test_features, test_labels = extract_features(encoder, test_loader, device)
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Evaluate with different k values
    results = {}
    
    print("\n" + "="*60)
    print("Running kNN Evaluation")
    print("="*60)
    
    for k in k_values:
        print(f"\n--- k = {k} ---")
        predictions = knn_predict_batched(train_features, train_labels, test_features, k=k)
        
        # Top-1 accuracy
        top1_acc = top1_accuracy(predictions, test_labels)
        print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
        
        results[f'k{k}_top1'] = top1_acc
        results[f'k{k}_predictions'] = predictions
    
    # Use best k (default k=20) for detailed metrics
    best_k = 20 if 20 in k_values else k_values[-1]
    predictions = results[f'k{best_k}_predictions']
    
    # Top-5 accuracy
    print(f"\n--- Additional Metrics (using k={best_k}) ---")
    top5_acc = top5_accuracy(train_features, train_labels, test_features, test_labels)
    print(f"Top-5 Accuracy: {top5_acc * 100:.2f}%")
    results['top5'] = top5_acc
    
    # Per-class accuracy
    per_class_acc = per_class_accuracy(predictions, test_labels, num_classes)
    print(f"\nPer-class accuracy (mean): {per_class_acc.mean() * 100:.2f}%")
    print(f"Per-class accuracy (std): {per_class_acc.std() * 100:.2f}%")
    results['per_class_acc'] = per_class_acc
    
    # Sklearn classification report
    if HAS_SKLEARN:
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)
        print(classification_report(test_labels.numpy(), predictions.numpy()))
    
    # Store features for potential reuse
    results['train_features'] = train_features
    results['train_labels'] = train_labels
    results['test_features'] = test_features
    results['test_labels'] = test_labels
    
    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate SimCLR model with kNN')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint')
    
    # Data source options
    parser.add_argument('--use_hf', action='store_true',
                        help='Use HuggingFace dataset instead of local directories')
    parser.add_argument('--train_dir', type=str, default='eval_public/train',
                        help='Path to training data directory (for feature bank)')
    parser.add_argument('--test_dir', type=str, default='eval_public/test',
                        help='Path to test data directory')
    parser.add_argument('--dataset_id', type=str, default='tsbpp/fall2025_deeplearning',
                        help='HuggingFace dataset ID (if using --use_hf)')
    
    # kNN parameters
    parser.add_argument('--k', type=int, nargs='+', default=[1, 10, 20],
                        help='Values of k to evaluate (e.g., --k 1 10 20)')
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='DataLoader workers')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Path to save predictions (e.g., predictions.pt)')
    parser.add_argument('--save_features', type=str, default=None,
                        help='Path to save extracted features (e.g., features.pt)')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    architecture = checkpoint.get('architecture', 'resnet18')
    print(f"Architecture: {architecture}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'loss' in checkpoint:
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")

    # Create and load model
    model = SimCLR(architecture=architecture)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract encoder and freeze
    encoder = model.encoder
    encoder = encoder.to(device)
    encoder.eval()
    
    for param in encoder.parameters():
        param.requires_grad = False
    print("Encoder loaded and frozen")

    # Define evaluation transform (no augmentation, just normalize)
    eval_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create datasets
    print("\nLoading data...")
    
    if args.use_hf:
        if not HAS_HF:
            raise ImportError("datasets library required for HuggingFace datasets. "
                            "Install with: pip install datasets")
        
        dataset = load_dataset(args.dataset_id)
        available_splits = list(dataset.keys())
        print(f"Available splits: {available_splits}")
        
        train_dataset = HFLabeledDataset(dataset['train'], transform=eval_transform)
        test_split = 'test' if 'test' in available_splits else 'validation'
        test_dataset = HFLabeledDataset(dataset[test_split], transform=eval_transform)
        num_classes = None  # Will be inferred
    else:
        # Use local directories
        print(f"Train directory: {args.train_dir}")
        print(f"Test directory: {args.test_dir}")
        
        train_dataset = LabeledDataset(args.train_dir, transform=eval_transform)
        test_dataset = LabeledDataset(args.test_dir, transform=eval_transform)
        num_classes = train_dataset.num_classes

    print(f"Train set size (feature bank): {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

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

    # Run evaluation
    results = evaluate(
        encoder, 
        train_loader, 
        test_loader, 
        device, 
        k_values=args.k,
        num_classes=num_classes
    )

    # Save predictions if requested
    if args.save_predictions:
        best_k = 20 if 20 in args.k else args.k[-1]
        predictions = results[f'k{best_k}_predictions']
        
        torch.save({
            'predictions': predictions,
            'test_labels': results['test_labels'],
            'k': best_k,
            'accuracy': results[f'k{best_k}_top1']
        }, args.save_predictions)
        print(f"\nPredictions saved to: {args.save_predictions}")

    # Save features if requested
    if args.save_features:
        torch.save({
            'train_features': results['train_features'],
            'train_labels': results['train_labels'],
            'test_features': results['test_features'],
            'test_labels': results['test_labels'],
        }, args.save_features)
        print(f"Features saved to: {args.save_features}")

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for k in args.k:
        print(f"k={k:2d} -> Top-1 Accuracy: {results[f'k{k}_top1'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5'] * 100:.2f}%")


if __name__ == '__main__':
    main()
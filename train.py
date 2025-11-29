#!/usr/bin/env python3
"""
SimCLR Training Script for Deep Learning Contest
Usage: python train.py [--epochs NUM] [--batch_size NUM] [--lr FLOAT] [--arch ARCH]
"""

import argparse
import os
from PIL import Image
from tqdm import tqdm
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==============================================================================
# Dataset Classes
# ==============================================================================

class HFUnlabeledDataset(Dataset):
    """
    Wrapper for HuggingFace dataset for SSL pretraining (unlabeled)
    """
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: HuggingFace dataset split (e.g., dataset['train'])
            transform: Augmentation pipeline that returns two views
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        import time
        t0 = time.time()
        
        item = self.hf_dataset[idx]
        t1 = time.time()
        image = item['image']
        t2 = time.time()

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        t3 = time.time()

        image = image.convert('RGB')
        t4 = time.time()

        if self.transform:
            view1, view2 = self.transform(image)
            t5 = time.time()
            
            # Only print first 5 and last 5 calls
            if idx < 5 or idx >= len(self.hf_dataset) - 5:
                print(f"[idx={idx}] hf_get:{t1-t0:.3f}s, image_type:{t2-t1:.3f}s, fromarray:{t3-t2:.3f}s, convert_rgb:{t4-t3:.3f}s, transform:{t5-t4:.3f}s")
            
            return view1, view2

        return image


# ==============================================================================
# Augmentation
# ==============================================================================

class SimCLRAugmentation:
    """SimCLR augmentation - optimized for speed"""
    
    def __init__(self, image_size=96):
        # Heavy augmentations
        self.heavy_aug = A.Compose([
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.2, 1.0),
                p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.8
            ),
            A.ToGray(p=0.2),
        ])
        
        # Normalization (applied to all images at end)
        self.normalize = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def __call__(self, image):
        # Convert PIL to numpy once
        image = np.array(image)
        
        # Apply heavy augmentation twice
        view1 = self.heavy_aug(image=image)['image']
        view2 = self.heavy_aug(image=image)['image']
        
        # Normalize both
        view1 = self.normalize(image=view1)['image']
        view2 = self.normalize(image=view2)['image']
        
        return view1, view2


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
        
        # Remove final classification layer
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
        """Get encoder features without projection (for downstream tasks)"""
        return self.encoder(x)


# ==============================================================================
# Loss Function
# ==============================================================================

class NTXentLoss(nn.Module):
    """Normalized Temperature-Scaled Cross Entropy Loss (NT-XENT)"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = nn.functional.normalize(z_i, dim=-1)
        z_j = nn.functional.normalize(z_j, dim=-1)

        # Concatenate: [2*batch_size, embedding_dim]
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create labels
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        loss = nn.functional.cross_entropy(sim_matrix, labels)
        return loss


# ==============================================================================
# Training Function
# ==============================================================================

def train_SimCLR(model, train_loader, num_epochs=50, lr=0.3, warmup_epochs=10,save_dir='.'):
    """
    Train SimCLR model with progress bars
    
    Args:
        model: SimCLR model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        warmup_epochs: Number of warmup epochs
        save_dir: Base directory to save checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directories
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    best_model_dir = os.path.join(save_dir, 'best_model')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoints_dir}")
    print(f"Best model will be saved to: {best_model_dir}")
    
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    # LR scheduler with warmup and cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = NTXentLoss(temperature=0.1)

    # Mixed precision scaler for fp16 in most computations & critical operations in fp32 (speeds up training)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("Mixed precision training: ENABLED")
    else:
        print("Mixed precision training: DISABLED (no CUDA)")

    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Progress bar for batches
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True,
            dynamic_ncols=True,
            total=len(train_loader)
        )

        for batch_idx, (view1, view2) in enumerate(pbar):
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()

            # NEW: Mixed precision forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    z1 = model(view1)
                    z2 = model(view2)
                    loss = criterion(z1, z2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Fallback for CPU
                z1 = model(view1)
                z2 = model(view2)
                loss = criterion(z1, z2)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'scheduler': scheduler.state_dict(),
                'architecture': model.architecture
            }, os.path.join(best_model_dir, 'best_model.pth'))
            print(f"  -> New best model saved! (loss: {avg_loss:.4f})")

        # Save checkpoint every 10 epochs (or every epoch if num_epochs < 10)
        if (epoch + 1) % 10 == 0 or num_epochs < 10:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'scheduler': scheduler.state_dict(),
                'architecture': model.architecture
            }, os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  -> Checkpoint saved at epoch {epoch+1}")

    # Save final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'scheduler': scheduler.state_dict(),
        'architecture': model.architecture
    }, os.path.join(checkpoints_dir, 'final_model.pth'))
    print(f"Training complete! Final model saved.")
    print(f"Best loss achieved: {best_loss:.4f}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train SimCLR model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.3, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--arch', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50'], help='Encoder architecture')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--cache_dir', type=str, default='./cached_dataset', help='Path to cached dataset')
    args = parser.parse_args()

    # Scale LR by batch size, similar to SimCLR paper
    scaled_lr = args.lr * args.batch_size / 256
    print(f"Base LR: {args.lr}, Batch size: {args.batch_size}, Scaled LR: {scaled_lr:.6f}")

    # Set HuggingFace cache directory
    hf_cache_dir = '/gpfs/scratch/rrr9340/hf_cache'
    os.makedirs(hf_cache_dir, exist_ok=True)
    print(f"HuggingFace cache directory: {hf_cache_dir}")

    # Load dataset
    print("Loading dataset...")
    DATASET_ID = 'tsbpp/fall2025_deeplearning'

    try:
        print(f"Loading from HF cache: {hf_cache_dir}")
        dataset = load_dataset(DATASET_ID, cache_dir=hf_cache_dir)
        print(f"✓ Loaded dataset from cache!")
    except Exception as e:
        print(f"Cache miss or error: {e}")
        print(f"Downloading dataset (will be cached automatically)...")
        dataset = load_dataset(DATASET_ID, cache_dir=hf_cache_dir)
        print(f"✓ Dataset loaded and cached!")

    # Define precomputed paths early
    precomputed_dir = os.path.join(args.save_dir, 'precomputed_augmentations')
    precomputed_file = os.path.join(precomputed_dir, 'augmented_pairs.pt')
    
    try: 
        print(f"Number of pretraining images: {len(dataset['train'])}")

        # Check if pre-computed dataset exists
        if os.path.exists(precomputed_file):
            print(f"\n✓ Found pre-computed augmentations at {precomputed_file}")
            print("Loading pre-computed dataset...")
            precomputed_pairs = torch.load(precomputed_file)
            print(f"✓ Loaded {len(precomputed_pairs)} pre-computed pairs")
        
        else:
            # Need to compute augmentations
            print(f"\n✗ Pre-computed augmentations not found at {precomputed_file}")
            print("Computing augmentations (first-time setup, takes ~20-40 minutes)...\n")
            
            # Load all images into RAM with parallel processing
            print("Materializing dataset into RAM...")
            import time
            start_time = time.time()
            
            # Use map with num_proc to parallelize disk I/O
            dataset['train'] = dataset['train'].map(
                lambda x: x,  # Identity mapping
                batched=False,
                num_proc=8,  # Parallel processes for I/O
                desc="Loading images to RAM"
            )
            
            # Now actually collect into a list with progress bar
            images_in_ram = []
            for sample in tqdm(dataset['train'], desc="Collecting images to RAM", total=len(dataset['train'])):
                images_in_ram.append(sample['image'])
            
            ram_load_time = time.time() - start_time
            print(f"✓ Materialized {len(images_in_ram)} images to RAM in {ram_load_time/60:.1f} minutes\n")
            
            # Pre-compute augmentations
            print("Pre-computing augmentations for all images...")
            print("(This takes 10-20 minutes, but subsequent runs will be instant!)\n")
            
            augmentation = SimCLRAugmentation(image_size=96)
            
            precomputed_pairs = []
            for img in tqdm(images_in_ram, desc="Augmenting images"):
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img = img.convert('RGB')
                view1, view2 = augmentation(img)
                precomputed_pairs.append((view1, view2))
            
            print(f"✓ Pre-computed {len(precomputed_pairs)} augmented pairs\n")
            
            # Save to disk BEFORE training starts
            print(f"Saving pre-computed dataset to {precomputed_file}...")
            os.makedirs(precomputed_dir, exist_ok=True)
            torch.save(precomputed_pairs, precomputed_file)
            precomputed_size = os.path.getsize(precomputed_file) / 1e9
            print(f"✓ Saved! File size: {precomputed_size:.1f} GB\n")
            print("="*60)
            print("PRECOMPUTED DATASET IS SAFE!")
            print(f"Location: {precomputed_file}")
            print("="*60 + "\n")
    
        # Pre-computed dataset class
        class PrecomputedDataset(Dataset):
            def __init__(self, pairs):
                self.pairs = pairs
            
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                return self.pairs[idx]
        
        # Create dataset and dataloader
        train_dataset = PrecomputedDataset(precomputed_pairs)

        print("Creating DataLoader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False
        )
        print(f"Train DataLoader created. Total batches per epoch: {len(train_loader)}\n")

        # Create and train model
        print(f"Initializing SimCLR model with {args.arch}...")
        model = SimCLR(architecture=args.arch)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {num_params:,}\n")

        print(f"Starting training for {args.epochs} epochs...")
        train_SimCLR(
            model, 
            train_loader, 
            num_epochs=args.epochs, 
            lr=scaled_lr,
            warmup_epochs=args.warmup_epochs,
            save_dir=args.save_dir
        )

        print("\n" + "="*60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print(f"Precomputed dataset saved at: {precomputed_dir}")
        raise
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print(f"Precomputed dataset saved at: {precomputed_dir}")
        raise


if __name__ == '__main__':
    main()

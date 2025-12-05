import os
import argparse
import itertools
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from timm.models.vision_transformer import VisionTransformer

# ============================================================================
# 1. MODEL & BACKBONE
# ============================================================================

def interpolate_pos_encoding(x, pos_embed):
    npatch = x.shape[1] - 1 
    N = pos_embed.shape[1] - 1
    if npatch == N: return pos_embed
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w1 = int(math.sqrt(N)) 
    w_new = int(math.sqrt(npatch))
    h_new = w_new
    patch_pos_embed = patch_pos_embed.reshape(1, w0, w1, dim).permute(0, 3, 1, 2)
    patch_pos_embed = F.interpolate(patch_pos_embed, size=(w_new, h_new), mode='bicubic', align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

class CustomViT(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward_features(self, x):
        x = self.patch_embed(x)
        if x.dim() == 4 and x.shape[1] == self.embed_dim: x = x.flatten(2).transpose(1, 2)
        elif x.dim() == 4 and x.shape[-1] == self.embed_dim: x = x.flatten(1, 2)
        elif x.dim() == 3 and x.shape[2] == self.embed_dim: pass
        else: raise RuntimeError(f"Unexpected shape: {x.shape}")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = x + interpolate_pos_encoding(x, self.pos_embed)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

def load_backbone(checkpoint_path, arch='vit_base', device='cuda'):
    print(f"Loading checkpoint: {checkpoint_path}")
    if arch == 'vit_base': embed_dim, num_heads, depth = 768, 12, 12
    else: embed_dim, num_heads, depth = 384, 6, 12

    model = CustomViT(img_size=96, patch_size=8, embed_dim=embed_dim, depth=depth,
                      num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
                      norm_layer=nn.LayerNorm, num_classes=0, dynamic_img_size=True)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('student', ckpt.get('teacher', ckpt))
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("backbone."): new_state_dict[k.replace("backbone.", "")] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# ============================================================================
# 2. LINEAR PROBE (GPU)
# ============================================================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False) 
        self.linear = nn.Linear(input_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.bn(x)
        return self.linear(x)

# ============================================================================
# 3. DATA UTILS & PRECOMPUTE
# ============================================================================

class ImageDataset(Dataset):
    def __init__(self, root, filenames, labels=None, resolution=96):
        self.root = Path(root)
        self.files = filenames
        self.labels = labels
        self.resolution = resolution
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.root / self.files[idx]
        img = Image.open(path).convert('RGB')
        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        if self.labels is not None: return img, self.labels[idx], self.files[idx]
        return img, self.files[idx]

def collate_fn(batch):
    if len(batch[0]) == 3:
        imgs, lbls, names = zip(*batch)
        return list(imgs), list(lbls), list(names)
    else:
        imgs, names = zip(*batch)
        return list(imgs), list(names)

def precompute_features(backbone, loader, device, desc):
    backbone.eval()
    features, labels = [], []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Precomputing {desc}"):
            if len(batch) == 3: imgs_pil, lbls, _ = batch
            else: imgs_pil, _ = batch
            imgs = torch.stack([transform(img) for img in imgs_pil]).to(device)
            feats = backbone.forward_features(imgs)
            cls_feat = feats[:, 0]
            avg_feat = feats[:, 1:].mean(dim=1)
            combined = torch.cat((cls_feat, avg_feat), dim=1)
            features.append(combined.cpu())
            if len(batch) == 3: labels.extend(lbls)
            
    return torch.cat(features), torch.tensor(labels) if labels else None

# ============================================================================
# 4. TUNING LOGIC
# ============================================================================

def train_one_config(X_train, y_train, X_val, y_val, input_dim, num_classes, lr, wd, epochs, device):
    """Trains a single configuration and returns best val acc and model state."""
    probe = LinearProbe(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    
    # Train Loop (Silent, no tqdm to avoid log spam)
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Eval every 10 epochs or last epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            probe.eval()
            with torch.no_grad():
                val_out = probe(X_val)
                _, preds = torch.max(val_out, 1)
                acc = (preds == y_val).sum().item() / len(y_val)
                
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(probe.state_dict())
                    
    return best_acc, best_state

# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission_linear_tuned.csv")
    parser.add_argument("--arch", type=str, default="vit_base")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs_per_config", type=int, default=100)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    data_dir = Path(args.data_dir)
    
    # 1. Load Data Info
    train_df = pd.read_csv(data_dir / "train_labels.csv")
    val_df = pd.read_csv(data_dir / "val_labels.csv")
    num_classes = train_df['class_id'].nunique()
    
    if (data_dir / "test_images.csv").exists():
        test_df = pd.read_csv(data_dir / "test_images.csv")
        test_files = test_df['filename'].tolist()
    else:
        test_files = [f.name for f in sorted((data_dir / "test").glob("*")) if f.suffix in ['.jpg', '.png']]

    # 2. Datasets
    train_ds = ImageDataset(data_dir / "train", train_df['filename'].tolist(), train_df['class_id'].tolist())
    val_ds = ImageDataset(data_dir / "val", val_df['filename'].tolist(), val_df['class_id'].tolist())
    test_ds = ImageDataset(data_dir / "test", test_files)
    
    backbone = load_backbone(args.checkpoint, args.arch, device)
    
    # 3. Precompute Features (CPU -> GPU)
    # We load with num_workers on CPU, then move big tensors to GPU for blazing fast tuning
    loader_args = dict(batch_size=args.batch_size, num_workers=8, collate_fn=collate_fn)
    
    X_train, y_train = precompute_features(backbone, DataLoader(train_ds, **loader_args), device, "Train")
    X_val, y_val = precompute_features(backbone, DataLoader(val_ds, **loader_args), device, "Val")
    X_test, _ = precompute_features(backbone, DataLoader(test_ds, **loader_args), device, "Test")
    
    # Move to GPU
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test = X_test.to(device)
    input_dim = X_train.shape[1]

    # 4. HYPERPARAMETER TUNING GRID
    # We tune Learning Rate (speed) and Weight Decay (regularization)
    
    # 1. Define Ranges
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.1]
    weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    epochs = [50, 100, 200, 300, 500]
    
    # 2. Generate Cartesian Product (All Combinations)
    # This creates a flat list: [(1e-4, 0), (1e-4, 1e-5), ...]
    configs = list(itertools.product(learning_rates, weight_decays, epochs))
    
    print("\n" + "="*50)
    print(f"STARTING GRID SEARCH: {len(configs)} configurations")
    print("="*50)
    
    results = []
    
    # 3. Single Loop with Progress Bar
    # Now you get a progress bar for the Grid Search itself!
    for lr, wd, ep in tqdm(configs, desc="Tuning"):
        # Train
        acc, state = train_one_config(
            X_train, y_train, X_val, y_val, 
            input_dim, num_classes, 
            lr, wd, ep, device
        )
        
        # Log (using tqdm.write keeps the progress bar clean)
        # tqdm.write(f"LR: {lr:<7} | WD: {wd:<7} | Val Acc: {acc*100:.2f}%")
        
        results.append({
            'lr': lr, 'wd': wd, 'acc': acc, 'state': state
        })
            
    # 4. Select Winner
    results.sort(key=lambda x: x['acc'], reverse=True)
    winner = results[0]
    
    print("\n" + "-"*50)
    print(f"[WINNER] LR: {winner['lr']} | WD: {winner['wd']} | Acc: {winner['acc']*100:.2f}%")
    print("-"*50)

    # 6. Predict with Winner
    final_probe = LinearProbe(input_dim, num_classes).to(device)
    final_probe.load_state_dict(winner['state'])
    final_probe.eval()
    
    with torch.no_grad():
        test_out = final_probe(X_test)
        _, test_preds = torch.max(test_out, 1)
        
    # 7. Save
    df = pd.DataFrame({'id': test_files, 'class_id': test_preds.cpu().numpy()})
    
    # Validation
    assert list(df.columns) == ['id', 'class_id']
    assert df['class_id'].min() >= 0
    assert df['class_id'].max() < num_classes
    assert df.isnull().sum().sum() == 0
    
    df.to_csv(args.output, index=False)
    print(f"Saved submission to: {args.output}")

if __name__ == "__main__":
    main()
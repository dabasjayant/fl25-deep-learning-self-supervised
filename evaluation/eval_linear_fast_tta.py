import os
import argparse
import math
import copy
import itertools
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
# 2. LINEAR PROBE & DATA UTILS
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
    
    # Standard Transform
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # TTA: Horizontal Flip
    flip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {desc} (Spatial+TTA)"):
            if len(batch) == 3: imgs_pil, lbls, _ = batch
            else: imgs_pil, _ = batch
            
            # 1. Prepare Batch (Original + Flipped)
            imgs_orig = torch.stack([base_transform(img) for img in imgs_pil]).to(device)
            imgs_flip = torch.stack([flip_transform(img) for img in imgs_pil]).to(device)
            
            # 2. Forward Pass (Get all tokens, not just CLS)
            # We process original and flip separately to save memory, or cat them if VRAM allows
            # Let's do them sequentially to be safe
            
            def get_rich_features(images):
                # Returns: (Batch, 1 + N_Patches, Dim)
                # Note: We need the raw token sequence, so we might need to modify 
                # forward_features slightly or access the output before pooling
                
                # Assuming your CustomViT returns (Batch, 145, 768)
                # If your forward_features returns something else, we need to adapt.
                # Standard timm/ViT returns sequence.
                return backbone.forward_features(images)

            feat_orig = get_rich_features(imgs_orig) # (B, 145, 768)
            feat_flip = get_rich_features(imgs_flip)
            
            # 3. SPATIAL PYRAMID POOLING LOGIC
            def process_spatial(raw_feats):
                # raw_feats: (B, 145, 768)
                
                # A. CLS Token (Global Semantic)
                cls_token = raw_feats[:, 0] # (B, 768)
                
                # B. Patch Tokens
                patch_tokens = raw_feats[:, 1:] # (B, 144, 768)
                
                # Reshape to Grid (12x12 for 96px img / patch8)
                # grid_size = sqrt(144) = 12
                grid_size = int(math.sqrt(patch_tokens.shape[1]))
                spatial_grid = patch_tokens.permute(0, 2, 1).view(-1, 768, grid_size, grid_size)
                
                # C. Global Avg Pool
                global_avg = patch_tokens.mean(dim=1) # (B, 768)
                
                # D. Quadrant Pooling (Spatial Layout)
                # Split into 2x2 grid -> 4 regions of size 6x6
                half = grid_size // 2
                q1 = spatial_grid[:, :, :half, :half].mean(dim=(2,3)) # Top-Left
                q2 = spatial_grid[:, :, :half, half:].mean(dim=(2,3)) # Top-Right
                q3 = spatial_grid[:, :, half:, :half].mean(dim=(2,3)) # Bot-Left
                q4 = spatial_grid[:, :, half:, half:].mean(dim=(2,3)) # Bot-Right
                
                # Concatenate Everything
                # Dim: 768 * 6 = 4608 features per image
                return torch.cat([cls_token, global_avg, q1, q2, q3, q4], dim=1)

            # Process both views
            rich_orig = process_spatial(feat_orig)
            rich_flip = process_spatial(feat_flip)
            
            # 4. TTA Average
            final_feats = (rich_orig + rich_flip) / 2.0
            
            # L2 Normalize the final giant vector
            final_feats = F.normalize(final_feats, p=2, dim=1)
            
            features.append(final_feats.cpu())
            if len(batch) == 3: labels.extend(lbls)
            
    return torch.cat(features), torch.tensor(labels) if labels else None

# ============================================================================
# 3. TRAINING LOGIC (Strict Mini-Batch)
# ============================================================================

def train_one_config_fast(X_train, y_train, X_val, y_val, input_dim, num_classes, config, device, max_epochs=500, patience=20):
    lr = config['lr']
    wd = config['wd']
    opt_name = config['opt']
    batch_size = config['bs'] 
    
    probe = LinearProbe(input_dim, num_classes).to(device)
    
    # Label Smoothing: Helps model generalize better
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # --- OPTIMIZER SELECTION ---
    if opt_name == 'lbfgs':
        # L-BFGS requires Full Batch (usually) and strong history
        # We enforce batch_size='full' logic inside the loop later
        optimizer = torch.optim.LBFGS(probe.parameters(), lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        # L-BFGS generally doesn't use a scheduler in the same way, but we can keep it
        scheduler = None 
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif opt_name == 'sgd':
        # Added Nesterov=True
        optimizer = torch.optim.SGD(probe.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_acc = 0.0
    best_state = None
    patience_counter = 0
    n_samples = X_train.shape[0]

    # --- TRAINING LOOP ---
    for epoch in range(max_epochs):
        probe.train()
        
        # LOGIC SPLIT: L-BFGS vs Standard
        if opt_name == 'lbfgs':
            # L-BFGS Logic: Needs a Closure and Full Batch
            def closure():
                optimizer.zero_grad()
                output = probe(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                return loss
            
            # Performs multiple optimization steps internally
            optimizer.step(closure)
            
        else:
            # Standard Mini-Batch Logic (AdamW / SGD)
            indices = torch.randperm(n_samples, device=device)
            current_bs = n_samples if batch_size == 'full' else batch_size
            
            for start in range(0, n_samples, current_bs):
                end = start + current_bs
                batch_idx = indices[start:end]
                
                optimizer.zero_grad()
                outputs = probe(X_train[batch_idx])
                loss = criterion(outputs, y_train[batch_idx])
                loss.backward()
                optimizer.step()
            
            if scheduler:
                scheduler.step()

        # --- EVALUATION ---
        if (epoch + 1) % 5 == 0:
            probe.eval()
            with torch.no_grad():
                val_out = probe(X_val)
                _, preds = torch.max(val_out, 1)
                acc = (preds == y_val).sum().item() / len(y_val)
                
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(probe.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= (patience // 5):
                break
                    
    return best_acc, best_state, epoch + 1

# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission_linear_lbfgs.csv")
    parser.add_argument("--arch", type=str, default="vit_base")
    parser.add_argument("--batch_size", type=int, default=256) # For loading from disk
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    data_dir = Path(args.data_dir)
    
    # Load Data Info
    train_df = pd.read_csv(data_dir / "train_labels.csv")
    val_df = pd.read_csv(data_dir / "val_labels.csv")
    num_classes = train_df['class_id'].nunique()
    
    if (data_dir / "test_images.csv").exists():
        test_df = pd.read_csv(data_dir / "test_images.csv")
        test_files = test_df['filename'].tolist()
    else:
        test_files = [f.name for f in sorted((data_dir / "test").glob("*")) if f.suffix in ['.jpg', '.png']]

    train_ds = ImageDataset(data_dir / "train", train_df['filename'].tolist(), train_df['class_id'].tolist())
    val_ds = ImageDataset(data_dir / "val", val_df['filename'].tolist(), val_df['class_id'].tolist())
    test_ds = ImageDataset(data_dir / "test", test_files)
    
    backbone = load_backbone(args.checkpoint, args.arch, device)
    
    # Precompute Features
    loader_args = dict(batch_size=args.batch_size, num_workers=8, collate_fn=collate_fn)
    X_train, y_train = precompute_features(backbone, DataLoader(train_ds, **loader_args), device, "Train")
    X_val, y_val = precompute_features(backbone, DataLoader(val_ds, **loader_args), device, "Val")
    X_test, _ = precompute_features(backbone, DataLoader(test_ds, **loader_args), device, "Test")
    
    # Move to GPU
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test = X_test.to(device)
    input_dim = X_train.shape[1]

    # =========================================================
    # PARAMETER GRID WITH L-BFGS
    # =========================================================
    
    # 1. Standard Configs (Mini-Batch)
    param_space_standard = {
        'opt': ['adamw', 'sgd'],
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.1, 0.5],
        'wd': [1e-5, 1e-4, 1e-3, 1e-2],
        'bs': [64, 128, 256, 512, 1024]
    }
    
    # 2. L-BFGS Configs (Must be Full Batch, LR usually 1.0 or 0.1)
    param_space_lbfgs = {
        'opt': ['lbfgs'],
        'lr': [1.0, 0.5, 0.1, 0.05, 0.01], # LBFGS takes big steps
        'wd': [1e-5, 1e-4, 1e-3, 1e-2],
        'bs': ['full'] # Forced full batch
    }
    
    # Combine the grids
    keys_s, values_s = zip(*param_space_standard.items())
    configs_s = [dict(zip(keys_s, v)) for v in itertools.product(*values_s)]
    
    keys_l, values_l = zip(*param_space_lbfgs.items())
    configs_l = [dict(zip(keys_l, v)) for v in itertools.product(*values_l)]
    
    configs = configs_s + configs_l # Test everything
    
    print("\n" + "="*60)
    print(f"STARTING GRID SEARCH: {len(configs)} configurations")
    print("Including L-BFGS (The Convex Solver)!")
    print("="*60)
    
    results = []
    
    for config in tqdm(configs, desc="Tuning"):
        # Heuristics
        if config['opt'] == 'adamw' and config['lr'] > 0.01: continue
        if config['opt'] == 'sgd' and config['lr'] < 0.001: continue
        
        acc, state, stops_at = train_one_config_fast(
            X_train, y_train, X_val, y_val, 
            input_dim, num_classes, 
            config, device, 
            max_epochs=args.max_epochs, 
            patience=args.patience
        )
        
        config['acc'] = acc
        config['state'] = state
        config['epochs'] = stops_at
        results.append(config)
        
        # tqdm.write(f"Opt: {config['opt']:<6} | BS: {str(config['bs']):<5} | LR: {config['lr']:<6} | WD: {config['wd']:<6} | Acc: {acc*100:.2f}%")

    # Leaderboard
    results.sort(key=lambda x: x['acc'], reverse=True)
    winner = results[0]
    
    print("\n" + "-"*60)
    print("TOP 5 CONFIGURATIONS")
    print("-"*(60))
    for i, res in enumerate(results[:5]):
        print(f"Rank {i+1} | Acc: {res['acc']*100:.2f}% | {res['opt']} | BS: {res['bs']} | LR: {res['lr']} | WD: {res['wd']} | Epochs: {res['epochs']}")
    print("-"*(60))

    # Predict
    final_probe = LinearProbe(input_dim, num_classes).to(device)
    final_probe.load_state_dict(winner['state'])
    final_probe.eval()
    
    with torch.no_grad():
        test_out = final_probe(X_test)
        _, test_preds = torch.max(test_out, 1)
        
    df = pd.DataFrame({'id': test_files, 'class_id': test_preds.cpu().numpy()})
    
    # Validation Matches Sample Script Logic
    print(f"\nValidating submission format...")
    assert list(df.columns) == ['id', 'class_id']
    assert df['class_id'].min() >= 0
    assert df['class_id'].max() < num_classes
    assert df.isnull().sum().sum() == 0
    print("✓ Submission format is valid!")
    
    df.to_csv(args.output, index=False)
    print(f"Saved submission to: {args.output}")

if __name__ == "__main__":
    main()
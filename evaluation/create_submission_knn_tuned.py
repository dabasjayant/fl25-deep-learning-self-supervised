import os
import argparse
import math
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
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from timm.models.vision_transformer import VisionTransformer

# ============================================================================
# 1. MODEL DEFINITION (Exact copy of your Training Architecture)
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
        # Handle the 4D vs 3D tensor shape quirk from timm
        if x.dim() == 4 and x.shape[1] == self.embed_dim:
            x = x.flatten(2).transpose(1, 2)
        elif x.dim() == 4 and x.shape[-1] == self.embed_dim:
            x = x.flatten(1, 2)
        elif x.dim() == 3 and x.shape[2] == self.embed_dim:
            pass
        else:
            raise RuntimeError(f"Unexpected shape: {x.shape}")

        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = x + interpolate_pos_encoding(x, self.pos_embed)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

def load_backbone(checkpoint_path, arch='vit_base', device='cuda'):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Configure for ViT-Base (Default) or Small
    if arch == 'vit_base':
        embed_dim = 768
        num_heads = 12
        depth = 12
    else: # vit_small
        embed_dim = 384
        num_heads = 6
        depth = 12

    model = CustomViT(
        img_size=96,
        patch_size=8,        # <--- Always 8 for this project
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        num_classes=0,
        dynamic_img_size=True
    )
    
    # Load Weights
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different saving formats (Student/Teacher/Raw)
    if 'student' in ckpt:
        state_dict = ckpt['student']
    elif 'teacher' in ckpt:
        state_dict = ckpt['teacher']
    else:
        state_dict = ckpt

    # Clean Prefix (backbone. / module.backbone.)
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip "module." (DDP)
        k = k.replace("module.", "")
        # Strip "backbone." (MultiCropWrapper)
        if k.startswith("backbone."):
            new_state_dict[k.replace("backbone.", "")] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded. Missing keys (usually head/fc, which is fine): {len(msg.missing_keys)}")
    
    model.to(device)
    model.eval()
    return model

# ============================================================================
# 2. FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_features(self, images):
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            # Forward pass
            features = self.backbone.forward_features(batch)
            
            # Strategy: Concat CLS + AvgPool (Best for Accuracy)
            cls_feat = features[:, 0]
            patch_feat = features[:, 1:].mean(dim=1)
            combined = torch.cat((cls_feat, patch_feat), dim=1)
            
            # L2 Normalize (Crucial for k-NN)
            combined = F.normalize(combined, p=2, dim=1)
            
        return combined.cpu().numpy()

# ============================================================================
# 3. DATA
# ============================================================================

class ImageDataset(Dataset):
    def __init__(self, root, filenames, labels=None, resolution=96):
        self.root = Path(root)
        self.files = filenames
        self.labels = labels
        self.resolution = resolution

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.root / self.files[idx]
        
        # Open and Convert to RGB (Standardize channels)
        img = Image.open(path).convert('RGB')
        
        # Resize to (96, 96) dimensions
        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        
        if self.labels is not None:
            return img, self.labels[idx], self.files[idx]
        return img, self.files[idx]

def collate_fn(batch):
    if len(batch[0]) == 3:
        imgs, lbls, names = zip(*batch)
        return list(imgs), list(lbls), list(names)
    else:
        imgs, names = zip(*batch)
        return list(imgs), list(names)

def extract_all(extractor, loader, desc):
    feats, lbls, names = [], [], []
    for batch in tqdm(loader, desc=desc):
        if len(batch) == 3:
            x, y, n = batch
            lbls.extend(y)
        else:
            x, n = batch
        
        f = extractor.get_features(x)
        feats.append(f)
        names.extend(n)
    
    return np.concatenate(feats, axis=0), np.array(lbls) if lbls else None, names

# ============================================================================
# 4. TUNING LOGIC
# ============================================================================

def tune_and_predict(X_train, y_train, X_val, y_val, X_test, test_names, output_file):
    print("\n" + "="*40)
    print("HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*40)
    
    # 1. Define the Parameter Grid
    # We test k, weights, and distance metric (p=1 vs p=2)
    param_grid = {
        'n_neighbors': [5, 10, 20, 30, 40, 50, 60, 70, 100, 150, 200],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1=Manhattan, 2=Euclidean (Cosine-like)
    }
    
    # 2. Setup Predefined Split (Train on X_train, Validate on X_val)
    # -1 means "training fold", 0 means "validation fold"
    train_indices = np.full((X_train.shape[0],), -1, dtype=int)
    val_indices = np.full((X_val.shape[0],), 0, dtype=int)
    test_fold = np.concatenate((train_indices, val_indices))
    
    ps = PredefinedSplit(test_fold)
    
    # Merge Data
    X_combined = np.concatenate((X_train, X_val), axis=0)
    y_combined = np.concatenate((y_train, y_val), axis=0)
    
    # 3. Run Grid Search
    knn = KNeighborsClassifier()
    grid = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=ps,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1 
    )
    
    print(f"Running Grid Search on {len(y_combined)} samples...")
    grid.fit(X_combined, y_combined)
    
    # =========================================================
    # NEW: PRINT DETAILED RESULTS
    # =========================================================
    print("\n" + "-"*60)
    print("TUNING LEADERBOARD")
    print("-"*(60))
    
    # Convert results to DataFrame for easy sorting/display
    results_df = pd.DataFrame(grid.cv_results_)
    
    # Keep only relevant columns and sort by score (descending)
    results_df = results_df[['params', 'mean_test_score', 'rank_test_score']]
    results_df = results_df.sort_values(by='rank_test_score')
    
    # Print top 15 results
    for i, row in results_df.head(15).iterrows():
        acc = row['mean_test_score'] * 100
        rank = row['rank_test_score']
        params = row['params']
        # Format: Rank 1 | 45.20% | {'n_neighbors': 20, 'p': 2, 'weights': 'distance'}
        print(f"Rank {rank:<2} | Acc: {acc:.2f}% | {params}")

    print("-"*(60))
    print(f"\n[WINNER] Best Accuracy: {grid.best_score_*100:.2f}%")
    print(f"         Best Params:   {grid.best_params_}")
    
    # 4. Generate Predictions
    print("\nGenerating Test Predictions with Best Model...")
    best_clf = grid.best_estimator_
    test_preds = best_clf.predict(X_test)
    
    df = pd.DataFrame({'id': test_names, 'class_id': test_preds})

    print(f"\nValidating submission format...")
    assert list(df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert df['class_id'].max() < len(test_names), f"Invalid class_id > {len(test_names)-1}"
    assert df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")

    df.to_csv(output_file, index=False)
    print(f"Saved submission to: {output_file}")

# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--arch", type=str, default="vit_base", choices=["vit_small", "vit_base"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device} | Checkpoint: {args.checkpoint}")
    
    # 1. Load DataFrames
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train_labels.csv")
    val_df = pd.read_csv(data_dir / "val_labels.csv")
    
    # Robust Test Loading
    if (data_dir / "test_images.csv").exists():
        test_df = pd.read_csv(data_dir / "test_images.csv")
        test_files = test_df['filename'].tolist()
    else:
        test_files = [f.name for f in sorted((data_dir / "test").glob("*")) if f.suffix in ['.jpg', '.png']]

    # 2. Datasets
    train_ds = ImageDataset(data_dir / "train", train_df['filename'].tolist(), train_df['class_id'].tolist())
    val_ds = ImageDataset(data_dir / "val", val_df['filename'].tolist(), val_df['class_id'].tolist())
    test_ds = ImageDataset(data_dir / "test", test_files)
    
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=False, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    
    # 3. Model
    backbone = load_backbone(args.checkpoint, args.arch, device)
    extractor = FeatureExtractor(backbone, device)
    
    # 4. Extract
    print("Extracting features (this allows fast tuning)...")
    X_train, y_train, _ = extract_all(extractor, train_loader, "Train")
    X_val, y_val, _ = extract_all(extractor, val_loader, "Val")
    X_test, _, test_names = extract_all(extractor, test_loader, "Test")
    
    # 5. Tune & Predict
    tune_and_predict(X_train, y_train, X_val, y_val, X_test, test_names, args.output)

if __name__ == "__main__":
    main()
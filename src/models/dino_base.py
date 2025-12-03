import os
import glob
import math
import time
import argparse
import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

import timm
from timm.models.vision_transformer import VisionTransformer

def get_args():
    parser = argparse.ArgumentParser(description="Train DINOv2-style SSL on Custom Data")
    
    # --- Critical System Params ---
    parser.add_argument("--data_path", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_ssl", help="Where to save weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., './checkpoints_ssl/checkpoint_epoch_010.pth')")

    # --- External Data Argument ---
    parser.add_argument("--external_data", type=str, nargs='*', default=None, help="List of additional image folders (e.g., --external_data ./inat ./places)")

    # --- System Args ---
    parser.add_argument("--workers", type=int, default=12, help="Number of data loading workers (default: 12 for 14-core CPU)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size (reduce to 512 if OOM)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    # Model/Optimization Params (defaults tuned for RTX 8000 + 96px images)
    parser.add_argument("--lr", type=float, default=0.002, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.04, help="Initial weight decay")
    parser.add_argument("--image_size", type=int, default=96, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=8, help="ViT Patch size (8 for small images, 16 for large)")
    parser.add_argument("--save_freq", type=int, default=1, help="Save checkpoint every N epochs")

    # Teacher temperature schedule
    parser.add_argument("--teacher_temp", type=float, default=0.05, help="Teacher temperature (final value)")
    parser.add_argument("--warmup_teacher_temp", type=float, default=0.05, help="Initial teacher temperature for warmup")
    parser.add_argument("--warmup_teacher_temp_epochs", type=int, default=0, help="Number of epochs for teacher temperature warmup")

    # --- Gradient Accumulation (Crucial for ViT-Base) ---
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps to simulate larger batch")
    
    return parser.parse_args()

# =============================================================================
# 1. CONFIGURATION & LOGGING
# =============================================================================

@dataclass
class SSLConfig:
    # --- Data ---
    output_dir: str
    external_data: List[str] 
    data_path: str = "./dataset"
    image_size: int = 96
    
    # --- System ---
    num_workers: int = 12
    batch_size: int = 512
    epochs: int = 800

    # --- Optimization ---
    lr: float = 0.0005
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    warmup_epochs: int = 10
    clip_grad: float = 3.0

    patch_size: int = 8
    grad_accum_steps: int = 4  # To simulate larger batch sizes: 512 * 4 = 2048
    
    # Defaults for ViT-Base
    arch: str = "vit_base" 
    embed_dim: int = 768
    out_dim: int = 65536      
    norm_last_layer: bool = True
    momentum_teacher: float = 0.995
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    clip_grad: float = 3.0
    
    # Multi-Crop Config (Optimized)
    local_crops_number: int = 8
    global_crops_scale: tuple = (0.4, 1.0)
    local_crops_scale: tuple = (0.2, 0.5)
    local_crops_size: int = 48


    # --- Checkpointing ---
    output_dir: str = "./checkpoints_ssl"
    keep_last_k: int = 10
    save_freq: int = 1


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger(__name__)

# =============================================================================
# 2. DATA AUGMENTATION (DINO RECIPE)
# =============================================================================

class DataAugmentationDINO:
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size):
        # 1. Standard Color Jitter
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # 2. Normalize (Standard ImageNet)
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # ================= GLOBAL CROPS (TEACHER) =================
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Reduced kernel
            normalize,
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Reduced kernel
            transforms.RandomSolarize(threshold=128, p=0.2),
            normalize,
        ])
        
        # ================= LOCAL CROPS (STUDENT) =================
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomSolarize(threshold=128, p=0.1), # Gentle solarization
            normalize,
        ])

        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
        # Filter for valid image extensions if necessary
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
# =============================================================================
# 2.2 MIXED DATASET LOADING FROM MULTIPLE SOURCES
# =============================================================================

def get_mixed_dataset(original_path, external_paths_list, transform=None):
    """
    original_path: Path to the contest provided 500k images
    external_paths_list: List of paths to iNat, OpenImages, etc.
    """
    datasets = []
    
    # 1. Original Data
    print("Loading Original Contest Data...")
    datasets.append(ImageFolderDataset(original_path, transform))
    
    # 2. External Data
    for ext_path in external_paths_list:
        if os.path.exists(ext_path):
            print(f"Loading External Data: {ext_path}")
            datasets.append(ImageFolderDataset(ext_path, transform))
        else:
            print(f"Warning: External path {ext_path} not found. Skipping.")
            
    # 3. Concatenate
    # This creates a virtual dataset that acts as one giant list
    mixed_dataset = ConcatDataset(datasets)
    print(f"Total Training Images: {len(mixed_dataset)}")
    
    return mixed_dataset

# =============================================================================
# 3. MODEL ARCHITECTURE (Explicit Shape Handling)
# =============================================================================

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
            
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

def interpolate_pos_encoding(x, pos_embed):
    """
    x shape: (Batch, N_patches, Dim)
    pos_embed shape: (1, N_original, Dim)
    """
    npatch = x.shape[1] - 1 
    N = pos_embed.shape[1] - 1
    
    if npatch == N:
        return pos_embed
        
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    
    dim = x.shape[-1]
    
    # Original Grid (e.g. 12x12)
    w0 = w1 = int(math.sqrt(N)) 
    
    # New Grid (e.g. 4x4)
    w_new = int(math.sqrt(npatch))
    h_new = w_new
    
    # Reshape and Interpolate
    # (1, N, C) -> (1, Grid, Grid, C) -> (1, C, Grid, Grid)
    patch_pos_embed = patch_pos_embed.reshape(1, w0, w1, dim).permute(0, 3, 1, 2)
    
    patch_pos_embed = F.interpolate(
        patch_pos_embed, 
        size=(w_new, h_new), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Flatten back: (1, C, Grid, Grid) -> (1, Grid, Grid, C) -> (1, N, C)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

class CustomViT(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward_features(self, x):
        # 1. Embed Patches
        x = self.patch_embed(x)
        
        # 2. FORCE CORRECT SHAPE
        # The goal is always (Batch, N_patches, Embed_Dim)
        
        # Case A: (Batch, Embed_Dim, H, W) -> Standard PyTorch / Timm
        # e.g., (512, 384, 12, 12)
        if x.dim() == 4 and x.shape[1] == self.embed_dim:
            x = x.flatten(2)       # -> (512, 384, 144)
            x = x.transpose(1, 2)  # -> (512, 144, 384)
            
        # Case B: (Batch, H, W, Embed_Dim) -> YOUR ERROR CASE (Channels Last)
        # e.g., (512, 12, 12, 384)
        elif x.dim() == 4 and x.shape[-1] == self.embed_dim:
            x = x.flatten(1, 2)    # Flatten H and W -> (512, 144, 384)

        # Case C: (Batch, N, Embed_Dim) -> Already Correct
        # e.g., (512, 144, 384)
        elif x.dim() == 3 and x.shape[2] == self.embed_dim:
            pass
            
        else:
            # If we get here, the shape is truly bizarre
            raise RuntimeError(f"Unknown tensor shape: {x.shape}. Expected dims to include {self.embed_dim}.")

        # 3. Add CLS Token
        # Now x is guaranteed to be (Batch, N, 384)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        
        # 4. Add Positional Embedding (Interpolated)
        x = x + interpolate_pos_encoding(x, self.pos_embed)
        
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple): features = features[0]
        cls_token = features[:, 0]
        return self.head(cls_token)

def get_model(cfg):
    if cfg.arch == "vit_base":
        embed_dim = 768
        depth = 12
        num_heads = 12
        drop_path_rate = 0.1 
    else: # vit_small
        embed_dim = 384
        depth = 12
        num_heads = 6
        drop_path_rate = 0.0

    backbone = CustomViT(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,   # Still 8!
        embed_dim=embed_dim,         # 768 for Base
        depth=depth,                 # 12
        num_heads=num_heads,         # 12
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=drop_path_rate, 
        norm_layer=nn.LayerNorm,
        num_classes=0,
        dynamic_img_size=True 
    )
    
    for p in backbone.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    head = DINOHead(
        in_dim=cfg.embed_dim,
        out_dim=cfg.out_dim,
        norm_last_layer=cfg.norm_last_layer
    )
    
    return MultiCropWrapper(backbone, head)

# =============================================================================
# 4. LOSS FUNCTION
# =============================================================================

class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Teacher temperature schedule
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def koleo_loss(self, student_output):
        """
        DINOv2 Regularization: Forces uniform feature distribution.
        """
        # 1. Normalize
        x = F.normalize(student_output, dim=-1, p=2)
        
        # 2. Pairwise distances
        pdist = torch.cdist(x, x, p=2)
        
        # 3. Ignore self-distance (diagonal)
        pdist.fill_diagonal_(float('inf'))
        
        # 4. Find nearest neighbor distance
        d_min, _ = pdist.min(dim=1)
        
        # 5. Maximize distance (minimize negative log)
        return -torch.log(d_min + 1e-6).mean()

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-Entropy between Teacher and Student + KoLeo Regularization
        """
        # 1. Prepare Student/Teacher Logic
        student_out = student_output / self.student_temp
        
        # Robust Chunking (Fix for Batch Size Mismatch)
        # Teacher always has 2 global crops. 
        batch_size = teacher_output.shape[0] // 2
        n_crops = student_output.shape[0] // batch_size
        
        student_out_chunked = student_out.chunk(n_crops)

        # Teacher centering & sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out_chunked = teacher_out.detach().chunk(2)

        # 2. Standard DINO Loss
        dino_loss_val = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out_chunked):
            for v in range(len(student_out_chunked)):
                if v == iq: continue # Skip same view
                loss = torch.sum(-q * F.log_softmax(student_out_chunked[v], dim=-1), dim=-1)
                dino_loss_val += loss.mean()
                n_loss_terms += 1
        dino_loss_val /= n_loss_terms
        
        # 3. KoLeo Loss (DINOv2)
        # Apply to all student outputs to force uniform spread
        koleo = self.koleo_loss(student_output)
        
        # 4. Update Center (Multi-GPU Safe)
        self.update_center(teacher_output)
        
        # Weighted Sum (0.1 is standard from DINOv2 paper)
        return dino_loss_val + (0.1 * koleo)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output centering.
        Multi-GPU Safe: Uses all_reduce to average across GPUs.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        
        # Multi-GPU Sync
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            len_total = len(teacher_output) * dist.get_world_size()
        else:
            len_total = len(teacher_output)
            
        batch_center = batch_center / len_total
        
        # EMA Update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# =============================================================================
# 5. TRAINING UTILS
# =============================================================================

class CheckpointManager:
    def __init__(self, directory, keep_last_k=5):
        self.directory = directory
        self.keep_last_k = keep_last_k
        os.makedirs(directory, exist_ok=True)
        self.checkpoints = deque() # Stores (epoch, path)

        self.best_metric = None

    def save(self, model, teacher, optimizer, epoch, metrics):
        filename = f"checkpoint_epoch_{epoch:03d}.pth"
        path = os.path.join(self.directory, filename)
        
        # Prepare state dict (unwrap from accelerator/DDP if needed externally, 
        # but here we pass raw model references usually)
        save_dict = {
            'student': model.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        torch.save(save_dict, path)

        if metrics and ('loss' in metrics):
            current_loss = metrics['loss']
            if (self.best_metric is None) or (current_loss < self.best_metric):
                self.best_metric = current_loss
                best_path = os.path.join(self.directory, "best_checkpoint.pth")
                torch.save(save_dict, best_path)
                logger.info(f"New best checkpoint saved at epoch {epoch} with loss {current_loss:.4f}")
        
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.keep_last_k:
            oldest = self.checkpoints.popleft()
            if os.path.exists(oldest):
                os.remove(oldest)
        
        logger.info(f"Saved checkpoint: {path}")

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = torch.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep)
    iters = torch.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + torch.cos(math.pi * iters / len(iters)))
    schedule = torch.cat((warmup_schedule, schedule))
    return schedule

def load_state_dict_robust(model, state_dict):
    """
    Robustly loads a state_dict into a model, handling 'module.' prefix mismatches
    caused by compiling (torch.compile), DDP, or Accelerator wrapping.
    """
    model_dict = model.state_dict()
    new_state_dict = {}
    
    # specific fix for DINO head which might have different keys if not careful
    for k, v in state_dict.items():
        # 1. Handle 'module.' prefix (common in DDP/Accelerate)
        if k.startswith("module.") and not "module." in list(model_dict.keys())[0]:
            name = k[7:] # remove 'module.'
        elif not k.startswith("module.") and "module." in list(model_dict.keys())[0]:
            name = "module." + k # add 'module.'
        else:
            name = k
            
        if name in model_dict:
            # Check shape mismatch (e.g. if you changed image size or patch size)
            if v.shape == model_dict[name].shape:
                new_state_dict[name] = v
            else:
                print(f"[WARNING] Skipping {name}: shape mismatch {v.shape} vs {model_dict[name].shape}")
        else:
            # print(f"[WARNING] Key {name} not found in model.")
            pass
            
    # Load the processed dict
    msg = model.load_state_dict(new_state_dict, strict=False)
    return msg

# =============================================================================
# 6. MAIN TRAINING LOOP
# =============================================================================

def train_one_epoch(student, teacher, teacher_without_ddp, loss_fn, data_loader, optimizer, 
                   lr_schedule, wd_schedule, momentum_schedule, epoch, accelerator, args):
    
    student.train()
    teacher.eval()
    
    metric_logger = {'loss': 0.0}
    
    # Use accelerator to handle the progress bar correctly
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not accelerator.is_local_main_process)
    
    for it, images in enumerate(pbar):
        # Update LR and WD
        cur_iter = epoch * len(data_loader) + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[cur_iter]
            if i == 0:  
                param_group["weight_decay"] = wd_schedule[cur_iter]

        # Images are on the correct device automatically via Accelerate's dataloader
        
        # Split crops
        global_crops = images[:2]
        local_crops = images[2:]
        
        # --- FORWARD PASS ---
        with torch.no_grad():
            teacher_input = torch.cat(global_crops, dim=0)
            teacher_output = teacher(teacher_input)

        student_input_global = torch.cat(global_crops, dim=0)
        student_input_local = torch.cat(local_crops, dim=0)
        
        student_output_global = student(student_input_global)
        student_output_local = student(student_input_local)
        student_output = torch.cat([student_output_global, student_output_local], dim=0)

        # --- LOSS ---
        loss = loss_fn(student_output, teacher_output, epoch)

        # Scale loss
        loss = loss / args.gradient_accumulation_steps 
        accelerator.backward(loss)

        # Step only after N batches
        if (it + 1) % args.gradient_accumulation_steps == 0:
            # --- BACKWARD ---
            optimizer.zero_grad()
            accelerator.backward(loss)
            
            if args.clip_grad:
                accelerator.clip_grad_norm_(student.parameters(), args.clip_grad)
                
            optimizer.step()

            # --- EMA UPDATE (THE FIX) ---
            with torch.no_grad():
                m = momentum_schedule[cur_iter]
                
                # FIX: Handle Single-GPU (no .module) vs Multi-GPU (has .module)
                student_params = student.module.parameters() if hasattr(student, "module") else student.parameters()
                
                for param_q, param_k in zip(student_params, teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # --- LOGGING ---
        loss_val = loss.item()
        metric_logger['loss'] += loss_val
        pbar.set_postfix({'loss': f"{loss_val:.4f}", 'lr': f"{lr_schedule[cur_iter]:.6f}"})

    return metric_logger['loss'] / len(data_loader)

def main():
    args = get_args()
    
    # Map CLI args to Config
    cfg = SSLConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        external_data=args.external_data if args.external_data else [],
        num_workers=args.workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        patch_size=args.patch_size,
        save_freq=args.save_freq
    )
    
    # Initialize Accelerator
    # Handles Mixed Precision (fp16) and Device placement automatically
    accelerator = Accelerator(mixed_precision="fp16") # or "bf16" if supported
    if accelerator.is_main_process:
        logger.info(f"System: {accelerator.device} | Workers: {cfg.num_workers} | Batch: {cfg.batch_size}")
        logger.info(f"Data: {cfg.data_path}")
    
    logger.info(f"Starting SSL training on {accelerator.device}")
    
    # Data Setup
    transform = DataAugmentationDINO(
        cfg.global_crops_scale,
        cfg.local_crops_scale,
        cfg.local_crops_number,
        cfg.local_crops_size
    )
    dataset = get_mixed_dataset(cfg.data_path, cfg.external_data, transform)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Model Setup
    student = get_model(cfg)
    teacher = get_model(cfg)
    
    # Move to device immediately to ensure weights are same
    student = student.to(accelerator.device)
    teacher = teacher.to(accelerator.device)
    
    # Teacher needs no gradients
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Sync weights initially
    teacher.load_state_dict(student.state_dict())
    
    # Optimizer (Standard ViT: separate weight decay for biases/norms)
    param_groups = [
        {'params': [p for n, p in student.named_parameters() if ('bias' not in n and 'norm' not in n)], 'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in student.named_parameters() if ('bias' in n or 'norm' in n)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    
    # Prepare with Accelerate
    # NOTE: We do NOT prepare the teacher. We manage teacher manually via EMA.
    student, optimizer, data_loader = accelerator.prepare(student, optimizer, data_loader)
    
    # Loss
    dino_loss = DINOLoss(
        cfg.out_dim, 
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=cfg.epochs
    ).to(accelerator.device)

    # --- RESUME LOGIC ---
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"==> Resuming from checkpoint: {args.resume}")
            # Map location is crucial to avoid OOM by loading everything to CPU first
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # 1. Load Epoch
            start_epoch = checkpoint['epoch'] + 1
            
            # 2. Load Models (Robustly)
            # Student is wrapped by Accelerate, Teacher is not.
            load_state_dict_robust(student, checkpoint['student'])
            load_state_dict_robust(teacher, checkpoint['teacher'])
            
            # 3. Load Optimizer
            # Accelerate handles optimizer mapping, but we must load the state dict
            # Standard PyTorch load works fine here because we are post-prepare
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                logger.warning(f"Optimizer load failed (likely device mismatch or param change). Skipping optimizer resume. Error: {e}")
            
            logger.info(f"==> Resumed successfully! Starting from Epoch {start_epoch}")
        else:
            logger.warning(f"==> Resume file not found at {args.resume}. Starting from scratch.")
    
    # Schedulers
    n_iter = len(data_loader)
    lr_schedule = cosine_scheduler(cfg.lr * (cfg.batch_size / 256.), cfg.min_lr, cfg.epochs, n_iter, warmup_epochs=cfg.warmup_epochs)
    wd_schedule = cosine_scheduler(cfg.weight_decay, cfg.weight_decay_end, cfg.epochs, n_iter)
    momentum_schedule = cosine_scheduler(cfg.momentum_teacher, 1.0, cfg.epochs, n_iter)
    
    # Checkpointing
    saver = CheckpointManager(cfg.output_dir, keep_last_k=cfg.keep_last_k)
    
    logger.info("Starting training loop...")
    
    for epoch in range(start_epoch, cfg.epochs):
        avg_loss = train_one_epoch(
            student=student,
            teacher=teacher,
            teacher_without_ddp=teacher, # Since no DDP, same obj
            loss_fn=dino_loss,
            data_loader=data_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            epoch=epoch,
            accelerator=accelerator,
            args=cfg
        )
        
        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            saver.save(student, teacher, optimizer, epoch, {'loss': avg_loss})

if __name__ == "__main__":
    main()
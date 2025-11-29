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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

import timm
from timm.models.vision_transformer import VisionTransformer

def get_args():
    parser = argparse.ArgumentParser(description="Train DINOv2-style SSL on Custom Data")
    
    # Critical System Params
    parser.add_argument("--data_path", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_ssl", help="Where to save weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., './checkpoints_ssl/checkpoint_epoch_010.pth')")
    parser.add_argument("--workers", type=int, default=12, help="Number of data loading workers (default: 12 for 14-core CPU)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size (reduce to 512 if OOM)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    # Model/Optimization Params (defaults tuned for RTX 8000 + 96px images)
    parser.add_argument("--lr", type=float, default=0.002, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.04, help="Initial weight decay")
    parser.add_argument("--image_size", type=int, default=96, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=8, help="ViT Patch size (8 for small images, 16 for large)")
    parser.add_argument("--save_freq", type=int, default=1, help="Save checkpoint every N epochs")
    
    return parser.parse_args()

# =============================================================================
# 1. CONFIGURATION & LOGGING
# =============================================================================

@dataclass
class SSLConfig:
    # --- Data & System ---
    data_path: str = "./data/unlabeled_images"
    image_size: int = 96
    
    # RTX 8000 (48GB) can handle massive batches with ViT-S/8. 
    # Larger batch = more negative samples (implicit) = better gradients.
    batch_size: int = 1024      
    num_workers: int = 12       # High throughput needed for small images
    
    # --- Model (ViT-Small/8) ---
    arch: str = "vit_small"
    patch_size: int = 8         # CRITICAL: 8x8 patches for 96px input (144 tokens)
    embed_dim: int = 384
    
    # Reduced from 65536. 
    # For 500k images, 65k prototypes is overkill and hard to learn. 
    # 16k is denser and stabilizes convergence on smaller datasets.
    out_dim: int = 16384        
    norm_last_layer: bool = True
    
    # --- Optimization (The "Recipe") ---
    epochs: int = 100           # 100 is enough for 500k images with Multi-Crop
    
    # Learning Rate:
    # Rule of thumb: 0.0005 * (BatchSize / 256). 
    # For BS=1024, this is 0.002.
    lr: float = 0.002           
    min_lr: float = 1e-6
    
    # Weight Decay:
    # We start lower (0.04) to let features grow, then tighten (0.4) 
    # to compact the hypersphere.
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    
    warmup_epochs: int = 10
    clip_grad: float = 3.0
    
    # --- Teacher Momentum (Stability) ---
    # Start higher (0.996) because small batches/images are noisier.
    # We want the teacher to be a "stable anchor".
    momentum_teacher: float = 0.996 
    
    # --- Extreme Multi-Crop (The Accuracy Booster) ---
    # seeing 2 global + 10 local = 12 views per image per step.
    # 1024 batch * 12 views = 12,288 effective examples per step.
    local_crops_number: int = 10
    
    # Crops for 96px:
    # Global: 40% to 100% of image
    # Local:  15% to 40% of image (small object parts)
    global_crops_scale: tuple = (0.4, 1.0)
    local_crops_scale: tuple = (0.15, 0.4)
    local_crops_size: int = 32
    
    # --- Checkpointing ---
    output_dir: str = "./checkpoints_ssl"
    keep_last_k: int = 5
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
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # Global transformation 1
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)), # Reduced kernel for 96px
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Global transformation 2 (Solarization added)
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Local transformation
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), # Smaller blur for small crops
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
# 3. MODEL ARCHITECTURE (Fixed for Multi-Crop & timm dynamic size)
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
    Interpolate pos_encoding from the existing pos_embed to the resolution of x.
    This is critical for handling 32x32 local crops when model is trained on 96x96.
    """
    # x shape here is (Batch, N_tokens, Dim) containing CLS token
    npatch = x.shape[1] - 1 
    N = pos_embed.shape[1] - 1
    
    if npatch == N:
        return pos_embed
        
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    
    dim = x.shape[-1]
    
    # Calculate original grid size (e.g., 12x12 for 96px image with patch 8)
    w0 = w1 = int(math.sqrt(N)) 
    
    # Calculate new grid size (e.g., 4x4 for 32px image with patch 8)
    w_new = int(math.sqrt(npatch))
    h_new = w_new
    
    # Reshape to (1, Dim, Grid, Grid) for F.interpolate
    patch_pos_embed = patch_pos_embed.reshape(1, w0, w1, dim).permute(0, 3, 1, 2)
    
    # Interpolate
    patch_pos_embed = F.interpolate(
        patch_pos_embed, 
        size=(w_new, h_new), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Flatten back to (1, N_new, Dim)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

class CustomViT(VisionTransformer):
    """
    Subclass of timm VisionTransformer that handles Multi-Crop positional interpolation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward_features(self, x):
        # 1. Embed patches
        x = self.patch_embed(x)
        
        # --- FIX START ---
        # When dynamic_img_size=True, timm returns (B, C, H, W).
        # We need (B, N, C).
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        # --- FIX END ---

        # 2. Add CLS token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        
        # 3. Add Positional Embedding (INTERPOLATED)
        x = x + interpolate_pos_encoding(x, self.pos_embed)
        
        x = self.pos_drop(x)
        
        # 4. Run blocks
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
    backbone = CustomViT(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        num_classes=0,
        dynamic_img_size=True  # Keeps assert from failing on 32px images
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

    def forward(self, student_output, teacher_output, epoch):
        """
        student_output: [n_crops * batch_size, dim]
        teacher_output: [2 * batch_size, dim]  (Teacher only processes global crops)
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(student_output.shape[0] // teacher_output.shape[0])

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2) # 2 global crops

        total_loss = 0
        n_loss_terms = 0
        
        # Cross-entropy between every student view and every teacher view
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: 
                    # Skip calculating loss where student and teacher view are the same original image crop
                    # (Standard DINO skips this, though some implementations keep it)
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # Distributed averaging would go here if using DDP across multiple nodes
        batch_center = batch_center / len(teacher_output)
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
    # Teacher is always in eval mode
    teacher.eval()
    
    metric_logger = {'loss': 0.0}
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not accelerator.is_local_main_process)
    
    for it, images in enumerate(pbar):
        # Update LR and WD
        cur_iter = epoch * len(data_loader) + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[cur_iter]
            if i == 0:  # only the first group is regularized usually
                param_group["weight_decay"] = wd_schedule[cur_iter]

        # Move images to device
        # Images is a list of crops. Structure: [Global1, Global2, Local1, ..., LocalN]
        # We concatenate them differently for Student and Teacher
        # Teacher sees only global views (first 2)
        # Student sees ALL views
        
        # list of tensors -> single tensor for efficiency? 
        # No, because resolutions differ. 
        # Global crops are 96x96, Local are 32x32. We must batch separately.
        
        global_crops = images[:2]
        local_crops = images[2:]
        
        # Multi-resolution forward pass strategy:
        # 1. Pass global crops through Teacher
        # 2. Pass global AND local crops through Student
        
        with torch.no_grad():
            # Teacher Forward
            # Stack the 2 global crops: [2*B, 3, 96, 96]
            teacher_input = torch.cat(global_crops, dim=0)
            teacher_output = teacher(teacher_input)

        # Student Forward
        # We cannot stack all crops simply because dimensions differ.
        # We forward global crops first, then local crops.
        student_input_global = torch.cat(global_crops, dim=0)
        student_input_local = torch.cat(local_crops, dim=0)
        
        # Use Accelerate's autocast context implicitly via backward()
        # But we need to handle forward. Accelerate handles device placement, 
        # but typically we just run model(input).
        
        # Student Output: [ (2 + n_local) * B, dim ]
        student_output_global = student(student_input_global)
        student_output_local = student(student_input_local)
        student_output = torch.cat([student_output_global, student_output_local], dim=0)

        # Calculate Loss
        loss = loss_fn(student_output, teacher_output, epoch)

        # Optimization
        optimizer.zero_grad()
        accelerator.backward(loss)
        
        if args.clip_grad:
            accelerator.clip_grad_norm_(student.parameters(), args.clip_grad)
            
        optimizer.step()

        # EMA Update of Teacher
        with torch.no_grad():
            m = momentum_schedule[cur_iter]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
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
    dataset = ImageFolderDataset(cfg.data_path, transform=transform)
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
        warmup_teacher_temp=0.04, 
        teacher_temp=0.04, 
        warmup_teacher_temp_epochs=0, # Typically 0 or small for DINO
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
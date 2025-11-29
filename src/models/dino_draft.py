# ssl_dino/config.py
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DINOConfig:
    # Data
    data_dir: str = '/path/to/unlabeled_images'
    image_size: int = 96
    global_crops_scale: Tuple[float, float] = (0.4, 1.0)
    local_crops_scale: Tuple[float, float] = (0.05, 0.4)
    global_crops_number: int = 2
    local_crops_number: int = 2  # fewer local crops for speed

    # Model (ViT-S-mini/8)
    patch_size: int = 8
    embed_dim: int = 320
    depth: int = 9
    num_heads: int = 5
    mlp_ratio: float = 4.0
    out_dim: int = 256  # projection dim
    use_prediction_head: bool = True

    # Training
    epochs: int = 100
    batch_size_per_gpu: int = 32
    num_workers: int = 4
    base_lr: float = 1.5e-4
    warmup_epochs: int = 10
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad_norm: float = 3.0

    # DINO specific
    teacher_momentum_base: float = 0.996
    teacher_momentum_end: float = 0.9995
    teacher_temp_base: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_epochs: int = 30
    student_temp: float = 0.1
    center_momentum: float = 0.9

    # Performance / infra
    amp: bool = True                 # enable mixed precision
    mixed_precision: str = 'fp16'    # Accelerate: 'no', 'fp16', 'bf16'
    grad_accum_steps: int = 1        # can bump if you want larger effective batch
    use_channels_last: bool = True
    use_compile: bool = True         # use torch.compile if available

    # Logging / checkpointing
    output_dir: str = './outputs_dino_accel'
    save_every_epoch: int = 1
    seed: int = 15

    # Device hint (Accelerate will still manage device)
    device: str = 'cuda'

# ssl_dino/data.py
import os
from typing import List, Set

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class Solarization:
    def __init__(self, p: float = 0.0):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            return Image.eval(img, lambda x: 255 - x)
        return img


class DinoMultiCropTransform:
    def __init__(self, cfg: DINOConfig):
        self.cfg = cfg

        normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        common_color = [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
        ]

        # Global crops
        self.global_transform1 = T.Compose([
            T.RandomResizedCrop(
                cfg.image_size,
                scale=cfg.global_crops_scale,
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            *common_color,
            T.ToTensor(),
            normalize,
        ])

        self.global_transform2 = T.Compose([
            T.RandomResizedCrop(
                cfg.image_size,
                scale=cfg.global_crops_scale,
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            *common_color,
            Solarization(p=0.2),
            T.ToTensor(),
            normalize,
        ])

        # Local crops
        self.local_transform = T.Compose([
            T.RandomResizedCrop(
                cfg.image_size,
                scale=cfg.local_crops_scale,
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            *common_color,
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, img: Image.Image):
        crops = []
        crops.append(self.global_transform1(img))
        crops.append(self.global_transform2(img))
        for _ in range(self.cfg.local_crops_number):
            crops.append(self.local_transform(img))
        return crops


class UnlabeledImageFolder(Dataset):
    def __init__(self, root_dir: str, transform=None, extensions: Set[str] | None = None):
        self.root_dir = root_dir
        self.transform = transform
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.extensions = extensions
        self.paths = self._collect_paths()

    def _collect_paths(self) -> List[str]:
        paths: List[str] = []
        for fname in os.listdir(self.root_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.extensions:
                paths.append(os.path.join(self.root_dir, fname))
        paths.sort()
        if not paths:
            raise RuntimeError(f'No images found in {self.root_dir}')
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            return self.transform(img)
        return img


def build_dataloader(cfg: DINOConfig) -> DataLoader:
    transform = DinoMultiCropTransform(cfg)
    ds = UnlabeledImageFolder(root_dir=cfg.data_dir, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    return dl


# ssl_dino/vit.py
from typing import Optional

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 96, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 320):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W], possibly channels_last
        x = self.proj(x)  # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 5, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 320,
        depth: int = 9,
        num_heads: int = 5,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]
        return cls


# ssl_dino/dino_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = True,
    ):
        super().__init__()

        layers = []
        dim_in = in_dim
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(dim_in, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            dim_in = hidden_dim

        layers.append(nn.Linear(dim_in, bottleneck_dim, bias=False))
        self.mlp = nn.Sequential(*layers)

        last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer = weight_norm(last_layer)

        if norm_last_layer:
            with torch.no_grad():
                w = self.last_layer.weight
                self.last_layer.weight.copy_(w / torch.norm(w, dim=1, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOPredictionHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return x


# ssl_dino/dino_loss.py
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .config import DINOConfig


class DINOLoss(nn.Module):
    def __init__(self, cfg: DINOConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim

        self.register_buffer('center', torch.zeros(1, out_dim))
        self.teacher_temp = cfg.teacher_temp_base

    def set_teacher_temp(self, epoch: int):
        if epoch < self.cfg.teacher_temp_warmup_epochs:
            t = self.cfg.teacher_temp_base + (self.cfg.teacher_temp_end - self.cfg.teacher_temp_base) * (
                epoch / float(self.cfg.teacher_temp_warmup_epochs)
            )
        else:
            t = self.cfg.teacher_temp_end
        self.teacher_temp = t

    @torch.no_grad()
    def update_center(self, teacher_outputs: torch.Tensor):
        batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
        self.center = self.center * self.cfg.center_momentum + batch_center * (1.0 - self.cfg.center_momentum)

    def forward(
        self,
        student_outputs: Tuple[torch.Tensor, ...],
        teacher_outputs: Tuple[torch.Tensor, ...],
        epoch: int,
    ) -> torch.Tensor:
        self.set_teacher_temp(epoch)

        num_global = len(teacher_outputs)
        num_crops = len(student_outputs)
        assert num_global >= 1, 'Need at least one global crop for teacher'

        with torch.no_grad():
            teacher_cat = torch.cat(teacher_outputs, dim=0)
            teacher_cat = (teacher_cat - self.center) / self.teacher_temp
            teacher_cat = F.softmax(teacher_cat, dim=-1)

            B = teacher_outputs[0].shape[0]
            teacher_probs: List[torch.Tensor] = []
            for i in range(num_global):
                start = i * B
                end = (i + 1) * B
                teacher_probs.append(teacher_cat[start:end])

            self.update_center(teacher_cat)

        student_logprobs: List[torch.Tensor] = [
            F.log_softmax(out / self.cfg.student_temp, dim=-1) for out in student_outputs
        ]

        total_loss = 0.0
        n_terms = 0

        for t_idx in range(num_global):
            t_prob = teacher_probs[t_idx].detach()
            for s_idx in range(num_crops):
                if s_idx == t_idx and s_idx < num_global:
                    continue
                s_logprob = student_logprobs[s_idx]
                loss = torch.sum(-t_prob * s_logprob, dim=-1).mean()
                total_loss += loss
                n_terms += 1

        if n_terms == 0:
            return torch.tensor(0.0, device=student_outputs[0].device)

        return total_loss / n_terms


# ssl_dino/trainer.py
import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from accelerate import Accelerator

# from .config import DINOConfig
# from .vit import VisionTransformer
# from .dino_heads import DINOHead, DINOPredictionHead
# from .dino_loss import DINOLoss


# ============================================================
# Cosine LR/WD/momentum schedule
# ============================================================
def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_epoch,
    warmup_epochs=0,
    start_warmup_value=0.0,
):
    warmup_iters = warmup_epochs * niter_per_epoch
    total_iters = epochs * niter_per_epoch
    schedule = []

    for i in range(total_iters):
        if warmup_iters > 0 and i < warmup_iters:
            val = start_warmup_value + (base_value - start_warmup_value) * i / warmup_iters
        else:
            progress = (i - warmup_iters) / max(1, total_iters - warmup_iters)
            val = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        schedule.append(val)

    return schedule


# ============================================================
# Init student and teacher networks
# ============================================================
def init_dino_models(cfg: DINOConfig, device):
    student_backbone = VisionTransformer(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    ).to(device)

    student_head = DINOHead(
        in_dim=cfg.embed_dim,
        out_dim=cfg.out_dim,
    ).to(device)

    pred_head = None
    if cfg.use_prediction_head:
        pred_head = DINOPredictionHead(
            in_dim=cfg.out_dim, out_dim=cfg.out_dim
        ).to(device)

    # teacher networks
    teacher_backbone = VisionTransformer(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    ).to(device)

    teacher_head = DINOHead(
        in_dim=cfg.embed_dim,
        out_dim=cfg.out_dim,
    ).to(device)

    # Start teacher as student
    teacher_backbone.load_state_dict(student_backbone.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())

    # teacher is EMA, no grads
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    return student_backbone, teacher_backbone, student_head, teacher_head, pred_head


# ============================================================
# Optional optimization: compile + channels_last
# ============================================================
def maybe_optimize_model(cfg, model):
    if model is None:
        return None
    if cfg.use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


# ============================================================
# EMA teacher update
# ============================================================
def update_teacher(student, teacher, momentum):
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)


# ============================================================
# Trainer
# ============================================================
class DINOTrainer:
    def __init__(self, cfg: DINOConfig, dataloader, resume=None):
        self.cfg = cfg

        # Mixed precision, multi-GPU, device handling
        self.accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision if cfg.amp else "no",
            gradient_accumulation_steps=cfg.grad_accum_steps,
        )
        self.device = self.accelerator.device

        # ----------------------------
        # Build student/teacher models
        # ----------------------------
        (
            student_backbone,
            teacher_backbone,
            student_head,
            teacher_head,
            pred_head,
        ) = init_dino_models(cfg, self.device)

        # Optional compile + channels_last BEFORE prepare()
        student_backbone = maybe_optimize_model(cfg, student_backbone)
        # teacher_backbone = maybe_optimize_model(cfg, teacher_backbone)
        # student_head = maybe_optimize_model(cfg, student_head)
        # teacher_head = maybe_optimize_model(cfg, teacher_head)
        # pred_head = maybe_optimize_model(cfg, pred_head)

        if cfg.use_channels_last:
            teacher_backbone = teacher_backbone.to(memory_format=torch.channels_last)
            student_head = student_head.to(memory_format=torch.channels_last)
            teacher_head = teacher_head.to(memory_format=torch.channels_last)
            if pred_head is not None:
                pred_head = pred_head.to(memory_format=torch.channels_last)

        # Loss
        criterion = DINOLoss(cfg, out_dim=cfg.out_dim).to(self.device)

        # Optimizer
        params = list(student_backbone.parameters()) + list(student_head.parameters())
        if pred_head is not None:
            params += list(pred_head.parameters())

        optimizer = AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)

        # ----------------------------
        # Schedules
        # ----------------------------
        n_iter_per_epoch = len(dataloader)

        self.lr_schedule = cosine_scheduler(
            cfg.base_lr,
            1e-6,
            cfg.epochs,
            n_iter_per_epoch,
            warmup_epochs=cfg.warmup_epochs,
            start_warmup_value=1e-6,
        )
        self.wd_schedule = cosine_scheduler(
            cfg.weight_decay,
            cfg.weight_decay_end,
            cfg.epochs,
            n_iter_per_epoch,
        )
        self.momentum_schedule = cosine_scheduler(
            cfg.teacher_momentum_base,
            cfg.teacher_momentum_end,
            cfg.epochs,
            n_iter_per_epoch,
        )

        # ----------------------------
        # Prepare models for distributed + AMP
        # ----------------------------
        (
            self.student_backbone,
            self.teacher_backbone,
            self.student_head,
            self.teacher_head,
            self.pred_head,
            self.optimizer,
            self.dataloader,
            self.criterion,
        ) = self.accelerator.prepare(
            student_backbone,
            teacher_backbone,
            student_head,
            teacher_head,
            pred_head,
            optimizer,
            dataloader,
            criterion,
        )

        # Resume metadata
        self.start_epoch = 0
        self.start_global_iter = 0

        if resume is not None:
            self.load_checkpoint(resume)

    # ========================================================
    # Load checkpoint (resume)
    # ========================================================
    def load_checkpoint(self, path):
        accelerator = self.accelerator
        ckpt = torch.load(path, map_location=self.device)
        accelerator.print(f"Loading checkpoint: {path}")

        # Restore weights
        accelerator.unwrap_model(self.student_backbone).load_state_dict(ckpt["student_backbone"])
        accelerator.unwrap_model(self.teacher_backbone).load_state_dict(ckpt["teacher_backbone"])
        accelerator.unwrap_model(self.student_head).load_state_dict(ckpt["student_head"])
        accelerator.unwrap_model(self.teacher_head).load_state_dict(ckpt["teacher_head"])

        if self.pred_head is not None and "pred_head" in ckpt:
            accelerator.unwrap_model(self.pred_head).load_state_dict(ckpt["pred_head"])

        # Optimizer state
        self.optimizer.load_state_dict(ckpt["optimizer"])

        # EMA center
        self.criterion.center = ckpt["criterion_center"].to(self.device)

        # Schedules
        self.lr_schedule = ckpt["lr_schedule"]
        self.wd_schedule = ckpt["wd_schedule"]
        self.momentum_schedule = ckpt["momentum_schedule"]

        # Resume point
        self.start_epoch = ckpt["epoch"] + 1
        self.start_global_iter = ckpt["global_iter"]

        accelerator.print(f"Resuming training at epoch {self.start_epoch}")

    # ========================================================
    # Save checkpoint
    # ========================================================
    def save_checkpoint(self, epoch, global_iter):
        accelerator = self.accelerator
        if not accelerator.is_main_process:
            return

        student_backbone = accelerator.unwrap_model(self.student_backbone)
        teacher_backbone = accelerator.unwrap_model(self.teacher_backbone)
        student_head = accelerator.unwrap_model(self.student_head)
        teacher_head = accelerator.unwrap_model(self.teacher_head)
        pred_head = accelerator.unwrap_model(self.pred_head) if self.pred_head is not None else None

        ckpt = {
            "epoch": epoch,
            "global_iter": global_iter,

            "student_backbone": student_backbone.state_dict(),
            "teacher_backbone": teacher_backbone.state_dict(),
            "student_head": student_head.state_dict(),
            "teacher_head": teacher_head.state_dict(),

            "optimizer": self.optimizer.state_dict(),
            "criterion_center": self.criterion.center.clone().cpu(),

            "lr_schedule": self.lr_schedule,
            "wd_schedule": self.wd_schedule,
            "momentum_schedule": self.momentum_schedule,
        }

        if pred_head is not None:
            ckpt["pred_head"] = pred_head.state_dict()

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        path = os.path.join(self.cfg.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(ckpt, path)
        accelerator.print(f"Saved checkpoint → {path}")

    # ========================================================
    # Training loop (supports resume)
    # ========================================================
    def train(self):
        cfg = self.cfg
        accelerator = self.accelerator
        iters_per_epoch = len(self.dataloader)

        global_iter = self.start_global_iter

        for epoch in range(self.start_epoch, cfg.epochs):
            self.student_backbone.train()
            self.student_head.train()
            if self.pred_head is not None:
                self.pred_head.train()

            self.teacher_backbone.eval()
            self.teacher_head.eval()

            # tqdm only on main process
            if accelerator.is_local_main_process:
                progress = tqdm(
                    self.dataloader,
                    ncols=120,
                    desc=f"Epoch {epoch+1}/{cfg.epochs}",
                )
            else:
                progress = self.dataloader

            for crops in progress:

                # channels_last
                if cfg.use_channels_last:
                    crops = [c.to(memory_format=torch.channels_last) for c in crops]

                # LR & WD
                lr = self.lr_schedule[global_iter]
                wd = self.wd_schedule[global_iter]
                momentum = self.momentum_schedule[global_iter]

                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
                    pg["weight_decay"] = wd

                with accelerator.accumulate(self.student_backbone):

                    # Student forward for all crops
                    student_outs = []
                    for crop in crops:
                        feat = self.student_backbone(crop)
                        proj = self.student_head(feat)
                        if self.pred_head is not None:
                            proj = self.pred_head(proj)
                        student_outs.append(proj)

                    # Teacher forward for global crops only
                    with torch.no_grad():
                        teacher_outs = []
                        for gcrop in crops[: cfg.global_crops_number]:
                            tfeat = self.teacher_backbone(gcrop)
                            tproj = self.teacher_head(tfeat)
                            teacher_outs.append(tproj)

                    loss = self.criterion(
                        tuple(student_outs),
                        tuple(teacher_outs),
                        epoch,
                    )

                    self.optimizer.zero_grad()
                    accelerator.backward(loss)
                    self.optimizer.step()

                    # EMA teacher update
                    update_teacher(self.student_backbone, self.teacher_backbone, momentum)
                    update_teacher(self.student_head, self.teacher_head, momentum)

                # Update tqdm
                if accelerator.is_local_main_process and isinstance(progress, tqdm):
                    progress.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr:.6f}",
                        'wd': f'{wd:.5f}',
                        "m": f"{momentum:.6f}",
                    })

                global_iter += 1

            # Save at end of epoch
            if (
                (epoch + 1) % cfg.save_every_epoch == 0
                or (epoch + 1) == cfg.epochs
            ):
                self.save_checkpoint(epoch, global_iter)


# ssl_dino/train_dino.py
import argparse
import random

import numpy as np
import torch

# from .config import DINOConfig
# from .data import build_dataloader
# from .trainer import DINOTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='DINO pretraining with Accelerate on unlabeled images')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder with unlabeled images')
    parser.add_argument('--output_dir', type=str, default='./outputs_dino_accel', help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = DINOConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size_per_gpu=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    set_seed(cfg.seed)

    dataloader = build_dataloader(cfg)
    trainer = DINOTrainer(cfg, dataloader)
    trainer.train()


if __name__ == '__main__':
    main()


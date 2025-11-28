"""
BYOL: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning
Paper: https://arxiv.org/abs/2006.07733
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BYOL(nn.Module):
    """
    BYOL model - learns without negative pairs using asymmetric networks.
    """
    
    def __init__(
        self,
        base_encoder,
        projection_dim=256,
        hidden_dim=4096,
        momentum=0.996
    ):
        """
        Args:
            base_encoder: Backbone network
            projection_dim: Output dimension
            hidden_dim: Hidden dimension for projector and predictor
            momentum: Momentum coefficient for target network update
        """
        super().__init__()
        self.momentum = momentum
        
        # Online network
        self.encoder = base_encoder
        
        # Target network (exponential moving average of online)
        self.target_encoder = copy.deepcopy(base_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Get encoder dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if hasattr(self.encoder, 'fc'):
                encoder_dim = self.encoder.fc.in_features
                self.encoder.fc = nn.Identity()
                self.target_encoder.fc = nn.Identity()
            else:
                _, features = self.encoder(dummy_input, return_features=True)
                encoder_dim = features.shape[1]
        
        # Online projector
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Target projector
        self.target_projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        # Predictor (only for online network)
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    @torch.no_grad()
    def _update_target_network(self):
        """Update target network using exponential moving average."""
        for param_online, param_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_target.data = (
                param_target.data * self.momentum + param_online.data * (1. - self.momentum)
            )
        
        for param_online, param_target in zip(
            self.projector.parameters(), self.target_projector.parameters()
        ):
            param_target.data = (
                param_target.data * self.momentum + param_online.data * (1. - self.momentum)
            )
    
    def forward(self, x1, x2):
        """
        Forward pass with two augmented views.
        
        Args:
            x1, x2: Two augmented views (B, C, H, W)
            
        Returns:
            loss: BYOL loss
        """
        # Online network predictions
        h1_online = self.encoder(x1)
        z1_online = self.projector(h1_online)
        p1 = self.predictor(z1_online)
        
        h2_online = self.encoder(x2)
        z2_online = self.projector(h2_online)
        p2 = self.predictor(z2_online)
        
        # Target network projections (no gradient)
        with torch.no_grad():
            h1_target = self.target_encoder(x1)
            z1_target = self.target_projector(h1_target)
            
            h2_target = self.target_encoder(x2)
            z2_target = self.target_projector(h2_target)
        
        # Compute loss (symmetrized)
        loss = self.compute_loss(p1, z2_target) + self.compute_loss(p2, z1_target)
        
        return loss
    
    def compute_loss(self, pred, target):
        """
        Compute negative cosine similarity loss.
        
        Args:
            pred: Predictions from online network
            target: Targets from target network
            
        Returns:
            loss: Mean squared error of normalized predictions and targets
        """
        pred = F.normalize(pred, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()
    
    def update_moving_average(self):
        """Update target network (call after optimizer step)."""
        self._update_target_network()

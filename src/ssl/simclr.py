"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Paper: https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Projection head for SimCLR."""
    
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised learning.
    
    The model takes two augmented views of the same image and learns
    representations by maximizing agreement between them using contrastive loss.
    """
    
    def __init__(self, base_encoder, projection_dim=128, temperature=0.5):
        """
        Args:
            base_encoder: Backbone network (e.g., ResNet, ViT)
            projection_dim: Output dimension of projection head
            temperature: Temperature parameter for NT-Xent loss
        """
        super().__init__()
        self.encoder = base_encoder
        self.temperature = temperature
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if hasattr(self.encoder, 'fc'):
                # ResNet-like architecture
                encoder_dim = self.encoder.fc.in_features
                # Remove final FC layer
                self.encoder.fc = nn.Identity()
            else:
                # Get feature dimension
                _, features = self.encoder(dummy_input, return_features=True)
                encoder_dim = features.shape[1]
        
        # Projection head
        self.projection = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=encoder_dim,
            output_dim=projection_dim
        )
    
    def forward(self, x1, x2):
        """
        Forward pass for training.
        
        Args:
            x1: First augmented view (B, C, H, W)
            x2: Second augmented view (B, C, H, W)
            
        Returns:
            z1, z2: Projected representations for contrastive loss
        """
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project to contrastive space
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        
        return z1, z2
    
    def compute_loss(self, z1, z2):
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
        
        Args:
            z1, z2: Projected representations (B, projection_dim)
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = z1.shape[0]
        
        # Normalize representations
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # (2B, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
        
        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class SimCLRTrainer:
    """Helper class for training SimCLR."""
    
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train_step(self, x1, x2):
        """
        Single training step.
        
        Args:
            x1, x2: Augmented views of images
            
        Returns:
            loss: Loss value
        """
        self.model.train()
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # Forward pass
        z1, z2 = self.model(x1, x2)
        loss = self.model.compute_loss(z1, z2)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

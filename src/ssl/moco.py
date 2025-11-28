"""
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
Paper: https://arxiv.org/abs/1911.05722
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    MoCo model with momentum encoder and queue.
    """
    
    def __init__(
        self,
        base_encoder,
        projection_dim=128,
        queue_size=65536,
        momentum=0.999,
        temperature=0.07
    ):
        """
        Args:
            base_encoder: Backbone network
            projection_dim: Output dimension of projection head
            queue_size: Size of the memory queue
            momentum: Momentum coefficient for updating key encoder
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        # Query encoder
        self.encoder_q = base_encoder
        
        # Key encoder (momentum encoder)
        self.encoder_k = self._build_momentum_encoder(base_encoder)
        
        # Get encoder dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if hasattr(self.encoder_q, 'fc'):
                encoder_dim = self.encoder_q.fc.in_features
                self.encoder_q.fc = nn.Identity()
                self.encoder_k.fc = nn.Identity()
            else:
                _, features = self.encoder_q(dummy_input, return_features=True)
                encoder_dim = features.shape[1]
        
        # Projection heads
        self.projection_q = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
        self.projection_k = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
        
        # Copy weights from query to key projection
        for param_q, param_k in zip(
            self.projection_q.parameters(), self.projection_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Create the queue
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _build_momentum_encoder(self, base_encoder):
        """Build momentum encoder by copying base encoder."""
        import copy
        encoder_k = copy.deepcopy(base_encoder)
        for param in encoder_k.parameters():
            param.requires_grad = False
        return encoder_k
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder using momentum."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(
            self.projection_q.parameters(), self.projection_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue with new keys."""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest batch in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, xq, xk):
        """
        Forward pass.
        
        Args:
            xq: Query images (B, C, H, W)
            xk: Key images (B, C, H, W)
            
        Returns:
            logits, labels: For contrastive loss
        """
        # Compute query features
        q = self.encoder_q(xq)
        q = self.projection_q(q)
        q = F.normalize(q, dim=1)
        
        # Compute key features (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k = self.encoder_k(xk)
            k = self.projection_k(k)
            k = F.normalize(k, dim=1)
        
        # Compute logits
        # Positive pairs: (B,)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative pairs: (B, queue_size)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Concatenate: (B, 1 + queue_size)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positive is first
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return logits, labels
    
    def compute_loss(self, logits, labels):
        """Compute contrastive loss."""
        return F.cross_entropy(logits, labels)

"""
Builder for different SSL methods.
"""

from .simclr import SimCLR
# from .moco import MoCo
# from .byol import BYOL


def build_ssl_method(config, base_encoder):
    """
    Build SSL method wrapper around base encoder.
    
    Args:
        config: SSL method configuration
        base_encoder: Base encoder model (e.g., ResNet, ViT)
        
    Returns:
        ssl_model: SSL method instance
    """
    method_name = config.get('method', 'simclr').lower()
    
    if method_name == 'simclr':
        return SimCLR(
            base_encoder=base_encoder,
            projection_dim=config.get('projection_dim', 128),
            temperature=config.get('temperature', 0.5),
        )
    # elif method_name == 'moco':
    #     return MoCo(
    #         base_encoder=base_encoder,
    #         projection_dim=config.get('projection_dim', 128),
    #         queue_size=config.get('queue_size', 65536),
    #         momentum=config.get('momentum', 0.999),
    #         temperature=config.get('temperature', 0.07),
        # )
    # elif method_name == 'byol':
    #     return BYOL(
    #         base_encoder=base_encoder,
    #         projection_dim=config.get('projection_dim', 256),
    #         hidden_dim=config.get('hidden_dim', 4096),
    #         momentum=config.get('momentum', 0.996),
    #     )
    else:
        raise ValueError(f"Unknown SSL method: {method_name}. Only 'simclr' is currently supported.")

# Copyright (c) OpenMMLab. All rights reserved.
"""Prompt Fusion Module for Multi-Modal Interactive Segmentation.

This module fuses different types of prompts (point, box, text) for interactive segmentation.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType


@MODELS.register_module()
class TextEncoder(nn.Module):
    """Text encoder for text prompts.
    
    Uses CLIP text encoder to encode text prompts.
    
    Args:
        text_model_cfg (ConfigType): Configuration for text model.
            If None, uses a simple embedding layer. Default: None.
        feat_channels (int): Output feature channel dimension. Default: 256.
    """
    
    def __init__(self,
                 text_model_cfg: OptConfigType = None,
                 feat_channels: int = 256):
        super().__init__()
        self.feat_channels = feat_channels
        self.text_model = None
        
        if text_model_cfg is not None:
            # Build text model from config (e.g., CLIP text encoder)
            self.text_model = MODELS.build(text_model_cfg)
            # Project text features to feat_channels if needed
            if hasattr(self.text_model, 'embed_dim'):
                text_dim = self.text_model.embed_dim
                if text_dim != feat_channels:
                    self.text_proj = nn.Linear(text_dim, feat_channels)
                else:
                    self.text_proj = nn.Identity()
            else:
                self.text_proj = nn.Identity()
        else:
            # Simple embedding layer as fallback
            # This is a placeholder - in practice, you should use CLIP text encoder
            self.text_proj = nn.Linear(512, feat_channels)  # Placeholder
    
    def forward(self, text: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Encode text prompts.
        
        Args:
            text: Text token IDs [B, L] or text embeddings [B, L, C]. Optional.
            
        Returns:
            Text embeddings [B, N_text, C] or None.
        """
        if text is None:
            return None
        
        if self.text_model is not None:
            # Use text model to encode
            if text.dim() == 2:  # Token IDs
                text_embed = self.text_model(text)  # [B, C] or [B, L, C]
            else:  # Already embeddings
                text_embed = text
        else:
            # Placeholder: return None for now
            # In practice, this should use CLIP text encoder
            return None
        
        # Project to feat_channels
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)  # [B, 1, C]
        text_embed = self.text_proj(text_embed)
        
        return text_embed


@MODELS.register_module()
class PromptFusion(nn.Module):
    """Prompt Fusion Module for multi-modal interactive segmentation.
    
    Fuses point, box, and text prompts using cross-attention.
    
    Args:
        feat_channels (int): Feature channel dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        dropout (float): Dropout rate. Default: 0.1.
        use_text_encoder (bool): Whether to use text encoder. Default: True.
        text_encoder (OptConfigType): Text encoder config. Default: None.
    """
    
    def __init__(self,
                 feat_channels: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_text_encoder: bool = True,
                 text_encoder: OptConfigType = None):
        super().__init__()
        self.feat_channels = feat_channels
        self.num_heads = num_heads
        self.use_text_encoder = use_text_encoder
        
        # Text encoder
        if use_text_encoder:
            if text_encoder is None:
                text_encoder = dict(type='TextEncoder', feat_channels=feat_channels)
            self.text_encoder = MODELS.build(text_encoder)
        else:
            self.text_encoder = None
        
        # Cross-attention for prompt fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm and FFN
        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.ffn = nn.Sequential(
            nn.Linear(feat_channels, feat_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_channels * 4, feat_channels),
            nn.Dropout(dropout)
        )
    
    def forward(self,
                point_embed: Optional[torch.Tensor] = None,
                box_embed: Optional[torch.Tensor] = None,
                text: Optional[torch.Tensor] = None,
                text_embed: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Fuse different prompt embeddings.
        
        Args:
            point_embed: Point prompt embedding [B, N_point, C]. Optional.
            box_embed: Box prompt embedding [B, N_box, C]. Optional.
            text: Text token IDs [B, L] for encoding. Optional.
            text_embed: Pre-encoded text embedding [B, N_text, C]. Optional.
            
        Returns:
            Fused prompt embedding [B, N_total, C] or None if no prompts.
        """
        # Encode text if provided and text_embed is None
        if text is not None and text_embed is None and self.text_encoder is not None:
            text_embed = self.text_encoder(text)
        
        # Collect available prompts
        prompts = []
        if point_embed is not None:
            prompts.append(point_embed)
        if box_embed is not None:
            prompts.append(box_embed)
        if text_embed is not None:
            prompts.append(text_embed)
        
        if not prompts:
            # Return None if no prompts
            return None
        
        # Concatenate all prompts
        concat_prompts = torch.cat(prompts, dim=1)  # [B, N_total, C]
        
        # Self-attention for prompt fusion
        fused, _ = self.cross_attn(
            concat_prompts, concat_prompts, concat_prompts
        )
        fused = self.norm1(fused + concat_prompts)
        
        # FFN
        out = self.ffn(fused)
        out = self.norm2(out + fused)
        
        return out
    
    def compute_text_visual_alignment_loss(self,
                                           text_embed: torch.Tensor,
                                           visual_embed: torch.Tensor,
                                           mask_labels: torch.Tensor) -> torch.Tensor:
        """Compute text-visual alignment loss for multi-task training.
        
        Args:
            text_embed: Text embedding [B, N_text, C].
            visual_embed: Visual feature embedding [B, N_inst, C].
            mask_labels: Mask labels [B, N_inst, H, W].
            
        Returns:
            Alignment loss scalar.
        """
        # Compute similarity between text and visual embeddings
        # This is a simplified version - can be enhanced
        text_mean = text_embed.mean(dim=1)  # [B, C]
        visual_mean = visual_embed.mean(dim=1)  # [B, C]
        
        # Cosine similarity loss
        similarity = F.cosine_similarity(text_mean, visual_mean, dim=-1)
        loss = 1.0 - similarity.mean()
        
        return loss


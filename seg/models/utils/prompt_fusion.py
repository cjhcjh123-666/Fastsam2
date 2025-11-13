# Copyright (c) OpenMMLab. All rights reserved.
"""Prompt Fusion Module for Multi-Modal Interactive Segmentation.

This module fuses different types of prompts (point, box, text) for interactive segmentation.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class PromptFusion(nn.Module):
    """Prompt Fusion Module for multi-modal interactive segmentation.
    
    Fuses point, box, and text prompts using cross-attention.
    
    Args:
        feat_channels (int): Feature channel dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        dropout (float): Dropout rate. Default: 0.1.
    """
    
    def __init__(self,
                 feat_channels: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.feat_channels = feat_channels
        self.num_heads = num_heads
        
        # Prompt encoders (these should be provided by SAMPromptEncoder)
        # Here we just define fusion layers
        
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
                text_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse different prompt embeddings.
        
        Args:
            point_embed: Point prompt embedding [B, N_point, C]. Optional.
            box_embed: Box prompt embedding [B, N_box, C]. Optional.
            text_embed: Text prompt embedding [B, N_text, C]. Optional.
            
        Returns:
            Fused prompt embedding [B, N_total, C].
        """
        # Collect available prompts
        prompts = []
        if point_embed is not None:
            prompts.append(point_embed)
        if box_embed is not None:
            prompts.append(box_embed)
        if text_embed is not None:
            prompts.append(text_embed)
        
        if not prompts:
            # Return empty embedding if no prompts
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


# Copyright (c) OpenMMLab. All rights reserved.
"""Prompt Fusion Module for Multi-Modal Interactive Segmentation.

This module fuses different types of prompts (point, box, text) for interactive segmentation.
"""
from typing import Dict, List, Optional, Tuple, Union

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
            elif hasattr(self.text_model, 'text_proj') and hasattr(self.text_model.text_proj, 'shape'):
                # For OpenCLIP models, infer embed_dim from text_proj shape
                # text_proj is [transformer_width, embed_dim], output is embed_dim
                text_dim = self.text_model.text_proj.shape[1]
            else:
                # Default to 768 for CLIP-like models (e.g., ViT-L-14)
                text_dim = 768
            
            if text_dim != feat_channels:
                self.text_proj = nn.Linear(text_dim, feat_channels)
            else:
                self.text_proj = nn.Identity()
        else:
            # Simple embedding layer as fallback
            # This is a placeholder - in practice, you should use CLIP text encoder
            self.text_proj = nn.Linear(512, feat_channels)  # Placeholder
            self.text_model = None
    
    def forward(self, text: Optional[Union[torch.Tensor, List[str]]]) -> Optional[torch.Tensor]:
        """Encode text prompts.
        
        Args:
            text: Text token IDs [B, L], text embeddings [B, L, C], or list of strings. Optional.
            
        Returns:
            Text embeddings [B, N_text, C] or None.
        """
        # CRITICAL for DDP: Even when text is None, we must call text_model with dummy input
        # to ensure all parameters participate in gradient computation
        if text is None:
            if self.text_model is not None:
                # Create a dummy text input to ensure parameters participate in gradient computation
                # For OpenCLIPBackboneText, we need actual tokenized text
                # Safely get device from text_model or text_proj
                try:
                    device = next(self.text_model.parameters()).device
                except StopIteration:
                        device = next(self.text_proj.parameters()).device
                
                # Try to create a dummy token sequence
                # For CLIP models, we can use a padding token (usually 0) or EOS token
                # Create a minimal valid token sequence: [B=1, L=1] with padding token
                try:
                    # Try to get tokenizer to create proper dummy tokens
                    if hasattr(self.text_model, 'text_tokenizer'):
                        # Create a dummy text string and tokenize it
                        dummy_text = [""]  # Empty string
                        dummy_tokens = self.text_model.text_tokenizer(dummy_text).to(device)
                        # Encode the dummy tokens
                        text_embed = self.text_model(dummy_text)
                        # Project but multiply by 0 to not affect output
                        if text_embed.dim() == 2:
                            text_embed = text_embed.unsqueeze(1)
                        text_embed = self.text_proj(text_embed) * 0.0  # Zero out to not affect output
                        return text_embed
                    else:
                        # Fallback: return None but this might cause DDP issues
                        # In this case, the caller should handle it
                        return None
                except Exception:
                    # If anything fails, return None
                    # The caller (PromptFusion) should handle this
                    return None
            return None
        
        # Handle list of strings - would need tokenizer
        if isinstance(text, list):
            # If text_model is available and has tokenizer, use it
            # For now, return None as placeholder
            # In practice, this should:
            # 1. Tokenize strings using CLIP tokenizer
            # 2. Encode tokens using text_model
            if self.text_model is not None and hasattr(self.text_model, 'text_tokenizer'):
                # Tokenize
                tokenized = []
                for txt in text:
                    tokens = self.text_model.text_tokenizer(txt)
                    tokenized.append(tokens)
                text_tokens = torch.stack(tokenized)
                # Encode
                text_embed = self.text_model(text_tokens)
            else:
                return None
        elif self.text_model is not None:
            # Use text model to encode
            if text.dim() == 2:  # Token IDs
                text_embed = self.text_model(text)  # [B, C] or [B, L, C]
            else:  # Already embeddings
                text_embed = text
        else:
            # No text model available
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
                text: Optional[Union[torch.Tensor, List[str]]] = None,
                text_embed: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Fuse different prompt embeddings.
        
        Args:
            point_embed: Point prompt embedding [B, N_point, C]. Optional.
            box_embed: Box prompt embedding [B, N_box, C]. Optional.
            text: Text token IDs [B, L] or list of strings for encoding. Optional.
            text_embed: Pre-encoded text embedding [B, N_text, C]. Optional.
            
        Returns:
            Fused prompt embedding [B, N_total, C] or None if no prompts.
        """
        # CRITICAL for DDP: Always call text_encoder to ensure parameters have gradients
        # This prevents "Expected to have finished reduction" errors in distributed training
        if self.text_encoder is not None:
            # Always call text_encoder, even with None input, to ensure DDP compatibility
            if text is not None and text_embed is None:
                # Handle list of strings
                if isinstance(text, list):
                    text_embed = self.text_encoder(text)
                else:
                    text_embed = self.text_encoder(text)
            elif text is None and text_embed is None:
                # CRITICAL: Always call text_encoder even with None to ensure parameters participate
                # This creates a dummy embedding that ensures gradient flow
                dummy_text_embed = self.text_encoder(None)
                # If dummy_text_embed is not None, we'll use it (multiplied by 0 to not affect output)
                # If it's None, we'll handle it below
                if dummy_text_embed is not None:
                    text_embed = dummy_text_embed * 0.0  # Zero out to not affect output but maintain gradient flow
        
        # Strategy: Use text to enhance point/box embeddings via cross-attention
        # This maintains the number of queries (points/boxes) while incorporating text information
        
        # Start with point or box embeddings as the base
        base_embed = point_embed if point_embed is not None else box_embed
        
        if base_embed is None:
            # No point or box embeddings, only text
            # In this case, return text_embed directly (rare case)
            if text_embed is not None:
                return text_embed
            else:
                return None
        
        batch_size = base_embed.shape[0]
        
        # If we have text embeddings, use cross-attention to fuse them with base embeddings
        # Query: base_embed (point/box), Key+Value: text_embed
        # This enhances base_embed with text information without changing its shape
        if text_embed is not None and text_embed.abs().sum() > 0:  # Check if not all zeros
            # Use cross-attention: text enhances point/box embeddings
            # Query from base, Key+Value from text
            enhanced, _ = self.cross_attn(
                base_embed, text_embed, text_embed
            )
            enhanced = self.norm1(enhanced + base_embed)
            
            # FFN
            out = self.ffn(enhanced)
            out = self.norm2(out + enhanced)
            
            return out
        else:
            # No text or text is dummy (all zeros), return base_embed unchanged
            # But still need to pass through network for gradient flow if text_encoder was called
            if text_embed is not None:
                # Text encoder was called (for DDP), so pass base through attention with itself
                enhanced, _ = self.cross_attn(
                    base_embed, base_embed, base_embed
                )
                enhanced = self.norm1(enhanced + base_embed)
                
                # FFN
                out = self.ffn(enhanced)
                out = self.norm2(out + enhanced)
                
                return out
            else:
                # No text encoder at all, return base as is
                return base_embed
    
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


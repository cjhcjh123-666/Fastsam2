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
        
        # Collect available prompts
        prompts = []
        batch_size = None
        
        if point_embed is not None:
            prompts.append(point_embed)
            batch_size = point_embed.shape[0]
        if box_embed is not None:
            prompts.append(box_embed)
            if batch_size is None:
                batch_size = box_embed.shape[0]
            elif batch_size != box_embed.shape[0]:
                # Batch size mismatch, pad or trim to match
                if batch_size < box_embed.shape[0]:
                    # Trim box_embed
                    box_embed = box_embed[:batch_size]
                else:
                    # Pad box_embed with empty embeddings
                    feat_dim = box_embed.shape[-1]
                    pad_size = batch_size - box_embed.shape[0]
                    empty_box = torch.empty((pad_size, 0, feat_dim), device=box_embed.device)
                    box_embed = torch.cat([box_embed, empty_box], dim=0)
                prompts[-1] = box_embed  # Update the last added prompt
        if text_embed is not None:
            prompts.append(text_embed)
            if batch_size is None:
                batch_size = text_embed.shape[0]
            elif batch_size != text_embed.shape[0]:
                # Batch size mismatch, pad or trim to match
                if batch_size < text_embed.shape[0]:
                    # Trim text_embed
                    text_embed = text_embed[:batch_size]
                else:
                    # Pad text_embed with empty embeddings
                    feat_dim = text_embed.shape[-1]
                    pad_size = batch_size - text_embed.shape[0]
                    empty_text = torch.empty((pad_size, 0, feat_dim), device=text_embed.device)
                    text_embed = torch.cat([text_embed, empty_text], dim=0)
                prompts[-1] = text_embed  # Update the last added prompt
        
        if not prompts:
            # No real prompts, but we called text_encoder for DDP
            # If we have a dummy text_embed (from text_encoder(None)), use it to ensure gradient flow
            if text_embed is not None:
                # Use the dummy embedding (already zeroed out) to ensure gradient flow
                # Return a minimal embedding that doesn't affect output
                return text_embed
            # If no prompts at all, return None
            # This should be rare if text_encoder always returns something
            return None
        
        # Ensure all prompts have the same batch size
        if batch_size is not None:
            for i, prompt in enumerate(prompts):
                if prompt.shape[0] != batch_size:
                    # Pad or trim to match batch_size
                    if prompt.shape[0] < batch_size:
                        feat_dim = prompt.shape[-1]
                        pad_size = batch_size - prompt.shape[0]
                        empty_prompt = torch.empty((pad_size, 0, feat_dim), device=prompt.device)
                        prompts[i] = torch.cat([prompt, empty_prompt], dim=0)
                    else:
                        prompts[i] = prompt[:batch_size]
        
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


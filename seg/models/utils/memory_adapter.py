# Copyright (c) OpenMMLab. All rights reserved.
"""Streaming Memory Adapter for Video Object Segmentation.

This module provides long-term and short-term memory management for VOS tasks.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine.structures import InstanceData


@MODELS.register_module()
class StreamingMemoryAdapter(nn.Module):
    """Streaming Memory Adapter for VOS.
    
    Maintains long-term and short-term memory for video object segmentation:
    - long_mem: Key frames (sparse, high-quality)
    - short_mem: Recent frames (dense, for temporal consistency)
    
    Args:
        feat_channels (int): Feature channel dimension. Default: 256.
        long_mem_size (int): Maximum number of key frames in long-term memory.
            Default: 10.
        short_mem_size (int): Maximum number of frames in short-term memory.
            Default: 5.
        update_strategy (str): Memory update strategy ('fifo', 'quality', 'adaptive').
            Default: 'adaptive'.
    """
    
    def __init__(self,
                 feat_channels: int = 256,
                 long_mem_size: int = 10,
                 short_mem_size: int = 5,
                 update_strategy: str = 'adaptive'):
        super().__init__()
        self.feat_channels = feat_channels
        self.long_mem_size = long_mem_size
        self.short_mem_size = short_mem_size
        self.update_strategy = update_strategy
        
        # Memory storage
        self.long_mem: Dict[int, Dict] = {}  # {frame_id: {embed, mask, instance_id}}
        self.short_mem: List[Dict] = []  # [{frame_id, embed, mask, instance_id}]
        
        # Memory update network
        self.memory_update = nn.Sequential(
            nn.Linear(feat_channels * 2, feat_channels),
            nn.ReLU(),
            nn.Linear(feat_channels, feat_channels)
        )
        
        # Quality scorer for adaptive update
        if update_strategy == 'adaptive':
            self.quality_scorer = nn.Sequential(
                nn.Linear(feat_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def reset(self):
        """Reset memory storage."""
        self.long_mem.clear()
        self.short_mem.clear()
    
    def update(self,
               frame_id: int,
               instance_embed: torch.Tensor,
               mask: torch.Tensor,
               instance_id: Optional[int] = None,
               text_embed: Optional[torch.Tensor] = None):
        """Update memory with new frame.
        
        Args:
            frame_id: Current frame ID.
            instance_embed: Instance embedding tensor [N, C].
            mask: Instance mask tensor [N, H, W] or BitmapMasks object.
            instance_id: Optional instance ID for tracking.
            text_embed: Optional text embedding for language-guided memory.
        """
        # Convert mask to tensor if it's BitmapMasks
        from mmdet.structures.mask import BitmapMasks
        if isinstance(mask, BitmapMasks):
            # Convert BitmapMasks to tensor
            mask_tensor = mask.to_tensor(dtype=torch.float32, device=instance_embed.device)
        elif isinstance(mask, torch.Tensor):
            mask_tensor = mask.detach() if mask.requires_grad else mask
        else:
            # Try to convert to tensor
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=instance_embed.device)
        
        # Update short-term memory (FIFO)
        mem_entry = {
            'frame_id': frame_id,
            'embed': instance_embed.detach() if instance_embed.requires_grad else instance_embed,
            'mask': mask_tensor,
            'instance_id': instance_id,
            'text_embed': text_embed.detach() if text_embed is not None and text_embed.requires_grad else text_embed
        }
        
        self.short_mem.append(mem_entry)
        if len(self.short_mem) > self.short_mem_size:
            self.short_mem.pop(0)
        
        # Update long-term memory based on strategy
        if self.update_strategy == 'fifo':
            self._update_long_mem_fifo(frame_id, mem_entry)
        elif self.update_strategy == 'quality':
            self._update_long_mem_quality(frame_id, mem_entry)
        elif self.update_strategy == 'adaptive':
            self._update_long_mem_adaptive(frame_id, mem_entry)
    
    def _update_long_mem_fifo(self, frame_id: int, mem_entry: Dict):
        """Update long-term memory using FIFO strategy."""
        self.long_mem[frame_id] = mem_entry
        if len(self.long_mem) > self.long_mem_size:
            # Remove oldest frame
            oldest_frame = min(self.long_mem.keys())
            del self.long_mem[oldest_frame]
    
    def _update_long_mem_quality(self, frame_id: int, mem_entry: Dict):
        """Update long-term memory based on quality score."""
        if len(self.long_mem) < self.long_mem_size:
            self.long_mem[frame_id] = mem_entry
        else:
            # Score all frames including new one
            all_frames = list(self.long_mem.keys()) + [frame_id]
            all_embeds = [self.long_mem[f]['embed'] for f in self.long_mem.keys()] + [mem_entry['embed']]
            all_embeds = torch.stack(all_embeds)
            
            # Compute quality scores
            with torch.no_grad():
                scores = self.quality_scorer(all_embeds.mean(dim=1))
            
            # Keep top-k frames
            _, top_indices = torch.topk(scores.squeeze(), self.long_mem_size)
            new_long_mem = {}
            for idx in top_indices:
                frame_key = all_frames[idx.item()]
                if frame_key in self.long_mem:
                    new_long_mem[frame_key] = self.long_mem[frame_key]
                else:
                    new_long_mem[frame_key] = mem_entry
            self.long_mem = new_long_mem
    
    def _update_long_mem_adaptive(self, frame_id: int, mem_entry: Dict):
        """Update long-term memory using adaptive strategy."""
        # For now, use quality-based update
        self._update_long_mem_quality(frame_id, mem_entry)
    
    def fetch(self,
              frame_id: int,
              instance_id: Optional[int] = None,
              text_query: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch relevant memory for current frame.
        
        Args:
            frame_id: Current frame ID.
            instance_id: Optional instance ID to filter memory.
            text_query: Optional text embedding for language-guided retrieval.
            
        Returns:
            Tuple of (memory_embed, memory_mask):
            - memory_embed: Aggregated memory embedding [N, C]
            - memory_mask: Aggregated memory mask [N, H, W]
        """
        # Collect relevant memory entries
        relevant_mem = []
        
        # Add short-term memory
        for entry in self.short_mem:
            if entry['frame_id'] != frame_id:  # Exclude current frame
                if instance_id is None or entry['instance_id'] == instance_id:
                    relevant_mem.append(entry)
        
        # Add long-term memory
        for frame_key, entry in self.long_mem.items():
            if frame_key != frame_id:  # Exclude current frame
                if instance_id is None or entry['instance_id'] == instance_id:
                    relevant_mem.append(entry)
        
        if not relevant_mem:
            # Return empty memory
            return None, None
        
        # Language-guided filtering if text_query is provided
        if text_query is not None:
            relevant_mem = self._filter_by_text_similarity(relevant_mem, text_query)
        
        # Aggregate memory
        embeds = [entry['embed'] for entry in relevant_mem]
        masks = [entry['mask'] for entry in relevant_mem]
        
        # Simple aggregation (can be improved with attention)
        memory_embed = torch.stack(embeds).mean(dim=0)
        memory_mask = torch.stack(masks).mean(dim=0)
        
        return memory_embed, memory_mask
    
    def _filter_by_text_similarity(self,
                                   mem_entries: List[Dict],
                                   text_query: torch.Tensor) -> List[Dict]:
        """Filter memory entries by text similarity.
        
        Args:
            mem_entries: List of memory entries.
            text_query: Text embedding query [C].
            
        Returns:
            Filtered memory entries.
        """
        if not mem_entries:
            return []
        
        # Compute similarities
        similarities = []
        for entry in mem_entries:
            if entry['text_embed'] is not None:
                sim = F.cosine_similarity(
                    text_query.unsqueeze(0),
                    entry['text_embed'].unsqueeze(0)
                )
                similarities.append(sim.item())
            else:
                similarities.append(0.0)
        
        # Keep top-k similar entries
        if len(mem_entries) <= self.short_mem_size:
            return mem_entries
        
        sorted_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )
        top_k = sorted_indices[:self.short_mem_size]
        
        return [mem_entries[i] for i in top_k]
    
    def forward(self,
                frame_id: int,
                current_embed: torch.Tensor,
                instance_id: Optional[int] = None,
                text_query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass to retrieve and fuse memory.
        
        Args:
            frame_id: Current frame ID.
            current_embed: Current frame embedding [N, C].
            instance_id: Optional instance ID.
            text_query: Optional text query for language-guided retrieval.
            
        Returns:
            Enhanced embedding with memory [N, C].
        """
        memory_embed, _ = self.fetch(frame_id, instance_id, text_query)
        
        if memory_embed is None:
            return current_embed
        
        # Fuse current and memory embeddings
        concat_embed = torch.cat([current_embed, memory_embed], dim=-1)
        fused_embed = self.memory_update(concat_embed)
        
        return fused_embed


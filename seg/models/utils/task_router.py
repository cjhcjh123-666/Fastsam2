# Copyright (c) OpenMMLab. All rights reserved.
"""Task Router for Multi-Task Segmentation.

This module provides task routing functionality to support:
- Interactive Image Segmentation (point/box/text)
- Interactive Video Segmentation (point/box/text)
- Video Object Segmentation (VOS)
- Panoptic Segmentation
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from mmdet.registry import MODELS


class TaskType(Enum):
    """Task type enumeration."""
    INTERACTIVE_IMAGE = 'interactive_image'
    INTERACTIVE_VIDEO = 'interactive_video'
    VOS = 'vos'
    PANOPTIC = 'panoptic'


class PromptType(Enum):
    """Prompt type enumeration."""
    POINT = 'point'
    BOX = 'box'
    TEXT = 'text'
    NONE = 'none'


@MODELS.register_module()
class TaskRouter(nn.Module):
    """Task Router for multi-task segmentation.
    
    Routes different tasks to appropriate processing paths based on:
    - Task type (interactive/vos/panoptic)
    - Prompt type (point/box/text/none)
    - Video length
    - Mode (training/inference)
    
    Args:
        feat_channels (int): Feature channel dimension. Default: 256.
        num_decoder_stages (int): Number of decoder stages. Default: 3.
        enable_streaming_memory (bool): Whether to enable streaming memory for VOS.
            Default: True.
        interactive_stages (int): Number of decoder stages for interactive tasks.
            Default: 3.
        vos_stages (int): Number of decoder stages for VOS. Default: 3.
        panoptic_stages (int): Number of decoder stages for panoptic. Default: 3.
    """
    
    def __init__(self,
                 feat_channels: int = 256,
                 num_decoder_stages: int = 3,
                 enable_streaming_memory: bool = True,
                 interactive_stages: int = 3,
                 vos_stages: int = 3,
                 panoptic_stages: int = 3):
        super().__init__()
        self.feat_channels = feat_channels
        self.num_decoder_stages = num_decoder_stages
        self.enable_streaming_memory = enable_streaming_memory
        self.interactive_stages = interactive_stages
        self.vos_stages = vos_stages
        self.panoptic_stages = panoptic_stages
    
    def detect_task_type(self, 
                        data_samples: List,
                        prompts: Optional[Dict] = None) -> TaskType:
        """Detect task type from data samples and prompts.
        
        Args:
            data_samples: List of data samples.
            prompts: Optional prompts dict containing point_coords, bboxes, text.
            
        Returns:
            TaskType: Detected task type.
        """
        # Check if it's video data
        from mmdet.structures import TrackDataSample
        is_video = isinstance(data_samples[0], TrackDataSample) if data_samples else False
        
        # Check if prompts exist (interactive task)
        has_prompts = False
        if prompts:
            has_prompts = any([
                prompts.get('point_coords') is not None,
                prompts.get('bboxes') is not None,
                prompts.get('text') is not None
            ])
        
        # Check if gt_instances_collected exists (interactive training)
        has_gt_collected = any(
            hasattr(ds, 'gt_instances_collected') and ds.gt_instances_collected is not None
            for ds in data_samples
        )
        
        # Check if it's VOS (has instance IDs in video)
        is_vos = False
        if is_video and data_samples:
            track_sample = data_samples[0]
            if hasattr(track_sample, 'video_data_samples') and track_sample.video_data_samples:
                first_frame = track_sample.video_data_samples[0]
                if hasattr(first_frame, 'gt_instances') and first_frame.gt_instances:
                    is_vos = hasattr(first_frame.gt_instances, 'instances_ids')
        
        # Determine task type
        if has_prompts or has_gt_collected:
            if is_video:
                return TaskType.INTERACTIVE_VIDEO
            else:
                return TaskType.INTERACTIVE_IMAGE
        elif is_vos:
            return TaskType.VOS
        else:
            return TaskType.PANOPTIC
    
    def detect_prompt_type(self, prompts: Optional[Dict] = None) -> PromptType:
        """Detect prompt type from prompts dict.
        
        Args:
            prompts: Prompts dict containing point_coords, bboxes, text.
            
        Returns:
            PromptType: Detected prompt type.
        """
        if not prompts:
            return PromptType.NONE
        
        if prompts.get('text') is not None:
            return PromptType.TEXT
        elif prompts.get('bboxes') is not None:
            return PromptType.BOX
        elif prompts.get('point_coords') is not None:
            return PromptType.POINT
        else:
            return PromptType.NONE
    
    def route(self,
              task_type: TaskType,
              prompt_type: PromptType,
              video_length: int = 1,
              mode: str = 'train') -> Dict:
        """Route task to appropriate configuration.
        
        Args:
            task_type: Task type.
            prompt_type: Prompt type.
            video_length: Video sequence length. Default: 1.
            mode: Mode ('train' or 'test'). Default: 'train'.
            
        Returns:
            Dict containing routing configuration:
            - num_stages: Number of decoder stages to use
            - use_streaming_memory: Whether to use streaming memory
            - active_query_subset: Query indices to activate
            - task_specific_config: Task-specific configuration
        """
        config = {
            'num_stages': self.num_decoder_stages,
            'use_streaming_memory': False,
            'active_query_subset': None,  # None means use all queries
            'task_specific_config': {}
        }
        
        if task_type == TaskType.INTERACTIVE_IMAGE:
            config['num_stages'] = self.interactive_stages
            config['use_streaming_memory'] = False
            config['task_specific_config'] = {
                'prompt_type': prompt_type.value,
                'enable_prompt_fusion': True
            }
        
        elif task_type == TaskType.INTERACTIVE_VIDEO:
            config['num_stages'] = self.interactive_stages
            config['use_streaming_memory'] = self.enable_streaming_memory and video_length > 1
            config['task_specific_config'] = {
                'prompt_type': prompt_type.value,
                'enable_prompt_fusion': True,
                'video_length': video_length
            }
        
        elif task_type == TaskType.VOS:
            config['num_stages'] = self.vos_stages
            config['use_streaming_memory'] = self.enable_streaming_memory
            config['task_specific_config'] = {
                'enable_mask_propagation': True,
                'enable_dpsr': True,  # Dual-Path Self-Refinement
                'video_length': video_length
            }
        
        elif task_type == TaskType.PANOPTIC:
            config['num_stages'] = self.panoptic_stages
            config['use_streaming_memory'] = False
            config['task_specific_config'] = {
                'enable_panoptic_fusion': True
            }
        
        return config
    
    def forward(self,
                data_samples: List,
                prompts: Optional[Dict] = None,
                video_length: int = 1,
                mode: str = 'train') -> Dict:
        """Forward pass to detect and route task.
        
        Args:
            data_samples: List of data samples.
            prompts: Optional prompts dict.
            video_length: Video sequence length.
            mode: Mode ('train' or 'test').
            
        Returns:
            Dict containing routing configuration.
        """
        task_type = self.detect_task_type(data_samples, prompts)
        prompt_type = self.detect_prompt_type(prompts)
        return self.route(task_type, prompt_type, video_length, mode)


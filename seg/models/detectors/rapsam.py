# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors import SingleStageDetector
from mmdet.structures import SampleList

from .mask2former_vid import Mask2formerVideo
from seg.models.utils import TaskRouter, StreamingMemoryAdapter, PromptFusion


@MODELS.register_module()
class RapSAM(Mask2formerVideo):
    """RapSAM: Multi-Task Segmentation Model.
    
    Supports:
    - Interactive Image Segmentation (point/box/text)
    - Interactive Video Segmentation (point/box/text)
    - Video Object Segmentation (VOS)
    - Panoptic Segmentation
    
    Args:
        use_task_router (bool): Whether to use task router. Default: True.
        task_router (OptConfigType): Task router config. Default: None.
        use_streaming_memory (bool): Whether to use streaming memory for VOS.
            Default: True.
        streaming_memory (OptConfigType): Streaming memory config. Default: None.
        use_prompt_fusion (bool): Whether to use prompt fusion. Default: True.
        prompt_fusion (OptConfigType): Prompt fusion config. Default: None.
    """
    OVERLAPPING = None

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 inference_sam: bool = False,
                 use_task_router: bool = True,
                 task_router: OptConfigType = None,
                 use_streaming_memory: bool = True,
                 streaming_memory: OptConfigType = None,
                 use_prompt_fusion: bool = True,
                 prompt_fusion: OptConfigType = None,
                 init_cfg: OptMultiConfig = None
                 ):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        # Pass multi-task configs to head
        panoptic_head_.update(
            use_task_router=use_task_router,
            use_streaming_memory=use_streaming_memory,
            use_prompt_fusion=use_prompt_fusion
        )
        self.panoptic_head = MODELS.build(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.alpha = 0.4
        self.beta = 0.8

        self.inference_sam = inference_sam
        
        # Multi-task components
        self.use_task_router = use_task_router
        if use_task_router:
            if task_router is None:
                task_router = dict(type='TaskRouter')
            self.task_router = MODELS.build(task_router)
        else:
            self.task_router = None
        
        self.use_streaming_memory = use_streaming_memory
        if use_streaming_memory:
            if streaming_memory is None:
                streaming_memory = dict(type='StreamingMemoryAdapter')
            self.streaming_memory = MODELS.build(streaming_memory)
        else:
            self.streaming_memory = None
        
        self.use_prompt_fusion = use_prompt_fusion
        if use_prompt_fusion:
            if prompt_fusion is None:
                prompt_fusion = dict(type='PromptFusion')
            self.prompt_fusion = MODELS.build(prompt_fusion)
        else:
            self.prompt_fusion = None
    
    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: SampleList) -> Dict[str, torch.Tensor]:
        """Forward function for training.
        
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W) or 
                (N, T, C, H, W) for video.
            batch_data_samples (list): The batch data samples.
            
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # Detect task type and route
        routing_config = None
        if self.use_task_router:
            # Extract prompts from data samples if available
            prompts = self._extract_prompts_from_samples(batch_data_samples)
            routing_config = self.task_router(
                batch_data_samples, prompts, mode='train'
            )
            # Pass routing config to head
            if hasattr(self.panoptic_head, 'set_routing_config'):
                self.panoptic_head.set_routing_config(routing_config)
        
        # Extract features
        from mmdet.structures import TrackDataSample
        if isinstance(batch_data_samples[0], TrackDataSample):
            bs, num_frames, three, h, w = batch_inputs.shape
            assert three == 3, "Only supporting images with 3 channels."
            x = batch_inputs.reshape((bs * num_frames, three, h, w))
            x = self.extract_feat(x)
        else:
            x = self.extract_feat(batch_inputs)
        
        # Forward through head
        losses = self.panoptic_head.loss(x, batch_data_samples)
        
        # Add task-specific losses (e.g., DPSR for VOS)
        if routing_config and routing_config.get('task_specific_config', {}).get('enable_dpsr'):
            dpsr_loss = self._compute_dpsr_loss(batch_data_samples)
            if dpsr_loss is not None:
                losses['loss_dpsr'] = dpsr_loss
        
        return losses
    
    def predict(self,
                batch_inputs: torch.Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict function for inference.
        
        Args:
            batch_inputs (Tensor): Input images.
            batch_data_samples (list): The batch data samples.
            rescale (bool): Whether to rescale results.
            
        Returns:
            SampleList: Prediction results.
        """
        # Detect task type and route
        routing_config = None
        if self.use_task_router:
            prompts = self._extract_prompts_from_samples(batch_data_samples)
            routing_config = self.task_router(
                batch_data_samples, prompts, mode='test'
            )
            if hasattr(self.panoptic_head, 'set_routing_config'):
                self.panoptic_head.set_routing_config(routing_config)
        
        # Call parent predict
        return super().predict(batch_inputs, batch_data_samples, rescale)
    
    def _extract_prompts_from_samples(self, 
                                     batch_data_samples: SampleList) -> Optional[Dict]:
        """Extract prompts from data samples.
        
        Args:
            batch_data_samples: List of data samples.
            
        Returns:
            Dict containing prompts or None.
        """
        prompts = {}
        
        # Check for point_coords, bboxes, text in data samples
        first_sample = batch_data_samples[0]
        
        # Check gt_instances_collected for training
        if hasattr(first_sample, 'gt_instances_collected') and first_sample.gt_instances_collected is not None:
            if hasattr(first_sample.gt_instances_collected, 'point_coords'):
                prompts['point_coords'] = first_sample.gt_instances_collected.point_coords
        
        # Check metainfo for text
        if hasattr(first_sample, 'metainfo') and 'text' in first_sample.metainfo:
            prompts['text'] = first_sample.metainfo['text']
        
        # Check for bboxes in gt_instances
        if hasattr(first_sample, 'gt_instances') and first_sample.gt_instances is not None:
            if hasattr(first_sample.gt_instances, 'bboxes'):
                prompts['bboxes'] = first_sample.gt_instances.bboxes
        
        return prompts if prompts else None
    
    def _compute_dpsr_loss(self, batch_data_samples: SampleList) -> Optional[torch.Tensor]:
        """Compute Dual-Path Self-Refinement (DPSR) loss for VOS.
        
        Args:
            batch_data_samples: List of data samples.
            
        Returns:
            DPSR loss tensor or None.
        """
        # This is a placeholder - implement based on VOS requirements
        # DPSR loss compares previous frame mask with current prediction
        return None

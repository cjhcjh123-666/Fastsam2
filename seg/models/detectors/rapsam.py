# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

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
        # Note: StreamingMemory and PromptFusion are built in panoptic_head
        # The detector-level instances are kept for reference but not used directly
        # All modules should be registered in panoptic_head to ensure proper device handling
        if use_streaming_memory:
            if streaming_memory is None:
                streaming_memory = dict(type='StreamingMemoryAdapter')
            # Don't build here - it's built in head
            self.streaming_memory = None
        else:
            self.streaming_memory = None
        
        self.use_prompt_fusion = use_prompt_fusion
        # Don't build here - it's built in head
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
        # For RefCOCO dataset with text, we need forward results for text alignment loss
        # Check if we have text in any sample
        has_text = False
        if self.use_prompt_fusion and self.prompt_fusion is not None:
            has_text = any(
                hasattr(ds, 'metainfo') and 'text' in ds.metainfo
                for ds in batch_data_samples
            )
        
        if has_text:
            # Do forward pass first to get visual features for text alignment loss
            all_cls_scores, all_mask_preds, all_iou_preds, _ = \
                self.panoptic_head(x, batch_data_samples)
            forward_results = (all_cls_scores, all_mask_preds, all_iou_preds, None)
            
            # Extract batch_gt_instances and batch_img_metas for loss_by_feat
            # Reuse the same logic as in head's loss method
            batch_img_metas = []
            batch_gt_instances = []
            batch_gt_semantic_segs = []
            
            from mmdet.structures import TrackDataSample
            from mmengine.structures import InstanceData
            
            # Check if prompt_training mode (SAM dataset)
            is_prompt_training = batch_data_samples[0].get('data_tag', 'coco') == 'sam'
            
            if is_prompt_training:
                for data_sample in batch_data_samples:
                    device = data_sample.gt_instances.labels.device
                    ori_masks = data_sample.gt_instances.masks.to_tensor(torch.bool, device)
                    indices = data_sample.gt_instances_collected.idx
                    gt_masks = ori_masks[indices]
                    gt_instances = InstanceData(masks=gt_masks)
                    batch_img_metas.append(data_sample.metainfo)
                    batch_gt_instances.append(gt_instances)
            else:
                for data_sample in batch_data_samples:
                    if isinstance(data_sample, TrackDataSample):
                        clip_meta = []
                        clip_instances = []
                        clip_sem_seg = []
                        for det_sample in data_sample:
                            clip_meta.append(det_sample.metainfo)
                            clip_instances.append(det_sample.gt_instances)
                            if 'gt_sem_seg' in det_sample:
                                clip_sem_seg.append(det_sample.gt_sem_seg)
                            else:
                                clip_sem_seg.append(None)
                        batch_img_metas.append(clip_meta)
                        batch_gt_instances.append(clip_instances)
                        batch_gt_semantic_segs.append(clip_sem_seg)
                    else:
                        batch_img_metas.append(data_sample.metainfo)
                        batch_gt_instances.append(data_sample.gt_instances)
                        if 'gt_sem_seg' in data_sample:
                            batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
                        else:
                            batch_gt_semantic_segs.append(None)
                
                # Preprocess GT if needed
                if hasattr(self.panoptic_head, 'preprocess_gt'):
                    batch_gt_instances = self.panoptic_head.preprocess_gt(
                        batch_gt_instances, batch_gt_semantic_segs
                    )
            
            # Compute loss with forward results
            losses = self.panoptic_head.loss_by_feat(
                all_cls_scores, all_mask_preds, all_iou_preds, 
                batch_gt_instances, batch_img_metas
            )
            
            # Add text-visual alignment loss
            text_alignment_loss = self._compute_text_visual_alignment_loss(
                batch_data_samples, forward_results
            )
            if text_alignment_loss is not None:
                losses['loss_text_align'] = text_alignment_loss
        else:
            # Normal path: no text, use standard loss computation
            losses = self.panoptic_head.loss(x, batch_data_samples)
        
        # Add task-specific losses (e.g., DPSR for VOS)
        if routing_config and routing_config.get('task_specific_config', {}).get('enable_dpsr'):
            # Try to get predictions from head for DPSR loss
            # In practice, we might need to store predictions during forward pass
            prev_predictions = getattr(self, '_prev_predictions', None)
            dpsr_loss = self._compute_dpsr_loss(batch_data_samples, prev_predictions)
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
    
    def _compute_text_visual_alignment_loss(self,
                                            batch_data_samples: SampleList,
                                            forward_results: Tuple) -> Optional[torch.Tensor]:
        """Compute text-visual alignment loss for RefCOCO dataset.
        
        This loss ensures that text embeddings are aligned with visual features
        of the corresponding instances, improving text-guided segmentation.
        
        Args:
            batch_data_samples: List of data samples.
            forward_results: Tuple of (all_cls_scores, all_mask_preds, all_iou_preds, _)
                from head's forward pass.
            
        Returns:
            Text-visual alignment loss tensor or None if not applicable.
        """
        all_cls_scores, all_mask_preds, all_iou_preds, _ = forward_results
        
        # Check if we have text in any sample
        has_text = False
        text_list = []
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'metainfo') and 'text' in data_sample.metainfo:
                text_raw = data_sample.metainfo['text']
                # Convert to string immediately if it's a Tensor
                if isinstance(text_raw, torch.Tensor):
                    if text_raw.dim() == 0:
                        text_clean = str(text_raw.item())
                    elif text_raw.dim() == 1 and text_raw.numel() > 0:
                        # Try to decode if it's token IDs, otherwise convert to string
                        try:
                            # If it looks like token IDs, skip (can't decode without tokenizer)
                            text_clean = None
                        except:
                            text_clean = str(text_raw.tolist())
                    else:
                        text_clean = None
                elif isinstance(text_raw, str):
                    text_clean = text_raw
                elif isinstance(text_raw, (list, tuple)):
                    # If it's a list, try to join if all are strings
                    if all(isinstance(t, str) for t in text_raw):
                        text_clean = ' '.join(text_raw)
                    else:
                        text_clean = None
                else:
                    try:
                        text_clean = str(text_raw)
                    except:
                        text_clean = None
                
                if text_clean is not None and isinstance(text_clean, str) and len(text_clean.strip()) > 0:
                    text_list.append(text_clean)
                    has_text = True
                else:
                    text_list.append(None)
            else:
                text_list.append(None)
        
        if not has_text or self.prompt_fusion is None:
            return None
        
        # Get text encoder from prompt_fusion
        if not hasattr(self.prompt_fusion, 'text_encoder') or self.prompt_fusion.text_encoder is None:
            return None
        
        text_encoder = self.prompt_fusion.text_encoder
        
        # Encode text prompts
        # At this point, text_list should only contain strings or None
        text_embeds = []
        valid_indices = []
        for idx, text in enumerate(text_list):
            if text is not None and isinstance(text, str) and len(text.strip()) > 0:
                try:
                    text_embed = text_encoder([text])  # TextEncoder expects list
                except Exception as e:
                    # Skip if encoding fails (e.g., text is still not a string)
                    continue
                
                if text_embed is not None:
                    text_embeds.append(text_embed)
                    valid_indices.append(idx)
        
        if len(text_embeds) == 0:
            return None
        
        # Stack text embeddings: [B_valid, N_text, C] or [B_valid, C]
        if text_embeds[0].dim() == 2:
            # [B, C] -> [B, 1, C]
            text_embeds = [te.unsqueeze(1) for te in text_embeds]
        text_embed = torch.cat(text_embeds, dim=0)  # [B_valid, N_text, C] or [B_valid, 1, C]
        
        # Extract visual features from the last decoder stage
        # Use cls_scores as proxy for visual features (normalized)
        if len(all_cls_scores) > 0:
            last_cls_scores = all_cls_scores[-1]  # [B, N_queries, num_classes]
            # Use normalized mean as visual embedding
            visual_embed = torch.nn.functional.normalize(
                last_cls_scores.mean(dim=-1, keepdim=True), dim=-1
            )  # [B, N_queries, 1]
            
            # Filter to only valid samples (those with text)
            if visual_embed.shape[0] >= len(valid_indices):
                visual_embed = visual_embed[valid_indices]
            else:
                # Pad if needed
                device = visual_embed.device
                pad_size = len(valid_indices) - visual_embed.shape[0]
                pad_embed = torch.zeros(
                    (pad_size, visual_embed.shape[1], visual_embed.shape[2]),
                    device=device
                )
                visual_embed = torch.cat([visual_embed, pad_embed], dim=0)
            
            # Get mask labels for alignment
            mask_labels = []
            for idx in valid_indices:
                data_sample = batch_data_samples[idx]
                if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
                    if hasattr(data_sample.gt_instances, 'masks'):
                        masks = data_sample.gt_instances.masks
                        # Convert to tensor if needed
                        from mmdet.structures.mask import BitmapMasks
                        if isinstance(masks, BitmapMasks):
                            mask_tensor = masks.to_tensor(dtype=torch.float32, device=visual_embed.device)
                        else:
                            mask_tensor = masks.to(device=visual_embed.device)
                        mask_labels.append(mask_tensor)
            
            if len(mask_labels) > 0:
                # Stack mask labels - use mean pooling for simplicity
                mask_label_tensors = []
                for m in mask_labels:
                    if m.dim() == 3:  # [N, H, W]
                        # Use mean across instances
                        mask_label_tensors.append(m.mean(dim=0))  # [H, W]
                    elif m.dim() == 2:  # [H, W]
                        mask_label_tensors.append(m)
                    else:
                        mask_label_tensors.append(m.squeeze(0))
                
                # Stack to [B, H, W]
                try:
                    mask_labels_tensor = torch.stack(mask_label_tensors)  # [B, H, W]
                    
                    # Compute alignment loss
                    alignment_loss = self.prompt_fusion.compute_text_visual_alignment_loss(
                        text_embed=text_embed,
                        visual_embed=visual_embed,
                        mask_labels=mask_labels_tensor
                    )
                    
                    # Apply weight (can be configured, default 0.5)
                    alignment_loss = alignment_loss * 0.5
                    
                    return alignment_loss
                except Exception as e:
                    # If shapes don't match, skip this loss
                    return None
        
        return None
    
    def _compute_dpsr_loss(self, batch_data_samples: SampleList, 
                          prev_predictions: Optional[Dict] = None) -> Optional[torch.Tensor]:
        """Compute Dual-Path Self-Refinement (DPSR) loss for VOS.
        
        DPSR loss enforces temporal consistency between consecutive frames:
        1. Mask consistency: Previous frame mask should be similar to current prediction
        2. Feature consistency: Instance embeddings should be stable across frames
        
        Args:
            batch_data_samples: List of data samples (TrackDataSample for video).
            prev_predictions: Optional dict containing previous frame predictions.
                If None, uses ground truth for training.
                Format: {
                    'masks': List of previous frame masks,
                    'embeddings': List of previous frame embeddings,
                    'instance_ids': List of instance IDs
                }
            
        Returns:
            DPSR loss tensor or None if not applicable.
        """
        from mmdet.structures import TrackDataSample
        import torch.nn.functional as F
        
        # Only compute for video data (TrackDataSample)
        if not batch_data_samples or not isinstance(batch_data_samples[0], TrackDataSample):
            return None
        
        total_mask_loss = 0.0
        total_feat_loss = 0.0
        num_frames_with_loss = 0
        
        for batch_idx, track_sample in enumerate(batch_data_samples):
            if not hasattr(track_sample, 'video_data_samples'):
                continue
            
            video_samples = track_sample.video_data_samples
            if len(video_samples) < 2:
                continue  # Need at least 2 frames
            
            # Get previous predictions if available
            prev_masks_pred = None
            prev_embeds_pred = None
            if prev_predictions is not None:
                prev_masks_pred = prev_predictions.get('masks', {}).get(batch_idx, None)
                prev_embeds_pred = prev_predictions.get('embeddings', {}).get(batch_idx, None)
            
            # Process consecutive frame pairs
            for frame_idx in range(1, len(video_samples)):
                prev_sample = video_samples[frame_idx - 1]
                curr_sample = video_samples[frame_idx]
                
                # Get instance IDs for tracking
                if not hasattr(prev_sample, 'gt_instances') or not hasattr(curr_sample, 'gt_instances'):
                    continue
                
                prev_instances = prev_sample.gt_instances
                curr_instances = curr_sample.gt_instances
                
                if not hasattr(prev_instances, 'instances_ids') or not hasattr(curr_instances, 'instances_ids'):
                    continue
                
                prev_ids = prev_instances.instances_ids
                curr_ids = curr_instances.instances_ids
                
                # Match instances by ID
                common_ids = set(prev_ids.cpu().numpy()) & set(curr_ids.cpu().numpy())
                if not common_ids:
                    continue
                
                # Compute mask consistency loss
                # Use predictions if available, otherwise use ground truth
                prev_masks = None
                curr_masks = None
                
                if prev_masks_pred is not None and frame_idx - 1 < len(prev_masks_pred):
                    # Use predicted masks from previous frame
                    prev_masks = prev_masks_pred[frame_idx - 1]
                elif hasattr(prev_instances, 'masks'):
                    prev_masks = prev_instances.masks
                
                if hasattr(curr_instances, 'masks'):
                    curr_masks = curr_instances.masks
                
                if prev_masks is not None and curr_masks is not None:
                    # Compute mask consistency loss (Dice loss)
                    for inst_id in common_ids:
                        prev_idx = (prev_ids == inst_id).nonzero(as_tuple=True)[0]
                        curr_idx = (curr_ids == inst_id).nonzero(as_tuple=True)[0]
                        
                        if len(prev_idx) > 0 and len(curr_idx) > 0:
                            # Convert BitmapMasks to tensor if needed
                            from mmdet.structures.mask import BitmapMasks
                            if isinstance(prev_masks, BitmapMasks):
                                prev_mask = prev_masks[prev_idx[0]].to_tensor(dtype=torch.float32, device=prev_ids.device)
                            elif isinstance(prev_masks, torch.Tensor):
                                prev_mask = prev_masks[prev_idx[0]].float()
                            else:
                                prev_mask = torch.tensor(prev_masks[prev_idx[0]], dtype=torch.float32, device=prev_ids.device)
                            
                            if isinstance(curr_masks, BitmapMasks):
                                curr_mask = curr_masks[curr_idx[0]].to_tensor(dtype=torch.float32, device=curr_ids.device)
                            elif isinstance(curr_masks, torch.Tensor):
                                curr_mask = curr_masks[curr_idx[0]].float()
                            else:
                                curr_mask = torch.tensor(curr_masks[curr_idx[0]], dtype=torch.float32, device=curr_ids.device)
                            
                            # Ensure same spatial size
                            if prev_mask.shape != curr_mask.shape:
                                prev_mask = F.interpolate(
                                    prev_mask.unsqueeze(0).unsqueeze(0),
                                    size=curr_mask.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0).squeeze(0)
                            
                            # Dice loss
                            intersection = (prev_mask * curr_mask).sum()
                            union = prev_mask.sum() + curr_mask.sum()
                            dice = 2.0 * intersection / (union + 1e-7)
                            mask_loss = 1.0 - dice
                            
                            total_mask_loss += mask_loss
                            
                            # Feature consistency loss (if embeddings available)
                            if prev_embeds_pred is not None and frame_idx - 1 < len(prev_embeds_pred):
                                prev_embed = prev_embeds_pred[frame_idx - 1]
                                if prev_idx[0] < prev_embed.shape[0]:
                                    # Get current frame embeddings (would need to be passed in)
                                    # For now, we compute based on mask similarity
                                    # In practice, this would use actual instance embeddings
                                    feat_loss = 0.0  # Placeholder
                                    total_feat_loss += feat_loss
                            
                            num_frames_with_loss += 1
        
        if num_frames_with_loss > 0:
            # Combine mask and feature losses
            mask_loss = total_mask_loss / num_frames_with_loss
            feat_loss = total_feat_loss / num_frames_with_loss if total_feat_loss > 0 else 0.0
            total_loss = mask_loss + 0.1 * feat_loss  # Weight feature loss less
            return total_loss
        
        return None

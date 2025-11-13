# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmdet.models import Mask2FormerTransformerDecoder
from mmengine.dist import get_dist_info
from mmengine.model import caffe2_xavier_init, ModuleList
from mmengine.structures import InstanceData, PixelData
from torch import Tensor
from mmdet.models.layers import MLP, inverse_sigmoid
from mmdet.models.layers import coordinate_to_encoding
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList, TrackDataSample
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig, reduce_mean)
from mmdet.models.layers import SinePositionalEncoding3D
from mmdet.models.utils import multi_apply, preprocess_panoptic_gt, get_uncertain_point_coords_with_randomness
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from seg.models.necks import SAMPromptEncoder
from seg.models.utils import (
    preprocess_video_panoptic_gt, mask_pool,
    PromptFusion, StreamingMemoryAdapter
)

from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

from .mask2former_vid import Mask2FormerVideoHead
from .yoso_head import CrossAttenHead, KernelUpdator

@MODELS.register_module()
class RapSAMVideoHead(Mask2FormerVideoHead):

    def __init__(self,
                 frozen_head=False,
                 frozen_pred=False,
                 use_adaptor=False,
                 prompt_with_kernel_updator=False,
                 panoptic_with_kernel_updator=False,
                 num_mask_tokens = 1,
                 num_stages = 3,
                 use_kernel_updator=False,
                 sphere_cls = False,
                 ov_classifier_name = None,
                 temperature=0.1,
                 feat_channels=256,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 loss_cls: ConfigType = None,
                 loss_mask: ConfigType = None,
                 loss_dice: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 matching_whole_map: bool = False,
                 enable_box_query: bool = False,
                 use_task_router: bool = False,
                 use_streaming_memory: bool = False,
                 use_prompt_fusion: bool = False,
                 streaming_memory: OptConfigType = None,
                 prompt_fusion: OptConfigType = None,
                 prompt_encoder: OptConfigType = None,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.prompt_with_kernel_updator = prompt_with_kernel_updator
        self.panoptic_with_kernel_updator = panoptic_with_kernel_updator
        self.use_adaptor = use_adaptor
        
        # Multi-task components
        self.use_task_router = use_task_router
        self.use_streaming_memory = use_streaming_memory
        self.use_prompt_fusion = use_prompt_fusion
        self.routing_config = None  # Will be set by detector via set_routing_config
        
        if use_streaming_memory:
            if streaming_memory is None:
                streaming_memory = dict(type='StreamingMemoryAdapter')
            self.streaming_memory = MODELS.build(streaming_memory)
        else:
            self.streaming_memory = None
        
        if use_prompt_fusion:
            if prompt_fusion is None:
                prompt_fusion = dict(type='PromptFusion')
            self.prompt_fusion_module = MODELS.build(prompt_fusion)
        else:
            self.prompt_fusion_module = None
        
        # SAMPromptEncoder for encoding point/box prompts
        if prompt_encoder is not None:
            self.prompt_encoder = MODELS.build(prompt_encoder)
        else:
            self.prompt_encoder = None

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Embedding(num_mask_tokens, feat_channels)
        self.pb_embedding = nn.Embedding(2, feat_channels)
        self.pos_linear = nn.Linear(2 * feat_channels, feat_channels)

        self.matching_whole_map = matching_whole_map
        self.enable_box_query = enable_box_query

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.num_stages = num_stages
        self.kernels = nn.Embedding(self.num_queries, feat_channels)
        self.mask_heads = nn.ModuleList()
        for _ in range(self.num_stages):
            self.mask_heads.append(CrossAttenHead(
                self.num_classes, self.feat_channels, self.num_queries,
                use_kernel_updator=use_kernel_updator,
                frozen_head=frozen_head, frozen_pred=frozen_pred,
                sphere_cls=sphere_cls,
                ov_classifier_name=ov_classifier_name, with_iou_pred=True))
        self.temperature = temperature

        if use_adaptor:
            cross_attn_cfg = dict(embed_dims=256, batch_first=True, num_heads=8)
            if self.panoptic_with_kernel_updator:
                self.panoptic_attn = KernelUpdator(feat_channels=256)
                self.panoptic_norm = nn.Identity()
                if sphere_cls:
                    cls_embed_dim = self.mask_heads[0].fc_cls.size(0)
                    self.panoptic_cls = nn.Sequential(
                        nn.Linear(feat_channels, cls_embed_dim)
                    )
                else:
                    raise NotImplementedError
                    self.panoptic_cls = nn.Linear(256, self.num_classes+1)
            else:
                self.panoptic_attn = MultiheadAttention(**cross_attn_cfg)
                self.panoptic_norm = nn.LayerNorm(256)
                if sphere_cls:
                    cls_embed_dim = self.mask_heads[0].fc_cls.size(0)
                    self.panoptic_cls = nn.Sequential(
                        nn.Linear(feat_channels, cls_embed_dim)
                    )
                else:
                    raise NotImplementedError
                    self.panoptic_cls = nn.Linear(256, self.num_classes+1)
            
            if self.prompt_with_kernel_updator:
                self.prompt_attn = KernelUpdator(feat_channels=256)
                self.prompt_norm = nn.Identity()
                self.prompt_iou = nn.Linear(256, 1)
            else:
                self.prompt_attn = MultiheadAttention(**cross_attn_cfg)
                self.prompt_norm = nn.LayerNorm(256)
                self.prompt_iou = nn.Linear(256, 1)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        

    def init_weights(self) -> None:
        pass
    
    def set_routing_config(self, routing_config: Optional[Dict]):
        """Set routing configuration from TaskRouter.
        
        Args:
            routing_config: Routing configuration dict from TaskRouter.
        """
        self.routing_config = routing_config
        if routing_config:
            # Update num_stages based on routing config
            if 'num_stages' in routing_config:
                # Note: This is a runtime config, actual stages are already built
                # We can use this to control which stages to use
                pass
    
    def forward(self, x, batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_img_metas = []
        if isinstance(batch_data_samples[0], TrackDataSample):
            for track_sample in batch_data_samples:
                cur_list = []
                for det_sample in track_sample:
                    cur_list.append(det_sample.metainfo)
                batch_img_metas.append(cur_list)
            num_frames = len(batch_img_metas[0])
        else:
            for data_sample in batch_data_samples:
                batch_img_metas.append(data_sample.metainfo)
            num_frames = 0
        bs = len(batch_img_metas)
        
        all_cls_scores = []
        all_masks_preds = []
        all_iou_preds = []
        
        # Apply prompt fusion if enabled and routing config indicates interactive task
        fused_prompts = None
        if self.use_prompt_fusion and self.routing_config:
            task_config = self.routing_config.get('task_specific_config', {})
            if task_config.get('enable_prompt_fusion', False):
                # Extract prompts from data samples
                point_embed, box_embed, text_embed, text_tokens = self._extract_prompt_embeddings(
                    batch_data_samples, batch_img_metas
                )
                # Fuse prompts
                if point_embed is not None or box_embed is not None or text_embed is not None or text_tokens is not None:
                    fused_prompts = self.prompt_fusion_module(
                        point_embed=point_embed,
                        box_embed=box_embed,
                        text=text_tokens,
                        text_embed=text_embed
                    )
        
        if self.prompt_training:
            input_query_label, input_query_bbox, self_attn_mask, mask_dict = self.prepare_for_dn_mo(
                batch_data_samples)
            pos_embed = coordinate_to_encoding(input_query_bbox.sigmoid())
            pos_embed = self.pos_linear(pos_embed)
            object_kernels = input_query_label + pos_embed
        else:
            object_kernels = self.kernels.weight[None].repeat(bs, 1, 1)
            self_attn_mask = None
        mask_features = x
        
        # Apply streaming memory for VOS tasks
        memory_enhanced_features = None
        if self.use_streaming_memory and self.routing_config and num_frames > 1:
            task_config = self.routing_config.get('task_specific_config', {})
            if task_config.get('enable_mask_propagation', False) and self.streaming_memory is not None:
                # For VOS, we process frames sequentially and use memory from previous frames
                # Extract instance embeddings and masks from previous frames for memory update
                if isinstance(batch_data_samples[0], TrackDataSample):
                    # Process each video in the batch
                    for batch_idx, track_sample in enumerate(batch_data_samples):
                        if not hasattr(track_sample, 'video_data_samples'):
                            continue
                        
                        video_samples = track_sample.video_data_samples
                        if len(video_samples) < 2:
                            continue
                        
                        # Process frames sequentially
                        for frame_idx in range(len(video_samples)):
                            frame_id = frame_idx
                            
                            # Get current frame's instance data
                            curr_sample = video_samples[frame_idx]
                            if hasattr(curr_sample, 'gt_instances') and curr_sample.gt_instances is not None:
                                gt_instances = curr_sample.gt_instances
                                
                                # Extract instance embeddings from object_kernels
                                # For current frame, we use the kernels as instance embeddings
                                if frame_idx < num_frames:
                                    # Get kernels for this frame (if available)
                                    frame_kernels = object_kernels[batch_idx:batch_idx+1]  # [1, N, C]
                                    
                                    # Get masks if available (from previous predictions or GT)
                                    if hasattr(gt_instances, 'masks'):
                                        masks = gt_instances.masks
                                    else:
                                        # Use predicted masks if available
                                        masks = None
                                    
                                    # Get instance IDs for tracking
                                    instance_ids = None
                                    if hasattr(gt_instances, 'instances_ids'):
                                        instance_ids = gt_instances.instances_ids
                                    
                                    # Update memory with current frame
                                    if masks is not None and instance_ids is not None:
                                        # Extract per-instance embeddings
                                        from mmdet.structures.mask import BitmapMasks
                                        
                                        for inst_idx, inst_id in enumerate(instance_ids):
                                            if inst_idx < frame_kernels.shape[1]:
                                                inst_embed = frame_kernels[0, inst_idx:inst_idx+1, :]  # [1, C]
                                                
                                                # Extract mask and convert to tensor if needed
                                                if isinstance(masks, BitmapMasks):
                                                    # BitmapMasks: extract single mask and convert to tensor
                                                    inst_mask = masks[inst_idx:inst_idx+1].to_tensor(
                                                        dtype=torch.float32, 
                                                        device=inst_embed.device
                                                    )  # [1, H, W]
                                                elif isinstance(masks, torch.Tensor):
                                                    # Already a tensor
                                                    inst_mask = masks[inst_idx:inst_idx+1]  # [1, H, W]
                                                else:
                                                    # Try to get mask directly
                                                    inst_mask = masks[inst_idx:inst_idx+1]
                                                
                                                # Update memory
                                                self.streaming_memory.update(
                                                    frame_id=frame_id,
                                                    instance_embed=inst_embed,
                                                    mask=inst_mask,
                                                    instance_id=inst_id.item() if isinstance(inst_id, torch.Tensor) else inst_id
                                                )
                            
                            # Fetch memory for current frame to enhance features
                            if frame_idx > 0:  # Skip first frame
                                # Fetch memory for all instances in current frame
                                if hasattr(curr_sample, 'gt_instances') and curr_sample.gt_instances is not None:
                                    gt_instances = curr_sample.gt_instances
                                    if hasattr(gt_instances, 'instances_ids'):
                                        instance_ids = gt_instances.instances_ids
                                        
                                        # Fetch memory for each instance
                                        memory_embeds_list = []
                                        memory_masks_list = []
                                        
                                        for inst_id in instance_ids:
                                            mem_embed, mem_mask = self.streaming_memory.fetch(
                                                frame_id=frame_id,
                                                instance_id=inst_id.item() if isinstance(inst_id, torch.Tensor) else inst_id
                                            )
                                            if mem_embed is not None and mem_mask is not None:
                                                memory_embeds_list.append(mem_embed)
                                                memory_masks_list.append(mem_mask)
                                        
                                        # Fuse memory with current features
                                        if memory_embeds_list and memory_masks_list:
                                            # Stack memory embeddings and masks
                                            memory_embeds = torch.cat(memory_embeds_list, dim=0)  # [N, C]
                                            memory_masks = torch.cat(memory_masks_list, dim=0)  # [N, H, W]
                                            
                                            # Enhance mask_features with memory
                                            # Reshape memory masks to match mask_features spatial size
                                            if memory_enhanced_features is None:
                                                memory_enhanced_features = mask_features.clone()
                                            
                                            # Use memory to enhance features via attention-like mechanism
                                            # Simple version: weighted combination
                                            for inst_idx in range(len(memory_embeds_list)):
                                                mem_embed = memory_embeds[inst_idx:inst_idx+1]  # [1, C]
                                                mem_mask = memory_masks[inst_idx:inst_idx+1]  # [1, H, W]
                                                
                                                # Resize memory mask to match feature size
                                                if mem_mask.shape[-2:] != mask_features.shape[-2:]:
                                                    mem_mask = F.interpolate(
                                                        mem_mask.unsqueeze(0),
                                                        size=mask_features.shape[-2:],
                                                        mode='bilinear',
                                                        align_corners=False
                                                    ).squeeze(0)
                                                
                                                # Enhance features with memory (simple weighted fusion)
                                                # In practice, this could be more sophisticated (e.g., cross-attention)
                                                memory_weight = 0.3  # Weight for memory contribution
                                                mem_feat = (mem_embed.unsqueeze(-1).unsqueeze(-1) * 
                                                           mem_mask.unsqueeze(0))  # [1, C, H, W]
                                                
                                                # Add memory-enhanced features
                                                if batch_idx < memory_enhanced_features.shape[0]:
                                                    memory_enhanced_features[batch_idx:batch_idx+1] = (
                                                        memory_enhanced_features[batch_idx:batch_idx+1] * (1 - memory_weight) +
                                                        mem_feat * memory_weight
                                                    )
        
        # Use memory-enhanced features if available
        if memory_enhanced_features is not None:
            mask_features = memory_enhanced_features
        
        if num_frames > 0: # (bs*num_frames, c, h, w) -> (bs, c, num_frames*h, w)
            mask_features = mask_features.unflatten(0, (bs, num_frames))
            mask_features = mask_features.transpose(1, 2).flatten(2, 3)
        
        mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
        for stage in range(self.num_stages):
            mask_head = self.mask_heads[stage]
            cls_scores, mask_preds, iou_preds, object_kernels = mask_head(
                mask_features, object_kernels, mask_preds, self_attn_mask)
            cls_scores = cls_scores / self.temperature
            all_iou_preds.append(iou_preds)
            all_cls_scores.append(cls_scores)
            if num_frames > 0: 
                #(bs,num_query, num_frames*h, w) --> (bs,num_query,num_frames,h,w)
                all_masks_preds.append(mask_preds.unflatten(2, (num_frames, -1)))
            else:
                all_masks_preds.append(mask_preds)
        
        if self.use_adaptor:
            keys = mask_features.flatten(2).transpose(1, 2).contiguous()
            if not self.prompt_training:
                if self.panoptic_with_kernel_updator:
                    hard_sigmoid_masks = (mask_preds.sigmoid() > 0.5).float()
                    f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, mask_features)
                    object_kernels = self.panoptic_attn(f, object_kernels)
                    object_kernels = self.panoptic_norm(object_kernels)
                    mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                else:
                    object_kernels = self.panoptic_attn(object_kernels, keys)
                    object_kernels = self.panoptic_norm(object_kernels)
                    mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                cls_embd = self.panoptic_cls(object_kernels)
                cls_scores = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.mask_heads[0].fc_cls)
                cls_scores = cls_scores.max(-1).values
                cls_scores = self.mask_heads[0].logit_scale.exp() * cls_scores
                
                if num_frames > 0: 
                    all_masks_preds.append(mask_preds.unflatten(2, (num_frames, -1)))
                else:
                    all_masks_preds.append(mask_preds)
                all_cls_scores.append(cls_scores)
                all_iou_preds.append(all_iou_preds[-1])
            else:
                if self.prompt_with_kernel_updator:
                    hard_sigmoid_masks = (mask_preds.sigmoid() > 0.5).float()
                    f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, mask_features)
                    object_kernels = self.prompt_attn(f, object_kernels)
                    object_kernels = self.prompt_norm(object_kernels)
                    iou_preds = self.prompt_iou(object_kernels)
                    mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                else:
                    object_kernels = self.prompt_attn(object_kernels, keys)
                    object_kernels = self.prompt_norm(object_kernels)
                    iou_preds = self.prompt_iou(object_kernels)
                    mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                if num_frames > 0: 
                    all_masks_preds.append(mask_preds.unflatten(2, (num_frames, -1)))
                else:
                    all_masks_preds.append(mask_preds)
                all_cls_scores.append(all_cls_scores[-1])
                all_iou_preds.append(iou_preds)
        return all_cls_scores, all_masks_preds, all_iou_preds, object_kernels

    def _loss_by_feat_single(self, cls_scores, mask_preds, iou_preds, batch_gt_instances, batch_img_metas):
        batch_size, num_ins = cls_scores.size(0), cls_scores.size(1)
        if self.prompt_training:
            num_imgs = mask_preds.size(0)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
            mask_targets = torch.cat([item.masks for item in batch_gt_instances])
            mask_weights = mask_targets.new_ones((batch_size, num_ins), dtype=torch.float)
            avg_factor = cls_scores.size(1)

            num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
            num_total_masks = max(num_total_masks, 1)

            mask_preds = mask_preds[mask_weights > 0]

            if mask_targets.shape[0] == 0:
                # zero match
                loss_dice = mask_preds.sum()
                loss_mask = mask_preds.sum()
                loss_iou = loss_dice.sum() * 0.0
                loss_cls = cls_scores.sum() * 0.0
                return loss_cls, loss_mask, loss_dice, loss_iou

            with torch.no_grad():
                points_coords = get_uncertain_point_coords_with_randomness(
                    mask_preds.unsqueeze(1), None, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                mask_point_targets = point_sample(
                    mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)

            mask_point_preds = point_sample(mask_preds.unsqueeze(1),
                                            points_coords).squeeze(1)

            # dice loss
            loss_mask = self.loss_mask(mask_point_preds,
                                    mask_point_targets,
                                    reduction_override='none').mean(1)
            loss_dice = self.loss_dice(mask_point_preds,
                                    mask_point_targets,
                                    reduction_override='none')

            iou_preds = iou_preds.flatten()  # (bs, 60, 6) --> (bs, 360)
            iou_target = 1 - (loss_dice / self.loss_dice.loss_weight)
            loss_iou = F.mse_loss(iou_preds, iou_target, reduction="none")
            loss_mask = loss_mask.sum() / num_total_masks
            loss_dice = loss_dice.sum() / num_total_masks
            loss_iou = loss_iou.sum() / num_total_masks * 10.0

            loss_cls = cls_scores.sum() * 0.0 + self.kernels.weight.sum() * 0.0
            if self.use_adaptor:
                for n, p in self.panoptic_attn.named_parameters():
                    loss_cls += p.sum() * 0.0
                for n, p in self.panoptic_norm.named_parameters():
                    loss_cls += p.sum() * 0.0
                for n, p in self.panoptic_cls.named_parameters():
                    loss_cls += p.sum() * 0.0
            return loss_cls, loss_mask, loss_dice, loss_iou
        else:
            cls_scores_list = [cls_scores[i] for i in range(batch_size)]
            mask_preds_list = [mask_preds[i] for i in range(batch_size)]
            labels_list, label_weights_list, mask_targets_list, mask_weights_list, avg_factor = \
                self.get_targets(cls_scores_list, mask_preds_list, batch_gt_instances, batch_img_metas)
            labels = torch.stack(labels_list, dim=0)
            label_weights = torch.stack(label_weights_list, dim=0)
            mask_targets = torch.cat(mask_targets_list, dim=0)
            mask_weights = torch.stack(mask_weights_list, dim=0)

        
            # classification loss
            # shape (batch_size * num_queries, )
            cls_scores = cls_scores.flatten(0, 1)
            labels = labels.flatten(0, 1)
            label_weights = label_weights.flatten(0, 1)
            class_weight = cls_scores.new_tensor(self.class_weight)
            ignore_inds = labels.eq(-1.)
            # zero will not be involved in the loss cal
            labels[ignore_inds] = 0
            label_weights[ignore_inds] = 0.
        
            loss_cls = self.loss_cls(
                cls_scores,
                labels,
                label_weights,
                # avg_factor=cls_avg_factor
                avg_factor=class_weight[labels].sum()
            )
        
            # loss_mask
            num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
            num_total_masks = max(num_total_masks, 1)
            # extract positive ones
            # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
            mask_preds = mask_preds[mask_weights > 0]

            if mask_targets.shape[0] == 0:
                # zero match
                loss_dice = mask_preds.sum()
                loss_mask = mask_preds.sum()
                loss_iou = iou_preds.sum() * 0.0
                loss_iou += (self.mask_tokens.weight.sum() + self.pb_embedding.weight.sum()) * 0.0
                loss_iou += (self.pos_linear.weight.sum() + self.pos_linear.bias.sum()) * 0.0
                if self.use_adaptor:
                    for n, p in self.prompt_attn.named_parameters():
                        loss_iou += p.sum() * 0.0
                    for n, p in self.prompt_norm.named_parameters():
                        loss_iou += p.sum() * 0.0
                    for n, p in self.prompt_iou.named_parameters():
                        loss_iou += p.sum() * 0.0
                return loss_cls, loss_mask, loss_dice, loss_iou

            with torch.no_grad():
                points_coords = get_uncertain_point_coords_with_randomness(
                    mask_preds.unsqueeze(1), None, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
                mask_point_targets = point_sample(
                    mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
            # shape (num_queries, h, w) -> (num_queries, num_points)
            mask_point_preds = point_sample(
                mask_preds.unsqueeze(1), points_coords).squeeze(1)
            # dice loss
            loss_dice = self.loss_dice(
                mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

            # mask loss
            # shape (num_queries, num_points) -> (num_queries * num_points, )
            mask_point_preds = mask_point_preds.reshape(-1)
            # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
            mask_point_targets = mask_point_targets.reshape(-1)
            loss_mask = self.loss_mask(
                mask_point_preds,
                mask_point_targets,
                avg_factor=num_total_masks * self.num_points
            )
            loss_iou = iou_preds.sum() * 0.0
            loss_iou += (self.mask_tokens.weight.sum() + self.pb_embedding.weight.sum()) * 0.0
            loss_iou += (self.pos_linear.weight.sum() + self.pos_linear.bias.sum()) * 0.0
            if self.use_adaptor:
                for n, p in self.prompt_attn.named_parameters():
                    loss_iou += p.sum() * 0.0
                for n, p in self.prompt_norm.named_parameters():
                    loss_iou += p.sum() * 0.0
                for n, p in self.prompt_iou.named_parameters():
                    loss_iou += p.sum() * 0.0
            
            # Add task-specific losses if needed (e.g., DPSR for VOS)
            if self.routing_config:
                task_config = self.routing_config.get('task_specific_config', {})
                if task_config.get('enable_dpsr', False):
                    # DPSR loss would be computed here
                    # For now, it's handled in the detector's loss method
                    pass
            
            return loss_cls, loss_mask, loss_dice, loss_iou
    
    def _extract_prompt_embeddings(self,
                                   batch_data_samples: SampleList,
                                   batch_img_metas: List) -> Tuple:
        """Extract prompt embeddings from data samples for prompt fusion.
        
        Args:
            batch_data_samples: List of data samples.
            batch_img_metas: List of image metadata.
            
        Returns:
            Tuple of (point_embed, box_embed, text_embed, text_tokens), each can be None.
        """
        point_embed = None
        box_embed = None
        text_embed = None
        text_tokens = None
        
        # Extract from first sample as example
        if not batch_data_samples:
            return point_embed, box_embed, text_embed, text_tokens
        
        # Collect all prompts from batch
        batch_point_coords = []
        batch_point_labels = []
        batch_bboxes = []
        batch_texts = []
        batch_image_sizes = []
        
        for i, data_sample in enumerate(batch_data_samples):
            # Get image size from metadata
            if i < len(batch_img_metas) and 'img_shape' in batch_img_metas[i]:
                img_shape = batch_img_metas[i]['img_shape']
                image_size = (img_shape[0], img_shape[1])
            else:
                image_size = (1024, 1024)  # Default size
            batch_image_sizes.append(image_size)
            
            # Extract point prompts
            if hasattr(data_sample, 'gt_instances_collected') and \
               data_sample.gt_instances_collected is not None:
                inst_collected = data_sample.gt_instances_collected
                if hasattr(inst_collected, 'point_coords') and inst_collected.point_coords is not None:
                    point_coords = inst_collected.point_coords
                    # Get point labels if available
                    if hasattr(inst_collected, 'pb_labels'):
                        point_labels = inst_collected.pb_labels
                    else:
                        point_labels = torch.ones(len(point_coords), dtype=torch.long, 
                                                   device=point_coords.device)
                    batch_point_coords.append(point_coords)
                    batch_point_labels.append(point_labels)
            
            # Extract box prompts
            if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
                gt_instances = data_sample.gt_instances
                if hasattr(gt_instances, 'bboxes') and len(gt_instances.bboxes) > 0:
                    bboxes = gt_instances.bboxes
                    batch_bboxes.append(bboxes)
            
            # Extract text prompts
            if hasattr(data_sample, 'metainfo') and 'text' in data_sample.metainfo:
                text_str = data_sample.metainfo['text']
                batch_texts.append(text_str)
        
        # Encode prompts using SAMPromptEncoder if available
        if self.prompt_encoder is not None:
            # Process point and box prompts
            if batch_point_coords or batch_bboxes:
                # Encode using SAMPromptEncoder
                # We need to process each sample separately due to different image sizes
                point_embeds_list = []
                box_embeds_list = []
                
                for i in range(len(batch_data_samples)):
                    image_size = batch_image_sizes[i]
                    has_points = i < len(batch_point_coords) and batch_point_coords[i] is not None
                    has_boxes = i < len(batch_bboxes) and batch_bboxes[i] is not None
                    
                    if has_points or has_boxes:
                        # Set point_coords if available
                        if has_points:
                            point_coords = batch_point_coords[i]
                            
                            # Normalize point_coords shape to [N, num_points, 2]
                            # point_coords can be:
                            # - [num_points, 2]: single point per instance -> [1, num_points, 2]
                            # - [num_points, 4]: two points per instance (x1,y1,x2,y2) -> reshape to [1, num_points*2, 2]
                            # - [1, num_points, 2]: already correct format
                            # - [1, num_points, 4]: two points per instance -> reshape to [1, num_points*2, 2]
                            
                            if point_coords.dim() == 2:
                                # [num_points, 2] or [num_points, 4]
                                if point_coords.shape[1] == 4:
                                    # Two points per instance: [num_points, 4] -> [num_points, 2, 2] -> [num_points*2, 2]
                                    num_points = point_coords.shape[0]
                                    point_coords = point_coords.view(num_points, 2, 2).view(-1, 2)  # [num_points*2, 2]
                                # Add batch dimension: [num_points, 2] -> [1, num_points, 2]
                                point_coords = point_coords.unsqueeze(0)  # [1, num_points, 2] or [1, num_points*2, 2]
                            elif point_coords.dim() == 3:
                                # [1, num_points, 2] or [1, num_points, 4]
                                if point_coords.shape[2] == 4:
                                    # Two points per instance: [1, num_points, 4] -> [1, num_points, 2, 2] -> [1, num_points*2, 2]
                                    num_points = point_coords.shape[1]
                                    point_coords = point_coords.view(1, num_points, 2, 2).view(1, -1, 2)  # [1, num_points*2, 2]
                                # Already has batch dimension
                            else:
                                # Unexpected shape, skip points
                                has_points = False
                                point_coords = None
                            
                            if has_points:
                                # Create InstanceData with correct point_coords
                                point_inst = InstanceData(point_coords=point_coords)
                                
                                # Encode points
                                if has_boxes:
                                    # If both exist, encode separately
                                    point_sparse, _ = self.prompt_encoder(
                                        point_inst, image_size,
                                        with_points=True,
                                        with_bboxes=False,
                                        with_masks=False
                                    )
                                    point_embeds_list.append(point_sparse)
                                else:
                                    # Only points
                                    point_sparse, _ = self.prompt_encoder(
                                        point_inst, image_size,
                                        with_points=True,
                                        with_bboxes=False,
                                        with_masks=False
                                    )
                                    point_embeds_list.append(point_sparse)
                        
                        # Set bboxes if available
                        if has_boxes:
                            bboxes = batch_bboxes[i]
                            # Normalize bboxes shape to [N, 4]
                            if bboxes.dim() == 1:
                                # [4] -> [1, 4]
                                bboxes = bboxes.unsqueeze(0)
                            elif bboxes.dim() == 2:
                                if bboxes.shape[0] == 0:
                                    # Empty bboxes, skip
                                    has_boxes = False
                                elif bboxes.shape[1] != 4:
                                    # Unexpected shape, skip
                                    has_boxes = False
                            else:
                                # Unexpected shape, skip
                                has_boxes = False
                            
                            if has_boxes:
                                # Create InstanceData with correct bboxes
                                box_inst = InstanceData(bboxes=bboxes)
                                
                                # Encode boxes
                                box_sparse, _ = self.prompt_encoder(
                                    box_inst, image_size,
                                    with_points=False,
                                    with_bboxes=True,
                                    with_masks=False
                                )
                                box_embeds_list.append(box_sparse)
                
                # Stack embeddings if available
                # Ensure all samples have embeddings (even if empty) to maintain batch consistency
                batch_size = len(batch_data_samples)
                if point_embeds_list:
                    # Check if we have embeddings for all samples
                    if len(point_embeds_list) == batch_size:
                        point_embed = torch.cat(point_embeds_list, dim=0)  # [B, N, C]
                    else:
                        # Some samples don't have point embeddings, create empty ones
                        # Get feature dimension from first embedding
                        feat_dim = point_embeds_list[0].shape[-1]
                        full_point_embeds = []
                        embed_idx = 0
                        for i in range(batch_size):
                            if i < len(batch_point_coords) and batch_point_coords[i] is not None:
                                full_point_embeds.append(point_embeds_list[embed_idx])
                                embed_idx += 1
                            else:
                                # Create empty embedding [1, 0, C]
                                full_point_embeds.append(torch.empty((1, 0, feat_dim), 
                                                                    device=point_embeds_list[0].device))
                        point_embed = torch.cat(full_point_embeds, dim=0)  # [B, N, C]
                else:
                    point_embed = None
                
                if box_embeds_list:
                    # Check if we have embeddings for all samples
                    if len(box_embeds_list) == batch_size:
                        box_embed = torch.cat(box_embeds_list, dim=0)  # [B, N, C]
                    else:
                        # Some samples don't have box embeddings, create empty ones
                        # Get feature dimension from first embedding
                        feat_dim = box_embeds_list[0].shape[-1]
                        full_box_embeds = []
                        embed_idx = 0
                        for i in range(batch_size):
                            if i < len(batch_bboxes) and batch_bboxes[i] is not None:
                                full_box_embeds.append(box_embeds_list[embed_idx])
                                embed_idx += 1
                            else:
                                # Create empty embedding [1, 0, C]
                                full_box_embeds.append(torch.empty((1, 0, feat_dim), 
                                                                  device=box_embeds_list[0].device))
                        box_embed = torch.cat(full_box_embeds, dim=0)  # [B, N, C]
                else:
                    box_embed = None
        
        # Process text prompts (if text encoder is available in PromptFusion)
        if batch_texts and self.prompt_fusion_module is not None:
            # Text tokens will be processed by TextEncoder in PromptFusion
            # For now, we just pass the text strings
            # The actual tokenization will happen in PromptFusion if TextEncoder is configured
            text_tokens = batch_texts  # Pass as list of strings
        
        return point_embed, box_embed, text_embed, text_tokens

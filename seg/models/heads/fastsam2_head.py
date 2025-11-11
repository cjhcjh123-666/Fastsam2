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
from seg.models.utils import preprocess_video_panoptic_gt, mask_pool

from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

from .mask2former_vid import Mask2FormerVideoHead
from .yoso_head import CrossAttenHead, KernelUpdator

class PromptFusion(nn.Module):
    def __init__(self, q_dim: int, t_dim: int = 512, hidden: int = 256):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(t_dim, hidden), nn.ReLU(True),
                                   nn.Linear(hidden, q_dim))
        self.beta  = nn.Sequential(nn.Linear(t_dim, hidden), nn.ReLU(True),
                                   nn.Linear(hidden, q_dim))

    def forward(self, query_feat: Tensor, z_text: Tensor) -> Tensor:
        # query_feat: [B, Q, C], z_text: [B, D]
        g = self.gamma(z_text).unsqueeze(1)  # [B, 1, C]
        b = self.beta(z_text).unsqueeze(1)   # [B, 1, C]
        return query_feat * (1 + g) + b

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
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.prompt_with_kernel_updator = prompt_with_kernel_updator
        self.panoptic_with_kernel_updator = panoptic_with_kernel_updator
        self.use_adaptor = use_adaptor

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
        
        # -------- Text alignment (lightweight) --------
        # lazy-inited adapter for text encoding
        self.text_adapter = None
        # project mask pooled feature to CLIP text space (default 512-d)
        self.text_proj = nn.Linear(feat_channels, 512, bias=False)
        # loss/logit weights can be overridden via cfg
        self.text_loss_weight = kwargs.get('text_loss_weight', 0.5)
        self.text_logits_weight = kwargs.get('text_logits_weight', 0.3)
        # prompt fusion (lazy build after first forward when dim known)
        self.prompt_fusion = None

    def init_weights(self) -> None:
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
        # cache features for text loss usage
        self._last_mask_features = None
        self._last_num_frames = 0
        if self.prompt_training:
            input_query_label, input_query_bbox, self_attn_mask, mask_dict = self.prepare_for_dn_mo(
                batch_data_samples)
            pos_embed = coordinate_to_encoding(input_query_bbox.sigmoid())
            pos_embed = self.pos_linear(pos_embed)
            object_kernels = input_query_label + pos_embed
        else:
            object_kernels = self.kernels.weight[None].repeat(bs, 1, 1)
            self_attn_mask = None
        # PromptFusion: apply text modulation to queries if available
        try:
            z_text, idxs = self._get_text_embeddings(batch_data_samples)
            if z_text is not None:
                # build batch-aligned z_text [B, D]
                D = z_text.shape[-1]
                z_batch = torch.zeros((bs, D), device=z_text.device, dtype=z_text.dtype)
                for b in range(bs):
                    # use matching index if provided, otherwise clamp
                    z_idx = b if b < z_text.shape[0] else (z_text.shape[0] - 1)
                    z_batch[b] = z_text[z_idx]
                if self.prompt_fusion is None:
                    self.prompt_fusion = PromptFusion(q_dim=object_kernels.shape[-1], t_dim=D)
                    self.prompt_fusion.to(object_kernels.device)
                object_kernels = self.prompt_fusion(object_kernels, z_batch)
        except Exception:
            pass
        mask_features = x
        if num_frames > 0: # (bs*num_frames, c, h, w) -> (bs, c, num_frames*h, w)
            mask_features = mask_features.unflatten(0, (bs, num_frames))
            mask_features = mask_features.transpose(1, 2).flatten(2, 3)
        # cache single/multi frame aware features
        self._last_mask_features = mask_features
        self._last_num_frames = num_frames
        
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

    # -------- helper: ensure text adapter --------
    def _ensure_text_adapter(self):
        if getattr(self, 'text_adapter', None) is not None:
            return
        try:
            # prefer local ext.open_clip if available in repo
            import ext.open_clip  # noqa: F401
        except Exception:
            pass
        from seg.models.utils.text_prompt_adapter import TextPromptAdapter
        self.text_adapter = TextPromptAdapter(model_name='ViT-B-32', pretrained='openai')

    # -------- helper: collect text list and encode --------
    def _get_text_embeddings(self, batch_data_samples):
        texts = []
        idxs = []
        # TrackDataSample or standard DetDataSample handling
        if isinstance(batch_data_samples[0], TrackDataSample):
            # take first frame's text if exists per track sample
            for i, track in enumerate(batch_data_samples):
                t = None
                for det_sample in track:
                    if hasattr(det_sample, 'text') and det_sample.text:
                        t = det_sample.text
                        break
                if t:
                    texts.append(t if isinstance(t, str) else str(t))
                    idxs.append(i)
        else:
            for i, sample in enumerate(batch_data_samples):
                if hasattr(sample, 'text') and sample.text:
                    texts.append(sample.text if isinstance(sample.text, str) else str(sample.text))
                    idxs.append(i)
        if len(texts) == 0:
            return None, None
        self._ensure_text_adapter()
        z = self.text_adapter.encode(texts)  # [M, D], normalized
        return z, idxs

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
            return loss_cls, loss_mask, loss_dice, loss_iou

        # 1) 文本向量获取（单例缓存）
    # 追加全文本对齐损失在 loss() 中实现

    def loss(
            self,
            x: Tuple[Tensor],
            batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        # largely follows Mask2FormerVideoHead.loss with augmentation to add text loss
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
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

        # forward
        all_cls_scores, all_mask_preds, all_iou_preds, _ = self(x, batch_data_samples)

        # preprocess ground truth
        if not self.enable_box_query or batch_data_samples[0].data_tag in ['coco', 'sam']:
            batch_gt_instances = self.preprocess_gt(batch_gt_instances, batch_gt_semantic_segs)

        # video flatten handling
        if isinstance(batch_data_samples[0], TrackDataSample):
            num_frames = len(batch_img_metas[0])
            all_mask_preds = [mask.flatten(2, 3) for mask in all_mask_preds]
            for instance in batch_gt_instances:
                instance['masks'] = instance['masks'].flatten(1, 2)
            film_metas = [
                {
                    'img_shape': (meta[0]['img_shape'][0] * num_frames,
                                  meta[0]['img_shape'][1])
                } for meta in batch_img_metas
            ]
            batch_img_metas = film_metas

        # base losses (cls/mask/dice/optional iou)
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas)

        if self.enable_box_query:
            # zero reg to keep dn params in graph similar to parent impl
            losses['loss_zero'] = 0 * self.kernels.weight.sum() + 0 * self.mask_tokens.weight.sum()
            losses['loss_zero'] += 0 * self.pb_embedding.weight.sum()
            losses['loss_zero'] += 0 * self.pos_linear.weight.sum() + 0 * self.pos_linear.bias.sum()

        # -------- add lightweight text loss on non-video for stability --------
        try:
            # only apply when there is text and single frame path (for simplicity first)
            z_text, idxs = self._get_text_embeddings(batch_data_samples)
            if (z_text is not None) and (getattr(self, '_last_num_frames', 0) == 0):
                # last stage preds [B, Q, H, W]
                mask_pred_last = all_mask_preds[-1]
                # cached mask features [B, C, H, W]
                mask_features = self._last_mask_features
                if mask_features is not None and mask_pred_last.dim() == 4:
                    # pooled features: [B, Q, C]
                    f_mask = mask_pool(mask_features, mask_pred_last)
                    # project to text space and compute per-image sim logits
                    B, Q, C = f_mask.shape
                    f_proj = self.text_proj(f_mask)
                    f_proj = f_proj / (f_proj.norm(dim=-1, keepdim=True) + 1e-6)
                    sim_logits = []
                    for b in range(B):
                        z = z_text[b] if b < z_text.shape[0] else z_text[-1]
                        sim = torch.matmul(f_proj[b], z)  # [Q]
                        sim_logits.append(sim)
                    sim_logits = torch.stack(sim_logits, 0)  # [B, Q]
                    # construct simple positives: use highest mask response as positive per image
                    with torch.no_grad():
                        pos_idx = mask_pred_last.sigmoid().mean(dim=(2, 3)).argmax(dim=1)  # [B]
                        target = torch.zeros_like(sim_logits)
                        target[torch.arange(sim_logits.size(0), device=sim_logits.device), pos_idx] = 1.0
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(sim_logits, target)
                    losses['loss_text'] = bce * float(self.text_loss_weight)
        except Exception:
            # be conservative: do not break training if text branch fails
            pass

        return losses

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                return_query=False,
                ) -> Tuple[Tensor, ...]:
        # largely mirror parent predict() and inject text re-ranking before return
        data_sample = batch_data_samples[0]
        if isinstance(data_sample, TrackDataSample):
            img_shape = data_sample[0].metainfo['batch_input_shape']
            num_frames = len(data_sample)
        else:
            img_shape = data_sample.metainfo['batch_input_shape']
            num_frames = 0
        all_cls_scores, all_mask_preds, all_iou_preds, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        iou_results = all_iou_preds[-1] if (all_iou_preds is not None and len(all_iou_preds) > 0) else None

        if num_frames > 0:
            mask_pred_results = mask_pred_results.flatten(1, 2)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)
        if num_frames > 0:
            num_queries = mask_cls_results.shape[1]
            mask_pred_results = mask_pred_results.unflatten(1, (num_queries, num_frames))

        # inject text-based re-ranking on single frame path
        try:
            z_text, idxs = self._get_text_embeddings(batch_data_samples)
            if z_text is not None:
                mask_features = self._last_mask_features  # single: [B,C,Hf,Wf]; video: [B,C,num_frames*hf,Wf]
                if mask_features is not None:
                    pre_masks = all_mask_preds[-1]
                    if num_frames > 0 and pre_masks.dim() == 5:
                        # video path: select middle keyframe for re-ranking
                        B, Q, T, hf, wf = pre_masks.shape
                        k = T // 2
                        # slice feature window corresponding to keyframe vertical band
                        feat_k = mask_features[:, :, k*hf:(k+1)*hf, :].contiguous()
                        mask_k = pre_masks[:, :, k, :, :].contiguous()
                        f_mask = mask_pool(feat_k, mask_k)  # [B, Q, C]
                    else:
                        # single frame path
                        f_mask = mask_pool(mask_features, pre_masks)  # [B, Q, C]
                    f_proj = self.text_proj(f_mask)
                    f_proj = f_proj / (f_proj.norm(dim=-1, keepdim=True) + 1e-6)
                    B, Q, D = f_proj.shape
                    sim_logits = []
                    for b in range(B):
                        z = z_text[b] if b < z_text.shape[0] else z_text[-1]
                        sim = torch.matmul(f_proj[b], z)  # [Q]
                        sim_logits.append(sim)
                    sim_logits = torch.stack(sim_logits, 0).unsqueeze(-1)  # [B, Q, 1]
                    mask_cls_results = mask_cls_results + self.text_logits_weight * sim_logits
        except Exception:
            pass

        if iou_results is None:
            return mask_cls_results, mask_pred_results

        if return_query:
            # we do not expose query_feat here; keep signature compatible
            return mask_cls_results, mask_pred_results, None, iou_results
        else:
            return mask_cls_results, mask_pred_results, iou_results
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
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from seg.models.necks import SAMPromptEncoder
from seg.models.utils import preprocess_video_panoptic_gt, mask_pool


@MODELS.register_module()
class Mask2FormerVideoHead(AnchorFreeHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_mask_tokens: int = 1,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = None,
                 loss_cls: ConfigType = None,
                 loss_mask: ConfigType = None,
                 loss_dice: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 # ov configs
                 sphere_cls: bool = False,
                 ov_classifier_name: Optional[str] = None,
                 logit: Optional[int] = None,
                 use_adaptor = False,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.use_adaptor = use_adaptor

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Embedding(num_mask_tokens, feat_channels)
        self.pb_embedding = nn.Embedding(2, feat_channels)
        self.pos_linear = nn.Linear(2 * feat_channels, feat_channels)

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        # assert pixel_decoder.encoder.layer_cfg. \
        #            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding3D(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        if not sphere_cls:
            self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.iou_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, 1))

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

        # prepare OV things
        # OV cls embed
        if sphere_cls:
            rank, world_size = get_dist_info()
            if ov_classifier_name is None:
                _dim = 1024  # temporally hard code
                cls_embed = torch.empty(self.num_classes, _dim)
                torch.nn.init.orthogonal_(cls_embed)
                cls_embed = cls_embed[:, None]
            else:
                ov_path = os.path.join(os.path.expanduser('~/.cache/embd'), f"{ov_classifier_name}.pth")
                cls_embed = torch.load(ov_path)
                cls_embed_norm = cls_embed.norm(p=2, dim=-1)
                assert torch.allclose(cls_embed_norm, torch.ones_like(cls_embed_norm))
            if self.loss_cls and self.loss_cls.use_sigmoid:
                pass
            else:
                _dim = cls_embed.size(2)
                _prototypes = cls_embed.size(1)

                if rank == 0:
                    back_token = torch.zeros(1, _dim, dtype=torch.float32, device='cuda')
                    # back_token = back_token / back_token.norm(p=2, dim=-1, keepdim=True)
                else:
                    back_token = torch.empty(1, _dim, dtype=torch.float32, device='cuda')
                if world_size > 1:
                    dist.broadcast(back_token, src=0)
                back_token = back_token.to(device='cpu')
                cls_embed = torch.cat([
                    cls_embed, back_token.repeat(_prototypes, 1)[None]
                ], dim=0)
            self.register_buffer('cls_embed', cls_embed.permute(2, 0, 1).contiguous(), persistent=False)

            # cls embd proj
            cls_embed_dim = self.cls_embed.size(0)
            self.cls_proj = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, cls_embed_dim)
            )

            # For the logit_scale, I refer to this issue.
            # https://github.com/openai/CLIP/issues/46#issuecomment-945062212
            # https://github.com/openai/CLIP/issues/46#issuecomment-782558799
            # Based on my understanding, it is a mistake of CLIP.
            # Because they mention that they refer to InstDisc (Wu, 2018) paper.
            # InstDisc set a non-learnable temperature to np.log(1 / 0.07).
            # 4.6052 is np.log(1 / 0.01)
            # np.log(1 / 0.07) will be fast converged to np.log(1 / 0.01)
            if logit is None:
                logit_scale = torch.tensor(4.6052, dtype=torch.float32)
            else:
                logit_scale = torch.tensor(logit, dtype=torch.float32)
            self.register_buffer('logit_scale', logit_scale, persistent=False)

            # Mask Pooling
            self.mask_pooling = mask_pool
            self.mask_pooling_proj = nn.Sequential(
                nn.LayerNorm(feat_channels),
                nn.Linear(feat_channels, feat_channels)
            )

        if use_adaptor:
            cross_attn_cfg = dict(embed_dims=256, batch_first=True, num_heads=8)
            self.panoptic_attn = MultiheadAttention(**cross_attn_cfg)
            self.panoptic_norm = nn.LayerNorm(256)
            if sphere_cls:
                cls_embed_dim = self.cls_embed.size(0)
                self.panoptic_cls = nn.Sequential(
                    nn.Linear(feat_channels, cls_embed_dim)
                )
            else:
                raise NotImplementedError
            self.prompt_attn = MultiheadAttention(**cross_attn_cfg)
            self.prompt_norm = nn.LayerNorm(256)
            self.prompt_iou = nn.Linear(256, 1)

    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def preprocess_gt(
            self, batch_gt_instances: InstanceList,
            batch_gt_semantic_segs: List[Optional[PixelData]]) -> InstanceList:
        """Preprocess the ground truth for all images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``labels``, each is
                ground truth labels of each bbox, with shape (num_gts, )
                and ``masks``, each is ground truth masks of each instances
                of a image, shape (num_gts, h, w).
            batch_gt_semantic_segs (list[Optional[PixelData]]): Ground truth of
                semantic segmentation, each with the shape (1, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.

        Returns:
            list[obj:`InstanceData`]: each contains the following keys

                - labels (Tensor): Ground truth class indices\
                    for a image, with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (Tensor): Ground truth mask for a\
                    image, with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(batch_gt_instances)
        num_stuff_list = [self.num_stuff_classes] * len(batch_gt_instances)
        if isinstance(batch_gt_instances[0], List):
            gt_labels_list = [
                [torch.stack([torch.ones_like(gt_instances['labels']) * frame_id, gt_instances['labels']], dim=1)
                 for frame_id, gt_instances in enumerate(gt_vid_instances)]
                for gt_vid_instances in batch_gt_instances
            ]
            gt_labels_list = [torch.cat(gt_labels, dim=0) for gt_labels in gt_labels_list]
            gt_masks_list = [
                [gt_instances['masks'] for gt_instances in gt_vid_instances]
                for gt_vid_instances in batch_gt_instances
            ]
            gt_semantic_segs = [
                [None if gt_semantic_seg is None else gt_semantic_seg.sem_seg
                 for gt_semantic_seg in gt_vid_semantic_segs]
                for gt_vid_semantic_segs in batch_gt_semantic_segs
            ]
            if gt_semantic_segs[0][0] is None:
                gt_semantic_segs = [None] * len(batch_gt_instances)
            else:
                gt_semantic_segs = [torch.stack(gt_sem_seg, dim=0) for gt_sem_seg in gt_semantic_segs]
            # 检查是否有instances_ids（VOS任务）
            has_instances_ids = any(
                hasattr(gt_vid_instances[0], 'instances_ids') if gt_vid_instances else False
                for gt_vid_instances in batch_gt_instances
            )
            
            if has_instances_ids:
                # VOS任务：使用instances_ids
                gt_instance_ids_list = [
                    [torch.stack([torch.ones_like(gt_instances['instances_ids']) * frame_id, gt_instances['instances_ids']],
                                 dim=1)
                     for frame_id, gt_instances in enumerate(gt_vid_instances)]
                    for gt_vid_instances in batch_gt_instances
                ]
                gt_instance_ids_list = [torch.cat(gt_instance_ids, dim=0) for gt_instance_ids in gt_instance_ids_list]
            else:
                # 交互视频任务：生成临时的instances_ids
                gt_instance_ids_list = []
                for gt_vid_instances in batch_gt_instances:
                    frame_ids_list = []
                    for frame_id, gt_instances in enumerate(gt_vid_instances):
                        num_instances = len(gt_instances.labels) if hasattr(gt_instances, 'labels') else 0
                        if num_instances > 0:
                            # 创建临时ID: [frame_id, instance_id]
                            instance_ids = torch.arange(num_instances, device=gt_instances.labels.device)
                            frame_ids = torch.stack([
                                torch.ones_like(instance_ids) * frame_id,
                                instance_ids
                            ], dim=1)
                            frame_ids_list.append(frame_ids)
                    if frame_ids_list:
                        gt_instance_ids_list.append(torch.cat(frame_ids_list, dim=0))
                    else:
                        # 空的情况，创建一个空tensor
                        gt_instance_ids_list.append(torch.empty((0, 2), dtype=torch.long, device='cuda'))
            targets = multi_apply(preprocess_video_panoptic_gt, gt_labels_list,
                                  gt_masks_list, gt_semantic_segs, gt_instance_ids_list,
                                  num_things_list, num_stuff_list)
        else:
            gt_labels_list = [
                gt_instances['labels'] for gt_instances in batch_gt_instances
            ]
            gt_masks_list = [
                gt_instances['masks'] for gt_instances in batch_gt_instances
            ]
            gt_semantic_segs = [
                None if gt_semantic_seg is None else gt_semantic_seg.sem_seg
                for gt_semantic_seg in batch_gt_semantic_segs
            ]
            targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                                  gt_masks_list, gt_semantic_segs, num_things_list,
                                  num_stuff_list)
        labels, masks = targets
        batch_gt_instances = [
            InstanceData(labels=label, masks=mask)
            for label, mask in zip(labels, masks)
        ]
        return batch_gt_instances

    def get_targets(
            self,
            cls_scores_list: List[Tensor],
            mask_preds_list: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            return_sampling_results: bool = False
    ) -> Tuple[List[Union[Tensor, int]]]:
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - avg_factor (int): Average factor that is used to average\
                    the loss. When using sampling method, avg_factor is
                    usually the sum of positive and negative priors. When
                    using `MaskPseudoSampler`, `avg_factor` is usually equal
                    to the number of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end.
        """
        results = multi_apply(
            self._get_targets_single, cls_scores_list, mask_preds_list, batch_gt_instances, batch_img_metas
        )
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, \
            pos_inds_list, neg_inds_list, sampling_results_list = results[:7]
        rest_results = list(results[7:])

        avg_factor = sum([results.avg_factor for results in sampling_results_list])
        res = (labels_list, label_weights_list, mask_targets_list, mask_weights_list, avg_factor)

        if return_sampling_results:
            res = res + sampling_results_list

        return res + tuple(rest_results)

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks

        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(mask_pred.unsqueeze(1),
                                        point_coords.repeat(num_queries, 1, 1)).squeeze(1)
        # shape (num_gts, num_points)
        # Ensure gt_masks is a tensor on the correct device
        if hasattr(gt_masks, 'to_tensor'):
            # BitmapMasks or PolygonMasks object
            gt_masks_tensor = gt_masks.to_tensor(dtype=torch.float32, device=cls_score.device)
        elif isinstance(gt_masks, torch.Tensor):
            gt_masks_tensor = gt_masks.to(dtype=torch.float32, device=cls_score.device)
        else:
            # Fallback: try to convert
            gt_masks_tensor = torch.tensor(gt_masks, dtype=torch.float32, device=cls_score.device)
        
        gt_points_masks = point_sample(gt_masks_tensor.unsqueeze(1),
                                        point_coords.repeat(num_gts, 1, 1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta
        )
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((num_queries,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((num_queries,))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((num_queries,))
        mask_weights[pos_inds] = 1.0

        return labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds, sampling_result

    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor, all_iou_preds,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_iou = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds, all_iou_preds, batch_gt_instances_list, img_metas_list
        )

        loss_dict = dict()
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_iou_i in zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        return loss_dict

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor, iou_preds,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        batch_size, num_ins = cls_scores.size(0), cls_scores.size(1)
        if self.prompt_training:
            print(f"\n[DEBUG _loss_by_feat_single] prompt_training=True")
            print(f"[DEBUG] cls_scores.shape={cls_scores.shape}, mask_preds.shape={mask_preds.shape}, iou_preds.shape={iou_preds.shape if iou_preds is not None else None}")
            num_imgs = mask_preds.size(0)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
            mask_targets = torch.cat([item.masks for item in batch_gt_instances])
            print(f"[DEBUG] mask_targets.shape={mask_targets.shape}")
            mask_weights = mask_targets.new_ones((batch_size, num_ins), dtype=torch.float)
            avg_factor = cls_scores.size(1)

            num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
            num_total_masks = max(num_total_masks, 1)

            mask_preds = mask_preds[mask_weights > 0]
            print(f"[DEBUG] mask_preds after filtering shape={mask_preds.shape}")

            if mask_targets.shape[0] == 0:
                # zero match
                loss_dice = mask_preds.sum()
                loss_mask = mask_preds.sum()
                loss_iou = iou_preds.sum() * 0.0
                loss_cls = cls_scores.sum() * 0.0
                if self.use_adaptor:
                    for n,p in self.panoptic_attn.named_parameters():
                        loss_cls += p.sum() * 0.0
                    for n,p in self.panoptic_norm.named_parameters():
                        loss_cls += p.sum() * 0.0
                    for n,p in self.panoptic_cls.named_parameters():
                        loss_cls += p.sum() * 0.0
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

            # 处理iou_preds：确保形状正确
            print(f"[DEBUG] Before iou processing: iou_preds.shape={iou_preds.shape if iou_preds is not None else None}, loss_dice.shape={loss_dice.shape}")
            if iou_preds is not None:
                # iou_preds的形状可能是 (bs, num_queries, 1) 或 (bs, num_queries, num_frames, 1)
                # 需要flatten到 (bs * num_queries,) 或 (bs * num_queries * num_frames,)
                original_shape = iou_preds.shape
                if iou_preds.dim() > 2:
                    # 如果是3D或4D，flatten所有维度除了batch
                    iou_preds = iou_preds.flatten(1)  # (bs, num_queries, ...) -> (bs, num_queries * ...)
                    print(f"[DEBUG] After flatten(1): {original_shape} -> {iou_preds.shape}")
                iou_preds = iou_preds.flatten()  # (bs, num_queries * ...) -> (bs * num_queries * ...)
                print(f"[DEBUG] After final flatten: -> {iou_preds.shape}")
                
                # 确保iou_preds和iou_target的形状匹配
                iou_target = 1 - (loss_dice / self.loss_dice.loss_weight)
                print(f"[DEBUG] iou_target.shape={iou_target.shape}, iou_preds.numel()={iou_preds.numel()}, loss_dice.numel()={loss_dice.numel()}")
                # loss_dice的形状是 (num_masks,)，需要扩展到与iou_preds匹配
                if iou_preds.numel() != loss_dice.numel():
                    # 如果形状不匹配，需要调整
                    # 通常iou_preds应该和mask_preds的数量一致
                    num_masks = mask_preds.shape[0]  # mask_preds已经通过mask_weights > 0过滤
                    print(f"[DEBUG] Shape mismatch! num_masks={num_masks}, iou_preds.numel()={iou_preds.numel()}, loss_dice.numel()={loss_dice.numel()}")
                    if iou_preds.numel() >= num_masks:
                        iou_preds = iou_preds[:num_masks]
                        print(f"[DEBUG] Truncated iou_preds to {iou_preds.shape}")
                    else:
                        # 如果iou_preds数量不够，需要扩展
                        device = iou_preds.device
                        iou_preds = torch.cat([iou_preds, torch.zeros(num_masks - iou_preds.numel(), device=device)])
                        print(f"[DEBUG] Extended iou_preds to {iou_preds.shape}")
                
                loss_iou = F.mse_loss(iou_preds, iou_target, reduction="none")
                print(f"[DEBUG] loss_iou before sum: shape={loss_iou.shape}, mean={loss_iou.mean().item():.6f}, sum={loss_iou.sum().item():.6f}")
            else:
                print(f"[DEBUG] iou_preds is None! Creating zero loss")
                # 如果iou_preds是None，创建零loss但保持梯度流
                device = mask_preds.device
                loss_iou = torch.tensor(0.0, device=device, requires_grad=True)
            loss_mask = loss_mask.sum() / num_total_masks
            loss_dice = loss_dice.sum() / num_total_masks
            loss_iou = loss_iou.sum() / num_total_masks * 10.0

            loss_cls = cls_scores.sum() * 0.0
            loss_cls += (self.query_embed.weight.sum() + self.query_feat.weight.sum()) * 0.0
            if self.use_adaptor:
                for n,p in self.panoptic_attn.named_parameters():
                    loss_cls += p.sum() * 0.0
                for n,p in self.panoptic_norm.named_parameters():
                    loss_cls += p.sum() * 0.0
                for n,p in self.panoptic_cls.named_parameters():
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
            obj_inds = labels.eq(self.num_classes)
        
            loss_cls = self.loss_cls(
                cls_scores,
                labels,
                label_weights,
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
                if iou_preds is not None:
                    loss_iou = iou_preds.sum() * 0.0
                else:
                    loss_iou = mask_preds.new_tensor(0.0, requires_grad=True)
                loss_iou += (self.mask_tokens.weight.sum() + self.pb_embedding.weight.sum()) * 0.0
                loss_iou += (self.pos_linear.weight.sum() + self.pos_linear.bias.sum()) * 0.0
                if self.use_adaptor:
                    for n,p in self.prompt_attn.named_parameters():
                        loss_iou += p.sum() * 0.0
                    for n,p in self.prompt_norm.named_parameters():
                        loss_iou += p.sum() * 0.0
                    for n,p in self.prompt_iou.named_parameters():
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
            
            # VOS和全景分割不需要IoU预测loss（使用cls_score表示置信度）
            # 创建零loss但保持梯度流，避免DDP错误
            loss_iou = iou_preds.sum() * 0.0 if iou_preds is not None else mask_preds.new_tensor(0.0, requires_grad=True)
            loss_iou += (self.mask_tokens.weight.sum() + self.pb_embedding.weight.sum()) * 0.0
            loss_iou += (self.pos_linear.weight.sum() + self.pos_linear.bias.sum()) * 0.0
            if self.use_adaptor:
                for n,p in self.prompt_attn.named_parameters():
                    loss_iou += p.sum() * 0.0
                for n,p in self.prompt_norm.named_parameters():
                    loss_iou += p.sum() * 0.0
                for n,p in self.prompt_iou.named_parameters():
                    loss_iou += p.sum() * 0.0
            
            return loss_cls, loss_mask, loss_dice, loss_iou

    def forward_logit(self, cls_embd):
        cls_pred = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.cls_embed)
        cls_pred = cls_pred.max(-1).values
        cls_pred = self.logit_scale.exp() * cls_pred
        return cls_pred

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int],
                      num_frames: int = 0) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
                - num_frames: How many frames are there in video.
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        if isinstance(self.cls_embed, nn.Module):
            cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        if not isinstance(self.cls_embed, nn.Module):
            maskpool_embd = self.mask_pooling(x=mask_feature, mask=mask_pred.detach())
            maskpool_embd = self.mask_pooling_proj(maskpool_embd)
            cls_embd = self.cls_proj(maskpool_embd + decoder_out)
            cls_pred = self.forward_logit(cls_embd)

        iou_pred = self.iou_embed(decoder_out)

        if num_frames > 0:
            assert len(mask_pred.shape) == 4
            assert mask_pred.shape[2] % num_frames == 0
            frame_h = mask_pred.shape[2] // num_frames
            num_q = mask_pred.shape[1]
            _mask_pred = mask_pred.unflatten(-2, (num_frames, frame_h)).flatten(1, 2)
            attn_mask = F.interpolate(
                _mask_pred,
                attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
            attn_mask = attn_mask.unflatten(1, (num_q, num_frames)).flatten(2, 3)
        else:
            attn_mask = F.interpolate(
                mask_pred,
                attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, iou_pred, attn_mask

    def forward(self, x: List[Tensor], batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
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
        batch_size = len(batch_img_metas)
        #(bs_nf, c, h,w)
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        if num_frames > 0:
            mask_features = mask_features.unflatten(0, (batch_size, num_frames))
            mask_features = mask_features.transpose(1, 2).flatten(2, 3) #(bs, c, nf*h,w)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i]) #(bs_nf, c, h,w)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1) #(bs_nf,h*w, c)
            if num_frames > 0:
                decoder_input = decoder_input.unflatten(0, (batch_size, num_frames))
                decoder_input = decoder_input.flatten(1, 2) #(bs, nf*h*w, c)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            num_frames_real = 1 if num_frames == 0 else num_frames
            mask = decoder_input.new_zeros(
                (batch_size, num_frames_real) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.transpose(
                1, 2).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input) #(bs, nf*h*w, c)
            decoder_positional_encodings.append(decoder_positional_encoding) #(bs, nf*h*w, c)

        if self.prompt_training:
            query_feat, input_query_bbox, self_attn_mask, _ = self.prepare_for_dn_mo(
                batch_data_samples)
            query_embed = coordinate_to_encoding(input_query_bbox.sigmoid())
            query_embed = self.pos_linear(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
            query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))
            self_attn_mask = None

        cls_pred_list = []
        mask_pred_list = []
        iou_pred_list = []
        cls_pred, mask_pred, iou_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:],
            num_frames=num_frames
        )
        cls_pred_list.append(cls_pred)
        iou_pred_list.append(iou_pred)
        if num_frames > 0: #(bs, 100, nf*h, w)-->(bs, 100, nf, h, w)
            mask_pred = mask_pred.unflatten(2, (num_frames, -1))
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat, #(bs, 100, c)
                key=decoder_inputs[level_idx], #(bs, nf*h*w, c)
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, iou_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                num_frames=num_frames
            )

            cls_pred_list.append(cls_pred)
            iou_pred_list.append(iou_pred)
            if num_frames > 0:
                mask_pred = mask_pred.unflatten(2, (num_frames, -1))
            mask_pred_list.append(mask_pred)
        
        if self.use_adaptor:
            keys = mask_features.flatten(2).transpose(1, 2).contiguous()
            h, w = mask_features.shape[-2] // num_frames_real, mask_features.shape[-1]
            mask = decoder_input.new_zeros((batch_size, num_frames_real, h, w), dtype=torch.bool)
            key_pos = self.decoder_positional_encoding(mask)
            key_pos = key_pos.transpose(1, 2).flatten(2).permute(0, 2, 1)
            if not self.prompt_training:
                object_kernels = self.panoptic_attn(query_feat, keys, key_pos=key_pos, query_pos=query_embed)
                object_kernels = self.panoptic_norm(object_kernels)
                mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                
                cls_embd = self.panoptic_cls(object_kernels)
                cls_scores = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.cls_embed)
                cls_scores = cls_scores.max(-1).values
                cls_scores = self.logit_scale.exp() * cls_scores
                
                if num_frames > 0: 
                    mask_pred_list.append(mask_preds.unflatten(2, (num_frames, -1)))
                else:
                    mask_pred_list.append(mask_preds)
                cls_pred_list.append(cls_scores)
                iou_pred_list.append(iou_pred_list[-1])
            else:
                object_kernels = self.prompt_attn(query_feat, keys, key_pos=key_pos, query_pos=query_embed)
                object_kernels = self.prompt_norm(object_kernels)
                iou_preds = self.prompt_iou(object_kernels)
                mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                
                if num_frames > 0: 
                    mask_pred_list.append(mask_preds.unflatten(2, (num_frames, -1)))
                else:
                    mask_pred_list.append(mask_preds)
                cls_pred_list.append(cls_pred_list[-1])
                iou_pred_list.append(iou_preds)

        return cls_pred_list, mask_pred_list, iou_pred_list, query_feat

    def loss(
            self,
            x: Tuple[Tensor],
            batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []

        # 检查是否是交互任务（有gt_instances_collected）
        # 交互任务包括：SAM (data_tag='sam')、RefCOCO (data_tag='refcoco')等
        # 统一通过gt_instances_collected来判断
        first_sample = batch_data_samples[0]
        has_prompt = False
        
        print(f"\n[DEBUG loss()] Checking prompt_training:")
        print(f"[DEBUG] first_sample type: {type(first_sample).__name__}")
        
        if isinstance(first_sample, TrackDataSample):
            # 视频交互任务：检查video_data_samples中的第一帧
            print(f"[DEBUG] TrackDataSample detected")
            if hasattr(first_sample, 'video_data_samples') and len(first_sample.video_data_samples) > 0:
                first_frame = first_sample.video_data_samples[0]
                print(f"[DEBUG] first_frame has gt_instances_collected: {hasattr(first_frame, 'gt_instances_collected')}")
                if hasattr(first_frame, 'gt_instances_collected') and first_frame.gt_instances_collected is not None:
                    has_prompt = True
                    print(f"[DEBUG] ✅ Video interactive task detected (has gt_instances_collected)")
        elif hasattr(first_sample, 'gt_instances_collected') and first_sample.gt_instances_collected is not None:
            # 图像交互任务 (SAM, RefCOCO等)
            has_prompt = True
            print(f"[DEBUG] ✅ Image interactive task detected (has gt_instances_collected)")
        
        self.prompt_training = has_prompt
        print(f"[DEBUG] prompt_training = {self.prompt_training}")

        if self.prompt_training:
            for data_sample in batch_data_samples:
                if isinstance(data_sample, TrackDataSample):
                    # 视频交互任务：处理TrackDataSample
                    clip_meta = []
                    clip_instances = []
                    clip_sem_seg = []
                    for det_sample in data_sample.video_data_samples:
                        clip_meta.append(det_sample.metainfo)
                        # 从gt_instances_collected中获取masks
                        if hasattr(det_sample, 'gt_instances_collected') and det_sample.gt_instances_collected is not None:
                            gt_collected = det_sample.gt_instances_collected
                            
                            # 检查是否有idx属性（用于索引原始masks）
                            if hasattr(gt_collected, 'idx'):
                                # 使用idx索引原始masks（保持BitmapMasks格式）
                                from mmdet.structures.mask import BitmapMasks
                                ori_masks = det_sample.gt_instances.masks
                                indices = gt_collected.idx.cpu().numpy()
                                if isinstance(ori_masks, BitmapMasks):
                                    # 从BitmapMasks中选择
                                    masks_np = ori_masks.masks[indices]
                                    gt_masks = BitmapMasks(masks_np, ori_masks.height, ori_masks.width)
                                else:
                                    # 如果是Tensor，转换为BitmapMasks
                                    masks_tensor = ori_masks[indices] if isinstance(ori_masks, torch.Tensor) else ori_masks.to_tensor()[indices]
                                    masks_np = masks_tensor.cpu().numpy().astype(np.uint8)
                                    gt_masks = BitmapMasks(masks_np, masks_tensor.shape[-2], masks_tensor.shape[-1])
                            elif hasattr(gt_collected, 'masks'):
                                # 直接使用gt_instances_collected中的masks（保持原格式）
                                from mmdet.structures.mask import BitmapMasks
                                if isinstance(gt_collected.masks, BitmapMasks):
                                    gt_masks = gt_collected.masks
                                elif isinstance(gt_collected.masks, torch.Tensor):
                                    # 转换为BitmapMasks
                                    masks_np = gt_collected.masks.cpu().numpy().astype(np.uint8)
                                    gt_masks = BitmapMasks(masks_np, gt_collected.masks.shape[-2], gt_collected.masks.shape[-1])
                                else:
                                    gt_masks = gt_collected.masks
                            else:
                                # 如果没有masks，使用原始gt_instances的masks
                                gt_masks = det_sample.gt_instances.masks
                            
                            gt_instances = InstanceData(masks=gt_masks)
                            # 复制labels和bboxes（如果存在）
                            if hasattr(det_sample.gt_instances, 'labels'):
                                gt_instances.labels = det_sample.gt_instances.labels
                            if hasattr(det_sample.gt_instances, 'bboxes'):
                                gt_instances.bboxes = det_sample.gt_instances.bboxes
                            clip_instances.append(gt_instances)
                        else:
                            # 没有gt_instances_collected，使用原始gt_instances
                            clip_instances.append(det_sample.gt_instances)
                        
                        if 'gt_sem_seg' in det_sample:
                            clip_sem_seg.append(det_sample.gt_sem_seg)
                        else:
                            clip_sem_seg.append(None)
                    batch_img_metas.append(clip_meta)
                    batch_gt_instances.append(clip_instances)
                    batch_gt_semantic_segs.append(clip_sem_seg)
                else:
                    # 图像交互任务：处理DetDataSample
                    gt_collected = data_sample.gt_instances_collected
                    
                    # 检查是否有idx属性（用于索引原始masks）
                    if hasattr(gt_collected, 'idx'):
                        # 使用idx索引原始masks（保持BitmapMasks格式）
                        from mmdet.structures.mask import BitmapMasks
                        import numpy as np
                        ori_masks = data_sample.gt_instances.masks
                        indices = gt_collected.idx.cpu().numpy()
                        if isinstance(ori_masks, BitmapMasks):
                            # 从BitmapMasks中选择
                            masks_np = ori_masks.masks[indices]
                            gt_masks = BitmapMasks(masks_np, ori_masks.height, ori_masks.width)
                        else:
                            # 如果是Tensor，转换为BitmapMasks
                            masks_tensor = ori_masks[indices] if isinstance(ori_masks, torch.Tensor) else ori_masks.to_tensor()[indices]
                            masks_np = masks_tensor.cpu().numpy().astype(np.uint8)
                            gt_masks = BitmapMasks(masks_np, masks_tensor.shape[-2], masks_tensor.shape[-1])
                    elif hasattr(gt_collected, 'masks'):
                        # 直接使用gt_instances_collected中的masks（保持原格式）
                        from mmdet.structures.mask import BitmapMasks
                        import numpy as np
                        if isinstance(gt_collected.masks, BitmapMasks):
                            gt_masks = gt_collected.masks
                        elif isinstance(gt_collected.masks, torch.Tensor):
                            # 转换为BitmapMasks
                            masks_np = gt_collected.masks.cpu().numpy().astype(np.uint8)
                            gt_masks = BitmapMasks(masks_np, gt_collected.masks.shape[-2], gt_collected.masks.shape[-1])
                        else:
                            gt_masks = gt_collected.masks
                    else:
                        # 如果没有masks，使用原始gt_instances的masks
                        gt_masks = data_sample.gt_instances.masks
                    
                    gt_instances = InstanceData(masks=gt_masks)
                    # 复制labels和bboxes（如果存在）
                    if hasattr(data_sample.gt_instances, 'labels'):
                        gt_instances.labels = data_sample.gt_instances.labels
                    if hasattr(data_sample.gt_instances, 'bboxes'):
                        gt_instances.bboxes = data_sample.gt_instances.bboxes
                    batch_img_metas.append(data_sample.metainfo)
                    batch_gt_instances.append(gt_instances)
                    if 'gt_sem_seg' in data_sample:
                        batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
                    else:
                        batch_gt_semantic_segs.append(None)
            # 对于prompt_training，也需要调用preprocess_gt来统一格式
            batch_gt_instances = self.preprocess_gt(batch_gt_instances, batch_gt_semantic_segs)
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
            batch_gt_instances = self.preprocess_gt(batch_gt_instances, batch_gt_semantic_segs)
        # forward
        # 🔥 关键：保存prompt_training的值，因为forward可能会修改它
        saved_prompt_training = self.prompt_training
        print(f"[DEBUG] Before forward(): prompt_training={self.prompt_training}")
        all_cls_scores, all_mask_preds, all_iou_preds, _ = self(x, batch_data_samples)
        # 🔥 恢复prompt_training的值
        self.prompt_training = saved_prompt_training
        print(f"[DEBUG] After forward(): prompt_training={self.prompt_training} (restored)")

        # loss
        if isinstance(batch_data_samples[0], TrackDataSample):
            num_frames = len(batch_img_metas[0])
            print(f"\n[DEBUG] 视频任务 - num_frames={num_frames}, batch_size={len(batch_img_metas)}, prompt_training={self.prompt_training}")
            print(f"[DEBUG] all_mask_preds shapes before flatten: {[m.shape for m in all_mask_preds[:2]]}")
            
            # 🔥 关键：对于prompt_training=True的视频交互任务，需要扩展cls_scores、mask_preds和iou_preds
            # mask_preds的形状是(bs, num_queries, num_frames, h, w)，需要reshape成(bs * num_frames, num_queries, h, w)
            if self.prompt_training:
                # 视频交互任务：扩展cls_scores、mask_preds和iou_preds
                all_cls_scores = [
                    cls.repeat_interleave(num_frames, dim=0) if cls.dim() == 3
                    else cls for cls in all_cls_scores
                ]
                all_mask_preds = [
                    mask.permute(0, 2, 1, 3, 4).flatten(0, 1).flatten(2, 3) if mask.dim() == 5 and mask.shape[2] == num_frames
                    else mask.flatten(2, 3) for mask in all_mask_preds
                ]
                all_iou_preds = [
                    iou.repeat_interleave(num_frames, dim=0) if iou is not None and iou.dim() == 3 and iou.shape[2] == 1
                    else iou for iou in all_iou_preds
                ]
                print(f"[DEBUG] ✅ Extended cls_scores, mask_preds and iou_preds for prompt_training")
                print(f"[DEBUG] all_cls_scores shapes after extension: {[c.shape for c in all_cls_scores[:2]]}")
                print(f"[DEBUG] all_mask_preds shapes after extension: {[m.shape for m in all_mask_preds[:2]]}")
                print(f"[DEBUG] all_iou_preds shapes after extension: {[iou.shape if iou is not None else None for iou in all_iou_preds[:2]]}")
            else:
                # VOS任务：只flatten空间维度，不扩展batch维度
                all_mask_preds = [mask.flatten(2, 3) for mask in all_mask_preds]
                print(f"[DEBUG] all_mask_preds shapes after flatten: {[m.shape for m in all_mask_preds[:2]]}")
                print(f"[DEBUG] ⏭️  Skipped iou_preds expansion for VOS (will be masked)")
            for instance in batch_gt_instances:
                instance['masks'] = instance['masks'].flatten(1, 2)
            film_metas = [
                {
                    'img_shape': (meta[0]['img_shape'][0] * num_frames,
                                  meta[0]['img_shape'][1])
                } for meta in batch_img_metas
            ]
            batch_img_metas = film_metas

        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_iou_preds, batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                return_query=False,
                ) -> Tuple[Tensor, ...]:
        """Test without augmentaton.

        Args:
            return_query:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        self.prompt_training = False
        data_sample = batch_data_samples[0]
        if isinstance(data_sample, TrackDataSample):
            img_shape = data_sample[0].metainfo['batch_input_shape']
            num_frames = len(data_sample)
            # 检查视频任务中的交互prompt
            if hasattr(data_sample, 'video_data_samples') and len(data_sample.video_data_samples) > 0:
                first_frame = data_sample.video_data_samples[0]
                if hasattr(first_frame, 'gt_instances_collected') and first_frame.gt_instances_collected is not None:
                    self.prompt_training = True
        else:
            if 'gt_instances_collected' in data_sample:
                self.prompt_training = True
            img_shape = data_sample.metainfo['batch_input_shape']
            num_frames = 0
        all_cls_scores, all_mask_preds, all_iou_preds, query_feat = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        iou_results = all_iou_preds[-1]

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

        if return_query:
            return mask_cls_results, mask_pred_results, query_feat, iou_results
        else:
            return mask_cls_results, mask_pred_results, iou_results

    def prepare_for_dn_mo(self, batch_data_samples):
        scalar, noise_scale = 100, 0.4
        gt_instances = [t.gt_instances_collected for t in batch_data_samples]

        point_coords = torch.stack([inst.point_coords for inst in gt_instances])
        # 检查并创建 pb_labels，如果不存在则使用默认值（全1）
        pb_labels_list = []
        for inst in gt_instances:
            if hasattr(inst, 'pb_labels'):
                pb_labels_list.append(inst.pb_labels)
            else:
                # 如果没有 pb_labels，创建默认值（全1，表示正样本）
                device = inst.point_coords.device
                pb_labels = torch.ones(len(inst.point_coords), dtype=torch.long, device=device)
                pb_labels_list.append(pb_labels)
        pb_labels = torch.stack(pb_labels_list)
        labels = torch.zeros_like(pb_labels).long()

        # 处理boxes：检查是否有真正的box坐标，如果没有则从点坐标创建
        # 点坐标是(x, y)，需要转换为box格式(x1, y1, x2, y2)
        has_boxes = any(hasattr(inst, 'bboxes') for inst in gt_instances)
        if has_boxes:
            # 使用实际的box坐标
            boxes = torch.stack([inst.bboxes if hasattr(inst, 'bboxes') else 
                                torch.cat([inst.point_coords, inst.point_coords], dim=-1) 
                                for inst in gt_instances])
        else:
            # 只有点坐标，将点扩展为小的box区域
            # 点(x, y) -> box(x-r, y-r, x+r, y+r)，r是一个小的半径
            radius = 5.0  # 像素半径
            boxes = torch.cat([
                point_coords - radius,  # x1, y1
                point_coords + radius   # x2, y2
            ], dim=-1)

        factors = []
        for i, data_sample in enumerate(batch_data_samples):
            h, w, = data_sample.metainfo['img_shape']
            # Box coordinates: (x1, y1, x2, y2)
            factor = boxes[i].new_tensor([w, h, w, h]).unsqueeze(0).repeat(boxes[i].size(0), 1)
            factors.append(factor)
        factors = torch.stack(factors, 0)

        # 归一化并转换为 cxcywh 格式
        boxes = bbox_xyxy_to_cxcywh(boxes / factors)
        # box_start = [t['box_start'] for t in targets]
        box_start = [len(point) for point in point_coords]

        known_labels = labels
        known_pb_labels = pb_labels
        known_bboxs = boxes

        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale > 0 and self.training:
            diff = torch.zeros_like(known_bbox_expand)
            # Box coordinates: 4维 (cx, cy, w, h)
            diff[:, :, :2] = known_bbox_expand[:, :, 2:] / 2
            diff[:, :, 2:] = known_bbox_expand[:, :, 2:]
            # add very small noise to input points; no box
            sc = 0.01
            for i, st in enumerate(box_start):
                diff[i, :st] = diff[i, :st] * sc
            
            known_bbox_expand += torch.mul(
                (torch.rand_like(known_bbox_expand) * 2 - 1.0),
                diff) * noise_scale

            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        input_label_embed = self.pb_embedding(known_pb_labels_expaned)

        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(
            self.num_mask_tokens,
            1) + self.mask_tokens.weight.unsqueeze(0).repeat(
                input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(
            self.num_mask_tokens, 1)

        single_pad = self.num_mask_tokens

        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1] / self.num_mask_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'pad_size': pad_size,
            'scalar': scalar,
        }
        return input_query_label, input_query_bbox, attn_mask, mask_dict
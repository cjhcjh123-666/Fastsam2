from mmengine.config import read_base
from torch.nn import GroupNorm, ReLU

from mmdet.models import BatchFixedSizePad, MSDeformAttnPixelDecoder, CrossEntropyLoss, \
    DiceLoss, MaskFormerFusionHead, FocalLoss
from mmdet.models.task_modules.assigners import HungarianAssigner, CrossEntropyLossCost, DiceCost
from mmdet.models.task_modules.samplers import MaskPseudoSampler

from seg.models.data_preprocessor import OVSAMVideoSegDataPreprocessor
from seg.models.utils import NO_OBJ
from seg.models.task_modules.cost import FlexibleClassificationCost
from seg.models.necks.fastsam2_neck import Fastsam2Neck
from seg.models.heads.fastsam2_head import Fastsam2VideoHead
from seg.models.detectors.fastsam2 import Fastsam2

with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.fastsam2 import *
    from .._base_.schedules.schedule_12e import *

batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=(image_size[1], image_size[0]),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=NO_OBJ
    )
]
data_preprocessor = dict(
    type=OVSAMVideoSegDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=NO_OBJ,
    batch_augments=batch_augments,
    use_point_pseudo_box=True,
    num_proposals=20,
)

num_things_classes = 136
num_stuff_classes = 101
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type=Fastsam2,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='RegNet',
        arch='regnetx_12gf',  # 升级到1.6GF模型
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='?')
    ),
    neck=dict(
        type=Fastsam2Neck,
        agg_dim=128,
        hidden_dim=256,
        backbone_shape=[224, 448, 896, 1920],
    ),
    panoptic_head=dict(
        type=Fastsam2VideoHead,
        prompt_with_kernel_updator=False,
        panoptic_with_kernel_updator=True,
        use_adaptor=True,
        use_kernel_updator=True,
        sphere_cls=True,
        ov_classifier_name='convnext_large_d_320_Concat_CocoPanopticOVDataset_YouTubeVISDataset_2019_VIPSegDataset_CityscapesPanopticDataset',
        num_stages=3,
        feat_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        # text alignment & routing & memory
        text_loss_weight=0.5,
        text_logits_weight=0.3,
        enable_dynamic_routing=True,
        early_exit=True,
        early_iou_thr=0.90,
        early_delta_thr=1e-3,
        enable_memory=True,
        memory_topk=10,
        memory_use_attention=True,
        memory_fuse_weight=0.2,
        keyframe_policy='middle',
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    
    panoptic_fusion_head=dict(
        type=MaskFormerFusionHead,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FlexibleClassificationCost, weight=2.0),
                dict(type=CrossEntropyLossCost, weight=5.0, use_sigmoid=True),
                dict(type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type=MaskPseudoSampler)),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint='work_dirs/m2_300q_convl_vlm_fix_4e_ov_obj365/epoch_4.pth'
    # )
)

val_dataloader = None
val_evaluator = None
val_cfg = None

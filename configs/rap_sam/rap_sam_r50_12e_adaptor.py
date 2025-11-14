from mmengine.config import read_base

from mmdet.models import BatchFixedSizePad, CrossEntropyLoss, DiceLoss, MaskFormerFusionHead
from mmdet.models.task_modules.assigners import HungarianAssigner, ClassificationCost, CrossEntropyLossCost, DiceCost
from mmdet.models.task_modules.samplers import MaskPseudoSampler
from mmdet.models.backbones import ResNet

from seg.models.necks.ramsam_neck import YOSONeck
from seg.models.heads.rapsam_head import RapSAMVideoHead
from seg.models.detectors.rapsam import RapSAM
from seg.models.backbones.openclip_backbone import OpenCLIPBackboneText  # Import to register

from seg.models.data_preprocessor.vid_sam_preprocessor import VideoPromptDataPreprocessor
with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.coco_panoptic_video_yt19_yt21_davis_vip_city_sam_ref import *
    from .._base_.schedules.schedule_12e import *

batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=(image_size[1], image_size[0]),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255
    )
]

data_preprocessor = dict(
    type=VideoPromptDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments
)

num_things_classes = 145
num_stuff_classes = 102
num_classes = num_things_classes + num_stuff_classes

# Multi-task components configuration
task_router = dict(
    type='TaskRouter',
    feat_channels=256,
    num_decoder_stages=3,
    enable_streaming_memory=True,
    interactive_stages=3,
    vos_stages=3,
    panoptic_stages=3
)

streaming_memory = dict(
    type='StreamingMemoryAdapter',
    feat_channels=256,
    long_mem_size=10,
    short_mem_size=5,
    update_strategy='adaptive'
)

prompt_fusion = dict(
    type='PromptFusion',
    feat_channels=256,
    num_heads=8,
    dropout=0.1,
    use_text_encoder=True,
    text_encoder=dict(
        type='TextEncoder',
        feat_channels=256,
        # text_model_cfg can be added here for CLIP text encoder
        # Example:
        text_model_cfg=dict(
            type=OpenCLIPBackboneText,
            model_name='ViT-L-14',
            init_cfg=dict(type='clip_pretrain', checkpoint='/mnt/chenjiahui/Fastsam2-main/checkpoints/openclip_vitl14_pretrain.pt')
        )
    )
)

# SAMPromptEncoder for encoding point/box prompts
prompt_encoder = dict(
    type='SAMPromptEncoder',
    model_name='vit_h',
    fix=True,
    init_cfg=dict(
        type='sam_pretrain',
        checkpoint='vit_h'  # Valid keys: 'vit_h', 'vit_l', 'vit_b'
    )
)

model = dict(
    type=RapSAM,
    data_preprocessor=data_preprocessor,
    # Multi-task configuration
    use_task_router=True,
    task_router=task_router,
    use_streaming_memory=True,
    streaming_memory=streaming_memory,
    use_prompt_fusion=True,
    prompt_fusion=prompt_fusion,
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='/mnt/chenjiahui/Fastsam2-main/checkpoints/resnet50-0676ba61.pth'),
    ),
    neck=dict(
        type=YOSONeck,
        agg_dim=128,
        hidden_dim=256,
        backbone_shape=[256, 512, 1024, 2048],
    ),
    panoptic_head=dict(
        type=RapSAMVideoHead,
        prompt_with_kernel_updator=False,
        panoptic_with_kernel_updator=True,
        use_adaptor=True,
        use_kernel_updator=True,
        prompt_encoder=prompt_encoder,
        sphere_cls=True,
        ov_classifier_name='convnext_large_d_320_Concat_CocoPanopticOVDataset_YouTubeVISDataset_2019_YouTubeVISDataset_2021_VIPSegDataset_CityscapesPanopticDataset',
        num_stages=3,
        feat_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
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
                dict(type=ClassificationCost, weight=2.0),
                dict(
                    type=CrossEntropyLossCost, weight=5.0, use_sigmoid=True),
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
)

val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None

# Multi-task model with conditional modules requires find_unused_parameters
# Some modules (e.g., TextEncoder) are only used for specific data types (RefCOCO)
# Others (e.g., StreamingMemory) are only used for video data (VOS)
# In mixed dataset training, not all parameters are used in every forward pass
find_unused_parameters = False
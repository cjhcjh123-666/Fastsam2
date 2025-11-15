"""
Video Panoptic Segmentation Evaluation Configuration  
视频全景分割：在视频序列上进行全景分割（thing + stuff）
使用 VIPSeg 或 Cityscapes-VPS 数据集
"""
from mmengine.config import read_base
from mmdet.models import BatchFixedSizePad
from mmengine.dataset import DefaultSampler
from seg.models.data_preprocessor.vidseg_data_preprocessor import VideoSegDataPreprocessor
from seg.evaluation.metrics.coco_video_metric import CocoVideoMetric
from mmcv.transforms import LoadImageFromFile
from mmdet.datasets.transforms import PackDetInputs, Resize, RandomFlip
from seg.datasets.pipelines.loading import LoadMultiImagesDirect
from seg.datasets.pipelines.loading import LoadPanopticAnnotationsAll
from seg.datasets.vipseg import VIPSegDataset

with read_base():
    from .._base_.default_runtime import *
    from .rap_sam_r50_12e_text import model

# 视频全景分割数据预处理器
data_preprocessor = dict(
    type=VideoSegDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=[
        dict(
            type=BatchFixedSizePad,
            size=(1024, 1024),
            img_pad_value=0,
            pad_mask=True,
            mask_pad_value=0,
            pad_seg=True,
            seg_pad_value=255
        )
    ]
)

# 视频全景分割 pipeline
backend_args = None
test_pipeline = [
    dict(
        type=LoadMultiImagesDirect,
        to_float32=False,
        backend_args=backend_args
    ),
    dict(
        type=LoadPanopticAnnotationsAll,
        with_bbox=True,
        with_label=True,
        with_mask=True,
        with_seg=True,
        backend_args=backend_args
    ),
    dict(type=Resize, scale=(1280, 720), keep_ratio=True),
    dict(type=PackDetInputs)
]

# VIPSeg 验证集配置
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=VIPSegDataset,
        data_root='data/VIPSeg/',
        ann_file='panoptic_gt_VIPSeg_val.json',
        data_prefix=dict(
            img_path='images/',
            seg_map_path='panoptic_masks/',
        ),
        num_ref_imgs=5,  # 每个视频序列5帧
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

# 视频全景分割评估器
val_evaluator = dict(
    type=CocoVideoMetric,
    metric=['PQ', 'SQ', 'RQ'],  # Panoptic Quality metrics
    format_only=False,
    outfile_prefix='./work_dirs/video_panoptic_results'
)
test_evaluator = val_evaluator

# 更新模型配置为视频全景分割模式
model.update(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(
        panoptic_on=True,   # 启用全景分割
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True,
    ),
)

# 日志配置
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50
    ),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/PQ',
        rule='greater'
    )
)

# VIPSeg 类别配置
# VIPSeg 有 124 个thing类别 + 58 个stuff类别
num_things_classes = 124
num_stuff_classes = 58
num_classes = num_things_classes + num_stuff_classes

# 可选：如果使用 Cityscapes-VPS，使用以下配置
# num_things_classes = 8   # Cityscapes thing classes
# num_stuff_classes = 11   # Cityscapes stuff classes
# num_classes = 19

# 备注：
# 1. 视频全景分割结合了thing（可数对象）和stuff（背景材料）的分割
# 2. 对视频序列的每一帧进行全景分割
# 3. 可以通过调整 num_ref_imgs 来改变视频序列的长度
# 4. 支持的数据集：VIPSeg, Cityscapes-VPS, KITTI-STEP等


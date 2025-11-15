"""
Video Interactive Segmentation Evaluation Configuration
视频交互式分割：在视频的每一帧应用点/框/文本提示
"""
from mmengine.config import read_base
from mmdet.models import BatchFixedSizePad
from mmengine.dataset import DefaultSampler
from seg.models.data_preprocessor.vid_sam_preprocessor import VideoPromptDataPreprocessor
from seg.evaluation.metrics.interactive_evaluation import InteractiveEvaluator
from mmcv.transforms import LoadImageFromFile
from mmdet.datasets.transforms import PackDetInputs, Resize
from seg.datasets.pipelines.loading import LoadMultiImagesDirect
from seg.datasets.pipelines.loading import LoadAnnotations
from seg.datasets.pipelines.formatting import GeneratePoint
from seg.datasets.coco_vid import CocoVideoDataset

with read_base():
    from .._base_.default_runtime import *
    from .rap_sam_r50_12e_text import model

# 视频交互分割数据预处理器
data_preprocessor = dict(
    type=VideoPromptDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=[
        dict(
            type=BatchFixedSizePad,
            size=(1024, 1024),
            img_pad_value=0,
            pad_mask=True,
            mask_pad_value=0,
        )
    ]
)

# 视频交互式分割 pipeline
# 支持在视频帧上应用点/框/文本提示
backend_args = None
test_pipeline = [
    dict(
        type=LoadMultiImagesDirect,
        to_float32=False,
        backend_args=backend_args
    ),
    dict(
        type=LoadAnnotations,
        with_bbox=True,
        with_mask=True,
        with_seg=False,
        backend_args=backend_args
    ),
    # 生成交互提示（点击）
    dict(
        type=GeneratePoint,
        num_clicks=1,  # 每个实例1次点击
        strategy='random',  # 随机点击策略
    ),
    dict(type=Resize, scale=(640, 480), keep_ratio=True),
    dict(type=PackDetInputs)
]

# COCO 视频数据集配置（用于视频交互分割测试）
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CocoVideoDataset,
        data_root='data/coco/',
        ann_file='annotations/panoptic_val2017.json',
        data_prefix=dict(img='val2017/'),
        num_ref_imgs=4,  # 每个视频序列4帧
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

# 交互式分割评估器
val_evaluator = dict(
    type=InteractiveEvaluator,
    num_tokens=1,
    format_only=False,
)
test_evaluator = val_evaluator

# 更新模型配置为视频交互式分割模式
model.update(
    data_preprocessor=data_preprocessor,
    inference_sam=True,  # 启用交互式推理模式
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,  # 交互式分割是实例分割
        max_per_image=100,
    ),
)

# 日志配置
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50
    )
)

# 类别配置
num_things_classes = 80  # COCO thing 类别
num_stuff_classes = 0    # 交互式分割不需要 stuff 类别
num_classes = num_things_classes + num_stuff_classes

# 备注：
# 1. 视频交互分割会在每一帧应用相同的提示
# 2. 支持点/框/文本混合提示
# 3. 可以通过修改 GeneratePoint 的参数来调整交互策略
# 4. 如果需要第一帧提示后跟踪，请使用 VOS 配置（eval_rap_sam_vos.py）


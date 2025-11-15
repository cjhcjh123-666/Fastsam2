"""
VOS (Video Object Segmentation) Evaluation Configuration
使用 DAVIS 2017 数据集进行 VOS 任务评估
"""
from mmengine.config import read_base
from mmdet.models import BatchFixedSizePad
from mmengine.dataset import DefaultSampler
from seg.models.data_preprocessor.vidseg_data_preprocessor import VideoSegDataPreprocessor
from seg.evaluation.metrics.vos_metric import VOSMetric
from mmcv import TransformBroadcaster, LoadImageFromFile
from mmdet.datasets.transforms import Resize, PackDetInputs
from seg.datasets.davis import DAVIS
from seg.datasets.pipelines.loading import LoadVideoSegAnnotations
from seg.datasets.pipelines.formatting import PackVidSegInputs

with read_base():
    from .._base_.default_runtime import *
    from .rap_sam_r50_12e_text import model

# VOS 数据预处理器
data_preprocessor = dict(
    type=VideoSegDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    seg_pad_value=255,
    batch_augments=[
        dict(
            type=BatchFixedSizePad,
            size=(1024, 1024),
            img_pad_value=0,
            pad_mask=True,
            mask_pad_value=0,
            pad_seg=False
        )
    ]
)

# VOS 测试 pipeline
backend_args = None
test_pipeline = [
    dict(
        type=TransformBroadcaster,
        transforms=[
            dict(type=LoadImageFromFile, backend_args=backend_args),
            dict(type=Resize, scale=(640, 480), keep_ratio=True),
            dict(
                type=LoadVideoSegAnnotations,
                with_bbox=True,
                with_label=True,
                with_mask=True,
                with_seg=False,
                backend_args=backend_args
            ),
        ]
    ),
    dict(type=PackVidSegInputs)
]

# DAVIS 2017 验证集配置
val_dataloader = dict(
    batch_size=1,  # VOS 通常使用 batch_size=1
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=DAVIS,
        dataset_version='2017',
        data_root='/mnt/chenjiahui/Fastsam2-main/data/DAVIS/',
        ann_file='ImageSets/2017/val.txt',
        data_prefix=dict(
            img='JPEGImages/480p/',
            ann='Annotations/480p/'
        ),
        pipeline=test_pipeline,
        test_mode=True,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

# VOS 评估器
val_evaluator = dict(
    type=VOSMetric,
    metric=['J', 'F', 'J&F'],  # J: Region similarity, F: Contour accuracy
    outfile_prefix='./work_dirs/vos_results'
)
test_evaluator = val_evaluator

# Val config (required by MMEngine)
val_cfg = dict(type='mmengine.runner.ValLoop')
test_cfg = dict(type='mmengine.runner.TestLoop')

# 更新模型配置为 VOS 模式
model.update(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,  # VOS 是实例分割任务
        max_per_image=100,
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
        save_best='vos_metric/J&F',
        rule='greater'
    )
)

# VOS 特定配置
# DAVIS 2017 只有 thing 类别，没有 stuff 类别
num_things_classes = 1  # VOS 将所有对象视为一个类别
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes


"""
author: chenjiahui
date: 2025-11-11
description: 新增文本交互数据集加载配置
"""
from mmengine.dataset import DefaultSampler
from seg.datasets.ref_seg import RefSegDataset

data_root = 'data/'
backend_args = None
image_size = (1024, 1024)  # 与现有管线保持一致

train_pipeline_ref = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(
        type='LoadPanopticAnnotations',  # 直接复用该变换，能解析 RLE mask
        with_bbox=True, with_mask=True, with_seg=False, backend_args=backend_args
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True
    ),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=RefSegDataset,
        data_root=data_root,
        ann_file='refcoco/converted/refcoco_train.jsonl',  # 或者指向 data/ref_seg/*.jsonl
        pipeline=train_pipeline_ref,
        backend_args=backend_args
    )
)
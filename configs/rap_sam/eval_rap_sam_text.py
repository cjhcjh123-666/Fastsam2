from mmengine.config import read_base
from mmdet.models import BatchFixedSizePad
from mmengine.dataset import DefaultSampler
from seg.models.data_preprocessor.vid_sam_preprocessor import VideoPromptDataPreprocessor
from seg.evaluation.metrics.interactive_evaluation import InteractiveEvaluator
from mmdet.datasets.transforms import LoadAnnotations, PackDetInputs, Resize
from mmcv.transforms import LoadImageFromFile
from seg.datasets.ref_seg import RefSegDataset

with read_base():
    from .._base_.default_runtime import *
    from .rap_sam_r50_12e_text import model

# 批次增强
batch_augments = [
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

# 数据预处理器
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

# 测试 pipeline（简化版，用于推理）
backend_args = None
test_pipeline = [
    dict(type=LoadImageFromFile, to_float32=True, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True, with_seg=False, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=PackDetInputs)
]

# RefCOCO 验证集配置
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=RefSegDataset,
        data_root='data/ref_seg/',
        ann_file='refcoco',  # 数据集目录名
        dataset_name='refcoco',  # 使用 REFER API
        split_by='unc',  # refcoco 使用 'unc' split
        split='val',  # 验证集
        max_sentences_per_image=1,  # 推理时每个图像只用1个句子
        debug=False,
        # 使用 COCO 2017 图像（代码会自动转换 COCO 2014 文件名）
        data_prefix=dict(img='/mnt/chenjiahui/Fastsam2-main/data/coco/train2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

# 评估器
val_evaluator = [
    dict(
        type=InteractiveEvaluator,
        num_tokens=1,
        #format_only=False,
    )
]
test_evaluator = val_evaluator

# Val config (required by MMEngine)
val_cfg = dict(type='mmengine.runner.ValLoop')
test_cfg = dict(type='mmengine.runner.TestLoop')

# 更新模型配置
model.update(
    data_preprocessor=data_preprocessor,
    inference_sam=True,  # 启用交互式推理模式
    test_cfg=dict(
        panoptic_on=False,  # 关闭全景分割
        semantic_on=False,
        instance_on=True,   # 只做实例分割
        max_per_image=100,
    ),
)

# COCO 类别配置（RefCOCO 基于 COCO）
num_things_classes = 80
num_stuff_classes = 0  # RefCOCO 只有 thing 类别
num_classes = num_things_classes + num_stuff_classes


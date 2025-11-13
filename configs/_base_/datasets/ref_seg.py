"""
author: chenjiahui
date: 2025-11-11
description: 新增文本交互数据集加载配置
支持 REFER API 加载方式（refcoco, refcoco+, refcocog, refclef）
也支持直接加载 COCO 格式 JSON + refs.p
"""
from mmengine.dataset import DefaultSampler
from mmcv.transforms import LoadImageFromFile
from mmdet.datasets.transforms import LoadAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize
from seg.datasets.ref_seg import RefSegDataset
from seg.datasets.pipelines.formatting import GeneratePoint

data_root = 'data/ref_seg/'  # RefCOCO 数据集的根目录
# 注意：RefCOCO 数据集使用 COCO 2014 图像，但如果您只有 COCO 2017 图像，
# 代码会自动将文件名从 COCO_train2014_000000098304.jpg 转换为 000000098304.jpg
# 因此可以使用 COCO 2017 的图像目录
backend_args = None
image_size = (1280, 736)  # 与现有管线保持一致

train_pipeline_ref = [
    dict(type=LoadImageFromFile, to_float32=True, backend_args=backend_args),
    dict(
        type=LoadAnnotations,  # 使用标准的 LoadAnnotations，能解析 RLE mask
        with_bbox=True, with_mask=True, with_seg=False, backend_args=backend_args
    ),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Resize, scale=image_size, keep_ratio=True),
    dict(
        type=RandomCrop,
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True
    ),
    dict(type=PackDetInputs,
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','text')),
    # generate point prompts from gt masks to align with current prompt pipeline
    dict(type='GeneratePoint', num_proposals=30, num_mask_tokens=1)
]

# 使用 REFER API 加载 RefCOCO 数据集
# 数据集目录结构应该是:
# data/ref_seg/
#   ├── refcoco/
#   │   ├── refs(unc).p
#   │   └── instances.json
#   ├── refcoco+/
#   │   ├── refs(unc).p
#   │   └── instances.json
#   ├── refcocog/
#   │   ├── refs(umd).p  # 或 refs(google).p
#   │   └── instances.json
#   └── images/
#       └── mscoco/
#           └── images/
#               └── train2014/

# RefCOCO 数据集配置
# 如果使用 COCO 2017 图像（推荐，因为代码会自动转换文件名）:
refcoco_dataset = dict(
    type=RefSegDataset,
    data_root=data_root,
    ann_file='refcoco',  # 数据集目录名
    dataset_name='refcoco',  # 使用 REFER API
    split_by='unc',  # refcoco 使用 'unc' split
    split='train',  # 训练集
    max_sentences_per_image=3,  # 每个图像最多采样3个句子
    debug=False,
    # 使用 COCO 2017 图像（代码会自动将 COCO 2014 文件名转换为 COCO 2017 格式）
    data_prefix=dict(img='/mnt/chenjiahui/Fastsam2-main/data/coco/train2017/'),  # 绝对路径
    pipeline=train_pipeline_ref,
    backend_args=backend_args
)

# 如果您有 COCO 2014 图像，可以使用以下配置：
# refcoco_dataset = dict(
#     type=RefSegDataset,
#     data_root=data_root,
#     ann_file='refcoco',
#     dataset_name='refcoco',
#     split_by='unc',
#     split='train',
#     max_sentences_per_image=3,
#     debug=False,
#     data_prefix=dict(img='images/mscoco/images/train2014/'),  # COCO 2014 图像路径
#     pipeline=train_pipeline_ref,
#     backend_args=backend_args
# )

# RefCOCO+ 数据集配置（使用 COCO 2017 图像）
refcoco_plus_dataset = dict(
    type=RefSegDataset,
    data_root=data_root,
    ann_file='refcoco+',
    dataset_name='refcoco+',
    split_by='unc',
    split='train',
    max_sentences_per_image=3,
    debug=False,
    data_prefix=dict(img='/mnt/chenjiahui/Fastsam2-main/data/coco/train2017/'),  # 使用 COCO 2017 图像
    pipeline=train_pipeline_ref,
    backend_args=backend_args
)

# RefCOCOg 数据集配置（使用 COCO 2017 图像）
refcocog_dataset = dict(
    type=RefSegDataset,
    data_root=data_root,
    ann_file='refcocog',
    dataset_name='refcocog',
    split_by='umd',  # refcocog 使用 'umd' 或 'google' split
    split='train',
    max_sentences_per_image=3,
    debug=False,
    data_prefix=dict(img='/mnt/chenjiahui/Fastsam2-main/data/coco/train2017/'),  # 使用 COCO 2017 图像
    pipeline=train_pipeline_ref,
    backend_args=backend_args
)

# RefCLEF 数据集配置（图像在 saiapr_tc-12 目录下）
refclef_dataset = dict(
    type=RefSegDataset,
    data_root=data_root,
    ann_file='refclef',
    dataset_name='refclef',
    split_by='unc',
    split='train',
    max_sentences_per_image=3,
    debug=False,
    data_prefix=dict(img='saiapr_tc-12/'),  # RefCLEF 图像在 saiapr_tc-12 目录下
    pipeline=train_pipeline_ref,
    backend_args=backend_args
)

# 默认使用 RefCOCO 数据集
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=refcoco_dataset
)

# #如果需要合并多个数据集，可以使用 ConcatOVDataset:
# from seg.datasets.concat_dataset import ConcatOVDataset
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type=DefaultSampler, shuffle=True),
#     dataset=dict(
#         type=ConcatOVDataset,
#         datasets=[
#             refcoco_dataset,
#             refcoco_plus_dataset,
#             refcocog_dataset,
#             refclef_dataset,
#         ]
#     )
# )
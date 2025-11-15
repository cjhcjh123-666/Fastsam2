# dataset settings
from mmengine import read_base
from mmengine.dataset import DefaultSampler, RepeatDataset

from seg.datasets.concat_dataset import ConcatOVDataset
from seg.datasets.samplers.batch_sampler import VideoSegAspectRatioBatchSampler

# 导入所有数据集的train_dataloader配置
with read_base():
    from .coco_panoptic_video_lsj import train_dataloader as _coco_vid_train_loader
    from .youtube_vis_2019 import train_dataloader as _yt19_train_loader
    from .youtube_vis_2021 import train_dataloader as _yt21_train_loader
    from .davis import train_dataloader as _davis_train_loader
    from .vipseg import train_dataloader as _vip_train_loader
    from .cityscapes_panoptic_720p import train_dataloader as _city_train_loader

from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler

from mmdet.datasets.transforms import LoadPanopticAnnotations, LoadAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize

from seg.datasets.coco_ov import CocoPanopticOVDataset
from seg.datasets.ref_seg import RefSegDataset
from seg.datasets.pipelines.loading import FilterAnnotationsHB
from seg.datasets.pipelines.formatting import GeneratePoint
from seg.datasets.pipelines.loading import LoadPanopticAnnotationsAll

data_root = 'data/coco/'
backend_args = None
image_size = (1280, 736)

num_mask_tokens = 1

# SAM数据集配置 (用于类别无关的分割任务)
sam_train_pipeline = [
    dict(type=LoadImageFromFile, to_float32=True, backend_args=backend_args),
    dict(type=LoadPanopticAnnotationsAll),
    dict(type=RandomFlip, prob=0.5),
    dict(type=RandomResize, resize_type=Resize, scale=image_size, ratio_range=(0.1, 2.0), keep_ratio=True),
    dict(type=RandomCrop, crop_size=image_size, crop_type='absolute', recompute_bbox=True, allow_negative_crop=True),
    dict(type=FilterAnnotationsHB, by_box=False, by_mask=True, min_gt_mask_area=32),
    dict(type=PackDetInputs),
    dict(type=GeneratePoint, num_proposals=30, num_mask_tokens=num_mask_tokens)
]

sam_dataset = dict(
    type=CocoPanopticOVDataset,
    data_root=data_root,
    ann_file='annotations/panoptic_train2017.json',
    data_prefix=dict(img='train2017/', seg='annotations/panoptic_train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sam_train_pipeline,
    backend_args=backend_args
)

# 联合数据集配置
# 包含所有8个数据集: coco, yt19, yt21, davis, vip, city, sam, refcoco
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=VideoSegAspectRatioBatchSampler),
    dataset=dict(
        type=ConcatOVDataset,
        # data_tag与datasets列表顺序对应，用于标识数据来源
        data_tag=('coco', 'yt21', 'davis','sam'),
        datasets=[
            # 1. COCO Panoptic Video数据集
            dict(
                type=RepeatDataset,
                dataset=_coco_vid_train_loader.dataset,
                times=1,
            ),
            # 3. YouTube VIS 2021数据集
            dict(
                type=RepeatDataset,
                dataset=_yt21_train_loader.dataset,
                times=25,
            ),
            # 4. DAVIS数据集
            dict(
                type=RepeatDataset,
                dataset=_davis_train_loader.dataset,
                times=10,
            ),
            # 7. SAM数据集 (类别无关分割)
            dict(
                type=RepeatDataset,
                dataset=sam_dataset,
                times=1,
            ),
        ]
    ),
)